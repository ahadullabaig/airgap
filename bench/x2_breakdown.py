"""X2: where the time goes in attention at the encoder's shape (12 heads, N = 1024, D = 64, FP32, this GPU).
Assembles measured files into results/attention/x2_breakdown.json; it times nothing itself.

Definitions. These were written before the numbers below were assembled, but after V1-V3 had been timed, so
the criteria are not blind.
- A version's time: its committed benchmark median (results/attention/v1.json, v2.json, v3.json, op
  v*_h12_n1024, event_batch_ms). Each version ran in its own group, at its own steady clock. Phase times are
  that op's phase medians.
- Fixed cost of a phase: an empty kernel with that phase's grid, block size and shared-memory request,
  launched back to back (results/attention/x2.json, kernels/attention/x2_bench.cu). The rest of the phase is
  "work". Fixed cost and work add up to the phase time, and the phases add up to the version's time (within
  the harness's phase-versus-whole difference, reported). Nothing else is stacked. The stacked fixed cost is
  the sum of the per-phase empty launches, each timed alone; it is larger than the empty launches timed back
  to back as one call (also reported), because consecutive launches overlap their overheads.
- Floors are marked, never stacked, because compute and DRAM traffic overlap in time.
  - Compute floor: the phase's FLOPs at the measured FP32 FMA peak, 7.46 TFLOPS at about 1.9 GHz
    (results/cuda/peak.json), and the same rescaled to the version's own clock (its group's median SM clock).
  - DRAM floor: the bytes the kernel moved to and from DRAM (Nsight Compute, results/attention/x2_ncu_*.txt)
    at the measured 188 GB/s.
  - A phase's floor is the larger of its compute floor at its own clock and its DRAM floor; within 15% of
    each other they are reported as a near tie.
  - The DRAM floor uses the bytes each kernel actually moved, so it is a floor for that kernel, not for the
    phase's job: V1's softmax floor includes its two avoidable extra reads of the N x N matrix.
- Stall shares: per kernel, each warp-stall reason divided by the sum of all reasons (Nsight Compute, in warp
  cycles per issued instruction). The sum is checked against smsp__average_warp_latency_per_inst_issued.
  The shares describe; they do not test a cause.
- Interventions: the sweeps' alternative tile shapes, timed in v2_tune.json and v3_tune.json and profiled in
  x2_ncu_v2alt.txt and x2_ncu_v3alt.txt. A cause counts as tested only when an alternative changes that
  cause. Anything else is reported as "consistent with". The clock has no intervention: pinning it needs root.
- Transfers: copying Q, K and V to the device and O back (x2.json). This is a reference for an attention call
  fed from the host, not part of any version's time; in the encoder Q, K and V never leave the GPU.
- H1, clause 1 ("limited by compute and memory traffic, not by fixed per-launch costs") holds if every
  version's fixed cost is below 5% of its time.
- H1, clause 2 ("fusion helps mainly by removing the N x N traffic") holds if V3's saving over V2 is at least
  half of V2's N x N DRAM floor, both in time and with V2 brought to V3's clock. V2's N x N bytes are its
  kernels' DRAM bytes minus V3's (the Q, K, V and O traffic both must move). Bringing V2 to V3's clock is
  bracketed: scaling all of V2's time by the clock ratio (a lower bound on the saving, since the softmax is
  DRAM-bound and would not speed up), and scaling only its two matmuls.

Example:
    .venv/bin/python -I bench/x2_breakdown.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "attention"
H, N, D = 12, 1024, 64
MM_FLOPS = 2.0 * H * N * N * D  # one of Q K^T or P V
FIXED_SHARE_MAX = 0.05
NXN_SAVING_MIN = 0.5
STALLS = ["barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss", "lg_throttle", "long_scoreboard",
          "math_pipe_throttle", "membar", "mio_throttle", "misc", "no_instruction", "not_selected", "selected",
          "short_scoreboard", "sleeping", "tex_throttle", "wait"]


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def ncu_kernels(name: str) -> list[dict]:
    """Kernels in an Nsight Compute text file, in launch order: {"kernel": name, metric: value (MB for bytes)}."""
    out = []
    for line in (RES / name).read_text().splitlines():
        m = re.match(r"\s+(?:void )?<unnamed>::(\w+)(<[^>]*>)?\(", line)
        if m:
            out.append({"kernel": m.group(1) + (m.group(2) or "")})
            continue
        parts = line.split()
        if out and len(parts) >= 2 and "__" in parts[0]:
            unit = parts[1] if len(parts) == 3 else ""
            val = float(parts[-1].replace(",", ""))
            if parts[0].startswith("dram__bytes"):
                val *= {"Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3, "byte": 1e-6}.get(unit, 1.0)
            out[-1][parts[0]] = val
    return out


def stall_shares(k: dict) -> dict:
    vals = {r: k[f"smsp__average_warps_issue_stalled_{r}_per_issue_active.ratio"] for r in STALLS}
    total = sum(vals.values())
    return {"cycles_per_issue": k["smsp__average_warp_latency_per_inst_issued.ratio"], "sum_of_reasons": total,
            "shares": {r: v / total for r, v in sorted(vals.items(), key=lambda x: -x[1]) if v > 0}}


def kernel_summary(k: dict) -> dict:
    return {"kernel": k["kernel"], "dram_mb": k["dram__bytes_read.sum"] + k["dram__bytes_write.sum"],
            "fma_pipe_pct": k["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"],
            "warps_active_pct": k["sm__warps_active.avg.pct_of_peak_sustained_active"],
            "theoretical_occupancy_pct": k["sm__maximum_warps_per_active_cycle_pct"],
            "registers": k["launch__registers_per_thread"],
            "blocks_per_sm_limit": {x: k[f"launch__occupancy_limit_{x}"] for x in
                                    ("registers", "shared_mem", "warps", "blocks")},
            "waves_per_sm": k["launch__waves_per_multiprocessor"],
            "shared_conflicts_load": k["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"],
            "shared_conflicts_store": k["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"],
            "stalls": stall_shares(k)}


def main() -> None:
    pk = json.loads((ROOT / "results" / "cuda" / "peak.json").read_text())
    peak_tflops = pk["fma_sustained"]["tflops_settled"]
    peak_mhz = sum(pk["fma_sustained"]["sm_mhz_per_second"]) / len(pk["fma_sustained"]["sm_mhz_per_second"])
    bw = pk["dram_read"]["gb_per_s"]
    x2 = {o["name"]: o for o in load("x2.json")["ops"]}
    med = lambda o, k: o["clocks"][k]["median"]  # noqa: E731

    phase_spec = {  # version -> [(phase, FLOPs, kernel index in its x2_ncu file)]
        "V1": [("scores", MM_FLOPS, 0), ("softmax", 0.0, 1), ("output", MM_FLOPS, 2)],
        "V2": [("scores", MM_FLOPS, 0), ("softmax", 0.0, 1), ("output", MM_FLOPS, 2)],
        "V3": [("fused", 2 * MM_FLOPS, 0)],
    }
    versions = {}
    for ver, phases in phase_spec.items():
        v = ver.lower()
        doc = load(f"{v}.json")
        op = {o["name"]: o for o in doc["ops"]}[f"{v}_h12_n1024"]
        mhz = [g["median_sm_mhz"] for g in doc["groups"] if g["group"] == op["group"]][0]
        empty = x2[f"empty_{v}"]
        kernels = ncu_kernels(f"x2_ncu_{v}.txt")
        total = med(op, "event_batch_ms")
        rows = []
        for name, flops, ki in phases:
            t = med(op, f"phase_{name}_ms") if len(phases) > 1 else total
            fixed = med(empty, f"phase_{name}_ms") if len(phases) > 1 else med(empty, "event_batch_ms")
            ks = kernel_summary(kernels[ki])
            c19 = flops / (peak_tflops * 1e12) * 1e3
            c_own = c19 * peak_mhz / mhz
            dram = ks["dram_mb"] * 1e6 / (bw * 1e9) * 1e3
            floor = max(c_own, dram)
            rows.append({"phase": name, "time_ms": t, "fixed_ms": fixed, "work_ms": t - fixed, "gflop": flops / 1e9,
                         "compute_floor_ms_at_1900mhz": c19, "compute_floor_ms_at_own_clock": c_own,
                         "dram_floor_ms": dram, "floor_ms": floor,
                         "floor_kind": ("DRAM = compute" if 0.85 <= c_own / max(dram, 1e-12) <= 1.15 else
                                        "compute" if c_own > dram else "DRAM"),
                         "time_over_floor": t / floor if floor > 0 else None, **ks})
        phase_sum = sum(r["time_ms"] for r in rows)
        fixed = sum(r["fixed_ms"] for r in rows)
        versions[ver] = {"time_ms": total, "median_sm_mhz": mhz, "megacycles": total * 1e-3 * mhz,
                         "phase_sum_ms": phase_sum, "phase_sum_minus_time_ms": phase_sum - total,
                         "fixed_ms": fixed, "fixed_share": fixed / total,
                         "fixed_ms_back_to_back_whole": med(empty, "event_batch_ms"),
                         "fixed_ms_with_sync": med(empty, "host_sync_ms"),
                         "dram_mb": sum(r["dram_mb"] for r in rows),
                         "floor_ms": sum(r["floor_ms"] for r in rows), "phases": rows}

    # interventions: the sweeps' alternative tile shapes, timed in the same group as the defaults and profiled
    t2 = {o["params"]["config"]: med(o, "event_batch_ms") for o in load("v2_tune.json")["ops"]
          if o["params"].get("phase") in ("scores", "output")}
    t2_phase = {(o["params"]["phase"], o["params"]["config"]): med(o, "event_batch_ms")
                for o in load("v2_tune.json")["ops"] if o["params"].get("phase") in ("scores", "output")}
    t3 = {o["params"]["config"]: med(o, "event_batch_ms") for o in load("v3_tune.json")["ops"]}
    k2, k2a = ncu_kernels("x2_ncu_v2.txt"), ncu_kernels("x2_ncu_v2alt.txt")
    k3, k3a = ncu_kernels("x2_ncu_v3.txt"), ncu_kernels("x2_ncu_v3alt.txt")
    # bound on V2's store conflicts: conflicts against all shared-memory wavefronts (results/attention/v2_ncu.txt,
    # the same default scores kernel) and the share of warp cycles spent waiting on MIO at all
    conflicts = kernel_summary(k2[0])["shared_conflicts_store"]
    wavefronts = ncu_kernels("v2_ncu.txt")[0]["l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"]
    sh = kernel_summary(k2[0])["stalls"]["shares"]
    mio_share = sh.get("short_scoreboard", 0.0) + sh.get("mio_throttle", 0.0)
    interventions = [
        {"cause": "shared-memory store conflicts (V2 scores)",
         "default": {"config": "64x64x16/8x4", "time_ms": t2_phase[("scores", "64x64x16/8x4")],
                     **{x: kernel_summary(k2[0])[x] for x in ("shared_conflicts_store", "warps_active_pct", "fma_pipe_pct")}},
         "alternative": {"config": "64x64x8/4x4", "time_ms": t2_phase[("scores", "64x64x8/4x4")],
                         **{x: kernel_summary(k2a[0])[x] for x in ("shared_conflicts_store", "warps_active_pct", "fma_pipe_pct")}},
         "tested": False,
         "bound": {"store_conflicts": conflicts, "shared_wavefronts": wavefronts,
                   "conflicts_share_of_wavefronts": conflicts / wavefronts, "mio_share_of_warp_cycles": mio_share},
         "reading": f"bounded rather than tested: the conflicts are {100 * conflicts / wavefronts:.0f}% of the "
                    f"kernel's shared-memory wavefronts, and all waiting on MIO is {100 * mio_share:.1f}% of its warp "
                    "cycles, so they cost at most a few percent. The conflict-free alternative was slower, but it "
                    "also changes the register tile (4x4 instead of 8x4), so that comparison is confounded"},
        {"cause": "occupancy (V2 output)",
         "default": {"config": "64x64x16/8x4", "time_ms": t2_phase[("output", "64x64x16/8x4")],
                     **{x: kernel_summary(k2[2])[x] for x in ("warps_active_pct", "theoretical_occupancy_pct", "registers", "fma_pipe_pct")}},
         "alternative": {"config": "64x64x16/4x4", "time_ms": t2_phase[("output", "64x64x16/4x4")],
                         **{x: kernel_summary(k2a[2])[x] for x in ("warps_active_pct", "theoretical_occupancy_pct", "registers", "fma_pipe_pct")}},
         "tested": False,
         "reading": "confounded: the alternative has twice the occupancy and is no faster, but it also halves each "
                    "thread's register tile, and the two effects could cancel; occupancy is not isolated"},
        {"cause": "rows per block at equal occupancy (V3)",
         "default": {"config": "128x32/256", "time_ms": t3["128x32/256"],
                     **{x: kernel_summary(k3[0])[x] for x in ("warps_active_pct", "theoretical_occupancy_pct", "fma_pipe_pct")},
                     "short_scoreboard_share": kernel_summary(k3[0])["stalls"]["shares"].get("short_scoreboard")},
         "alternative": {"config": "64x32/128", "time_ms": t3["64x32/128"],
                         **{x: kernel_summary(k3a[0])[x] for x in ("warps_active_pct", "theoretical_occupancy_pct", "fma_pipe_pct")},
                         "short_scoreboard_share": kernel_summary(k3a[0])["stalls"]["shares"].get("short_scoreboard")},
         "tested": False,
         "reading": "not an occupancy test: both shapes hold 8 warps per SM (one 256-thread block or two 128-thread "
                    "blocks). No swept V3 shape raises occupancy, because shared memory limits every one, so "
                    "occupancy is untested for V3; its leading stall (waiting on MIO results: shared-memory "
                    "loads, which dominate its inner loops, plus shuffles and exp) is consistent with too few "
                    "warps to hide that latency"},
        {"cause": "clock (software power cap)", "tested": False,
         "reading": "no intervention: pinning the clock needs root. V1 ran at about 1.95 GHz (reliability-voltage "
                 "limit), V2 at about 1.5 GHz and V3 at about 1.78 GHz, both under the power cap"},
    ]
    del t2

    b1 = {o["name"]: o for o in load("b1.json")["ops"]}
    b1_ref = {k: {"time_ms": med(b1[f"b1_{k}_h12_n1024"], "event_batch_ms"),
                  "host_ms": med(b1[f"b1_{k}_h12_n1024"], "host_sync_ms")} for k in ("math", "efficient_attention")}
    transfers = {n: {"event_ms": med(x2[n], "event_batch_ms"), "host_ms": med(x2[n], "host_sync_ms"),
                     "mb": float(x2[n]["params"]["bytes"]) / 1e6} for n in
                 ("h2d_qkv_pinned", "d2h_o_pinned", "h2d_qkv_pageable", "d2h_o_pageable")}
    for t in transfers.values():
        t["gb_per_s"] = t["mb"] / 1e3 / (t["host_ms"] / 1e3)

    v2, v3 = versions["V2"], versions["V3"]
    nxn_mb = v2["dram_mb"] - v3["dram_mb"]
    nxn_floor = nxn_mb * 1e6 / (bw * 1e9) * 1e3
    saving_ms = v2["time_ms"] - v3["time_ms"]
    saving_cycles_share = 1 - v3["megacycles"] / v2["megacycles"]
    ratio = v2["median_sm_mhz"] / v3["median_sm_mhz"]
    v2_matmuls_scaled = (sum(r["time_ms"] * (ratio if r["phase"] in ("scores", "output") else 1.0)
                             for r in v2["phases"]))
    saving_matmul_scaled = v2_matmuls_scaled - v3["time_ms"]
    saving_all_scaled = saving_cycles_share * v2["time_ms"]
    h1 = {
        "clause1": {"criterion": f"every version's fixed cost below {FIXED_SHARE_MAX:.0%} of its time",
                    "fixed_share": {k: v["fixed_share"] for k, v in versions.items()},
                    "holds": all(v["fixed_share"] < FIXED_SHARE_MAX for v in versions.values())},
        "clause2": {"criterion": f"V3's saving over V2 at least {NXN_SAVING_MIN:.0%} of V2's N x N DRAM floor, in time "
                                 "and in clock cycles",
                    "v2_nxn_mb": nxn_mb, "v2_nxn_dram_floor_ms": nxn_floor,
                    "saving_ms": saving_ms, "saving_share_of_nxn_floor": saving_ms / nxn_floor,
                    "saving_at_v3_clock_ms": [saving_all_scaled, saving_matmul_scaled],
                    "saving_at_v3_clock_note": "V2 brought to V3's clock: all of V2 scaled (a lower bound), or only "
                                               "its matmuls scaled",
                    "holds": saving_ms >= NXN_SAVING_MIN * nxn_floor
                    and saving_matmul_scaled >= NXN_SAVING_MIN * nxn_floor},
    }
    stall_check = {f"{ver}/{r['phase']}": r["stalls"]["sum_of_reasons"] / r["stalls"]["cycles_per_issue"]
                   for ver, v in versions.items() for r in v["phases"]}
    out = {"experiment": "X2: where the time goes at 12 heads, N = 1024, D = 64 (definitions: bench/x2_breakdown.py)",
           "peak": {"fma_tflops": peak_tflops, "fma_mhz": peak_mhz, "dram_gb_per_s": bw,
                    "launch_b2b_us": pk["launch_b2b"]["ms_median"] * 1e3, "launch_sync_us": pk["launch_rt"]["ms_median"] * 1e3},
           "empty_1block_us": {"back_to_back": med(x2["empty_1block"], "event_batch_ms") * 1e3,
                               "with_sync": med(x2["empty_1block"], "host_sync_ms") * 1e3},
           "versions": versions, "b1_reference": b1_ref, "transfers": transfers, "interventions": interventions,
           "stall_sum_over_reported_total": stall_check, "h1": h1}
    (RES / "x2_breakdown.json").write_text(json.dumps(out, indent=1) + "\n")

    for ver, v in versions.items():
        print(f"{ver}: {v['time_ms']:.3f} ms at {v['median_sm_mhz']} MHz, fixed {1e3 * v['fixed_ms']:.1f} us "
              f"({100 * v['fixed_share']:.2f}%), DRAM {v['dram_mb']:.0f} MB, floor {v['floor_ms']:.3f} ms")
        for r in v["phases"]:
            top = list(r["stalls"]["shares"].items())[:3]
            print(f"   {r['phase']:8s} {r['time_ms']:.3f} ms  floor {r['floor_ms']:.3f} ({r['floor_kind']}, x{r['time_over_floor']:.1f})"
                  f"  FMA {r['fma_pipe_pct']:.0f}%  warps {r['warps_active_pct']:.0f}%  stalls "
                  + ", ".join(f"{k} {100 * s:.0f}%" for k, s in top))
    print("stall sums / reported:", {k: round(x, 3) for k, x in stall_check.items()})
    print("H1 clause 1:", h1["clause1"]["holds"], "| clause 2:", h1["clause2"]["holds"],
          f"(saving {saving_ms:.3f} ms; N x N floor {nxn_floor:.3f} ms; at V3's clock {saving_all_scaled:.3f}-"
          f"{saving_matmul_scaled:.3f})")
    for i in interventions[:3]:
        print(i["cause"], "tested" if i["tested"] else "not tested", i.get("bound", ""), "|", {k: i["default"][k] for k in i["default"]}, "->", {k: i["alternative"][k] for k in i["alternative"]})
    print("wrote", RES / "x2_breakdown.json")


if __name__ == "__main__":
    main()

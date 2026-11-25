"""Quantities the paper derives from several result files, shared by make_numbers.py and figures.py so the text
and the figures cannot disagree.

pinned_breakdown(): the time breakdown at the pinned clock (the paper's headline timing set). For each version
at 12 heads, N = 1024:
- Phase times: the pinned run's phase medians (results/attention/clock1395/v1.json, v2.json; V3 is one kernel).
- Fixed cost per phase: an empty kernel with that phase's launch configuration (results/attention/x2.json,
  timed unpinned; launch costs are microseconds either way).
- Floors, as in bench/x2_breakdown.py but at the pinned clock:
  - compute floor: the phase's FLOPs at the FMA peak rescaled to the pinned clock;
  - DRAM floor: the bytes the kernel moved (Nsight Compute, via results/attention/x2_breakdown.json) at the
    measured DRAM bandwidth.
  The floor is the larger of the two; within 15% of each other they are a near tie.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PEAK_MHZ = 1900.0


def _load(rel: str) -> dict:
    return json.loads((RES / rel).read_text())


def pinned_breakdown() -> dict:
    pk = _load("cuda/peak.json")
    pin = _load("attention/clock1395/summary.json")
    pin_mhz = pin["pinned_mhz"]
    peak = pk["fma_sustained"]["tflops_settled"] * 1e12 * pin_mhz / PEAK_MHZ
    bw = pk["dram_read"]["gb_per_s"] * 1e9
    x2 = _load("attention/x2_breakdown.json")["versions"]
    empty = {o["name"]: o for o in _load("attention/x2.json")["ops"]}
    out = {}
    for v in ("V1", "V2", "V3"):
        r = pin["ops"][v]
        rows = []
        for ph in x2[v]["phases"]:
            name = ph["phase"]
            t = r["phases_ms"][name] if name in r["phases_ms"] else r["time_ms"]
            e = empty[f"empty_{v.lower()}"]["clocks"]
            fixed = e[f"phase_{name}_ms"]["median"] if f"phase_{name}_ms" in e else e["event_batch_ms"]["median"]
            compute = ph["gflop"] * 1e9 / peak * 1e3
            dram = ph["dram_mb"] * 1e6 / bw * 1e3
            floor = max(compute, dram)
            kind = ("DRAM = compute" if dram > 0 and 0.85 <= compute / dram <= 1.15 else
                    "compute" if compute > dram else "DRAM")
            rows.append({"phase": name, "time_ms": t, "fixed_ms": fixed, "compute_floor_ms": compute,
                         "dram_floor_ms": dram, "floor_ms": floor, "floor_kind": kind, "over_floor": t / floor,
                         "stalls": ph["stalls"]["shares"], "fma_pipe_pct": ph["fma_pipe_pct"],
                         "warps_active_pct": ph["warps_active_pct"]})
        fixed = sum(x["fixed_ms"] for x in rows)
        out[v] = {"time_ms": r["time_ms"], "phases": rows, "fixed_ms": fixed, "fixed_share": fixed / r["time_ms"],
                  "floor_ms": sum(x["floor_ms"] for x in rows)}
    return {"pinned_mhz": pin_mhz, "versions": out}


if __name__ == "__main__":
    d = pinned_breakdown()
    for v, r in d["versions"].items():
        print(v, f"{r['time_ms']:.3f} ms, fixed {1e3 * r['fixed_ms']:.1f} us ({100 * r['fixed_share']:.2f}%)",
              [(x["phase"], round(x["time_ms"], 3), round(x["floor_ms"], 3), x["floor_kind"], round(x["over_floor"], 1))
               for x in r["phases"]])

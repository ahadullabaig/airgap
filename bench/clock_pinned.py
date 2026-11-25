"""Attention at a pinned SM clock: V1, V2, V3 and B1 at 12 heads, N = 1024, with the GPU's graphics clock
locked (nvidia-smi -lgc, set by the user with root), compared at equal clock and against predictions
rescaled to that clock. Assembles results/attention/clock1395/*.json into summary.json; it times nothing.

Why: unpinned, each version settles at its own clock under the software power cap (V1 about 1.95 GHz, V2 1.51,
V3 1.78, B1 1.69), so wall-time comparisons and predicted-versus-measured checks mix the kernel with the clock.
1395 MHz is the lowest clock any version's samples showed unpinned (V2), so every version should hold it.

Definitions, written before the pinned run:
- Pin held: at least 90% of an op's samples within 1% of the pinned clock (per-sample SM clock from the
  harness). Ops where the pin did not hold are reported but not compared.
- Predictions at the pin: bench/predict_times.py's dense formulas with the FP32 FMA peak and the L2 bandwidth
  scaled by pinned / 1900 MHz (both run on the core clock; the peak was measured at about 1.9 GHz), DRAM
  bandwidth and launch cost unscaled (the memory clock is not pinned). At the unscaled peak the same formulas
  reproduce results/cuda/predicted_times.json (asserted).
- For each version: measured time against the rescaled range (inside, above or below).
- V3 against V2 at equal clock: the saving, as a share of V2's N x N DRAM floor (results/attention/
  x2_breakdown.json). H1's second clause holds at equal clock if that share is at least 0.5.
- Clock sensitivity: pinned time x pinned clock against unpinned time x unpinned median clock (megacycles). A
  ratio near 1 means the op scales with the SM clock; DRAM-bound work does not.

Example:
    .venv/bin/python -I bench/clock_pinned.py [--mhz 1395]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "attention"
sys.path.insert(0, str(ROOT / "bench"))
import predict_times as pt  # noqa: E402

N, D, H, B = pt.N, pt.D, pt.H, pt.B
PEAK_MHZ = 1900.0
HOLD_SHARE, HOLD_TOL = 0.9, 0.01
OPS = {"V1": ("v1.json", "v1_h12_n1024"), "V2": ("v2.json", "v2_h12_n1024"), "V3": ("v3.json", "v3_h12_n1024"),
       "B1 math": ("b1.json", "b1_math_h12_n1024"),
       "B1 memory-efficient": ("b1.json", "b1_efficient_attention_h12_n1024")}


def dense(P: float, BW: float, L2: float, launch: float) -> dict:
    """bench/predict_times.py's dense predictions (seconds, (low, high)) for the given limits."""
    f_mm = 2 * N * N * D * H
    s_bytes = N * N * H * B
    qkvo = 4 * N * D * H * B
    operand_l2 = N * N * H * 2 * D * B / pt.L1_REUSE
    t = lambda flops, eff: flops / (eff * P)  # noqa: E731
    E = pt.EFF
    v1 = (max(t(f_mm, E["naive_matmul"][1]), s_bytes / BW) + 2 * s_bytes / BW
          + max(t(f_mm, E["naive_matmul"][1]), s_bytes / BW) + 3 * launch,
          max(t(f_mm, E["naive_matmul"][0]), operand_l2 / L2) + 2 * s_bytes / BW
          + max(t(f_mm, E["naive_matmul"][0]), operand_l2 / L2) + 3 * launch)
    v2 = tuple(2 * max(t(f_mm, e), s_bytes / BW) + 2 * s_bytes / BW + 3 * launch
               for e in (E["tiled_matmul"][1], E["tiled_matmul"][0]))
    kv_l2 = (N // pt.BLOCK_ROWS) * 2 * N * D * H * B
    v3 = tuple(max(t(2 * f_mm, e), qkvo / BW, kv_l2 / L2) + launch for e in (E["fused_fp32"][1], E["fused_fp32"][0]))
    b1 = (t(2 * f_mm, E["library"][1]) + launch, t(2 * f_mm, E["library"][0]) + 2 * s_bytes / BW + 4 * launch)
    return {"V1": v1, "V2": v2, "V3": v3, "B1": b1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mhz", type=float, default=1395.0)
    a = ap.parse_args()
    pin = a.mhz
    pinned_dir = RES / f"clock{int(pin)}"
    pk = json.loads((ROOT / "results" / "cuda" / "peak.json").read_text())
    P = pk["fma_sustained"]["tflops_settled"] * 1e12
    BW = pk["dram_read"]["gb_per_s"] * 1e9
    L2 = pk["l2_read"]["gb_per_s"] * 1e9
    launch = pk["launch_b2b"]["ms_median"] * 1e-3
    committed = json.loads((ROOT / "results" / "cuda" / "predicted_times.json").read_text())["dense_ms"]
    check = dense(P, BW, L2, launch)
    for k, (lo, hi) in check.items():
        assert [round(lo * 1e3, 3), round(hi * 1e3, 3)] == committed[k], (k, lo, hi, committed[k])
    scale = pin / PEAK_MHZ
    pred = {k: (lo * 1e3, hi * 1e3) for k, (lo, hi) in dense(P * scale, BW, L2 * scale, launch).items()}

    rows = {}
    for name, (fname, opname) in OPS.items():
        pf = pinned_dir / fname
        if not pf.exists():
            continue
        doc = json.loads(pf.read_text())
        op = {o["name"]: o for o in doc["ops"]}[opname]
        if op["status"] != "ok":
            rows[name] = {"status": op["status"]}
            continue
        mhz = np.array([s["sm_mhz"] for s in op["samples"]], dtype=float)
        held = float(np.mean(np.abs(mhz - pin) <= HOLD_TOL * pin))
        events = sorted({e for s in op["samples"] for e in s["clock_events"]})
        t = op["clocks"]["event_batch_ms"]["median"]
        un = json.loads((RES / fname).read_text())
        uop = {o["name"]: o for o in un["ops"]}[opname]
        umhz = [g["median_sm_mhz"] for g in un["groups"] if g["group"] == uop["group"]][0]
        ut = uop["clocks"]["event_batch_ms"]["median"]
        key = "B1" if name.startswith("B1") else name
        lo, hi = pred[key]
        rows[name] = {
            "time_ms": t, "p10_ms": op["clocks"]["event_batch_ms"]["p10"], "p90_ms": op["clocks"]["event_batch_ms"]["p90"],
            "sm_mhz": {"min": float(mhz.min()), "median": float(np.median(mhz)), "max": float(mhz.max())},
            "share_within_1pct_of_pin": held, "pin_held": held >= HOLD_SHARE, "clock_events": events,
            "flagged": sum(s["flagged"] for s in op["samples"]),
            "phases_ms": {k[6:-3]: v["median"] for k, v in op["clocks"].items() if k.startswith("phase_")},
            "predicted_ms_at_pin": [lo, hi],
            "versus_prediction": "inside" if lo <= t <= hi else ("above" if t > hi else "below"),
            "unpinned": {"time_ms": ut, "median_sm_mhz": umhz},
            "megacycles_pinned_over_unpinned": (t * pin) / (ut * umhz),
        }
    out = {"experiment": "attention at a pinned SM clock (definitions: bench/clock_pinned.py)",
           "pinned_mhz": pin, "predictions_ms_at_pin": {k: list(v) for k, v in pred.items()}, "ops": rows}
    if "V2" in rows and "V3" in rows and rows["V2"].get("pin_held") and rows["V3"].get("pin_held"):
        x2 = json.loads((RES / "x2_breakdown.json").read_text())
        nxn = x2["h1"]["clause2"]["v2_nxn_dram_floor_ms"]
        saving = rows["V2"]["time_ms"] - rows["V3"]["time_ms"]
        out["v3_vs_v2_equal_clock"] = {"saving_ms": saving, "v2_nxn_dram_floor_ms": nxn,
                                       "share_of_nxn_floor": saving / nxn, "h1_clause2_holds": saving / nxn >= 0.5}
    pinned_dir.mkdir(parents=True, exist_ok=True)
    (pinned_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"pinned at {pin:.0f} MHz; predictions rescaled to it (ms):",
          {k: [round(x, 2) for x in v] for k, v in pred.items()})
    for name, r in rows.items():
        if "time_ms" not in r:
            print(f"{name:20s} {r['status']}")
            continue
        print(f"{name:20s} {r['time_ms']:.3f} ms  clock {r['sm_mhz']['min']:.0f}-{r['sm_mhz']['max']:.0f} "
              f"(held {100 * r['share_within_1pct_of_pin']:.0f}%) events {r['clock_events']}  predicted "
              f"{r['predicted_ms_at_pin'][0]:.2f}-{r['predicted_ms_at_pin'][1]:.2f}: {r['versus_prediction']}  "
              f"unpinned {r['unpinned']['time_ms']:.3f} at {r['unpinned']['median_sm_mhz']}  cycles ratio "
              f"{r['megacycles_pinned_over_unpinned']:.2f}")
    if "v3_vs_v2_equal_clock" in out:
        c = out["v3_vs_v2_equal_clock"]
        print(f"V3 vs V2 at equal clock: saving {c['saving_ms']:.3f} ms = {100 * c['share_of_nxn_floor']:.0f}% of the "
              f"{c['v2_nxn_dram_floor_ms']:.2f} ms N x N floor; H1 clause 2 {'holds' if c['h1_clause2_holds'] else 'fails'}")
    print("wrote", pinned_dir / "summary.json")


if __name__ == "__main__":
    main()

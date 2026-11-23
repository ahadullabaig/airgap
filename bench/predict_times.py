"""Predicted kernel times for PLAN.md section 3.1, written before any attention kernel exists.

Inputs: the measured limits of this GPU (results/cuda/peak.json, kernels/microbench). Shape: the
SmolVLM2 vision encoder's attention, N = 1024 tokens, 12 heads of d = 64, FP32, one frame.

Each prediction is a range, from explicit assumptions about how efficiently a kernel of that
kind uses the hardware (EFF below). The point is to write the expectation down, so the
measurements can confirm or refute it and the gap can be explained.

Example:
    .venv/bin/python -I bench/predict_times.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N, D, H, B = 1024, 64, 12, 4  # tokens, head dim, heads, bytes per FP32 value
D_MODEL = D * H

# Assumed share of measured FP32 FMA peak, (low, high). Guesses to test, not measurements.
EFF = {
    "naive_matmul": (0.10, 0.30),   # V1: one thread per output, operands through L1/L2, no shared-memory tiling
    "tiled_matmul": (0.40, 0.70),   # V2: shared-memory tiles, register blocking
    "fused_fp32": (0.35, 0.65),     # V3: FlashAttention-style in FP32, no tensor cores; row max/sum shuffles
    "library": (0.35, 0.85),        # B1: cuBLAS / PyTorch SDPA; low end = the whole PyTorch encoder's measured rate
                                    # (212 GFLOP per frame in 78 ms = 2.7 TFLOPS, results/tokens/checks.json)
}
L1_REUSE = 16            # V1 pessimistic end: 16 x 16 thread blocks reuse each operand row from L1 16 times
BLOCK_ROWS = 64          # query rows per thread block in V3/G1
BLOCKS_PER_SM = (2, 4)   # resident V3/G1 blocks per SM (register and shared-memory limited), (low, high)
G1_FIXED_US = (5, 15)    # G1 extra in-launch work: prefix sum over N, compaction, scatter
B2_FIXED_US = (50, 150)  # B2: boolean mask, nonzero (GPU-to-CPU sync to learn M), index_select, index_copy, dispatch


def main() -> None:
    pk = json.loads((ROOT / "results" / "cuda" / "peak.json").read_text())
    P = pk["fma_sustained"]["tflops_settled"] * 1e12  # about 10 s back to back, power-capped but steady
    BW = pk["dram_read"]["gb_per_s"] * 1e9
    L2 = pk["l2_read"]["gb_per_s"] * 1e9
    launch = pk["launch_b2b"]["ms_median"] * 1e-3
    roundtrip = pk["launch_rt"]["ms_median"] * 1e-3
    sms = pk["sms"]

    f_mm = 2 * N * N * D * H              # one of QK^T or PV, all heads (FLOPs)
    s_bytes = N * N * H * B               # the N x N score/probability matrix, all heads
    qkvo_bytes = 4 * N * D * H * B        # Q, K, V in; O out
    operand_l2_bytes = N * N * H * 2 * D * B / L1_REUSE  # naive: each output element streams two length-64 rows

    def t(flops, eff):
        return flops / (eff * P)

    pred = {}
    # V1: scores, softmax, PV as three kernels; N x N stored in DRAM
    lo = (max(t(f_mm, EFF["naive_matmul"][1]), s_bytes / BW)            # scores
          + 2 * s_bytes / BW                                             # softmax: read + write
          + max(t(f_mm, EFF["naive_matmul"][1]), s_bytes / BW) + 3 * launch)
    hi = (max(t(f_mm, EFF["naive_matmul"][0]), operand_l2_bytes / L2)
          + 2 * s_bytes / BW
          + max(t(f_mm, EFF["naive_matmul"][0]), operand_l2_bytes / L2) + 3 * launch)
    pred["V1"] = (lo, hi)
    # V2: tiled matmuls, softmax separate, N x N still stored
    lo = 2 * max(t(f_mm, EFF["tiled_matmul"][1]), s_bytes / BW) + 2 * s_bytes / BW + 3 * launch
    hi = 2 * max(t(f_mm, EFF["tiled_matmul"][0]), s_bytes / BW) + 2 * s_bytes / BW + 3 * launch
    pred["V2"] = (lo, hi)
    # V3: fused, online softmax; DRAM traffic is only Q, K, V, O; K and V re-read per query tile from L2
    kv_l2 = (N // BLOCK_ROWS) * 2 * N * D * H * B
    lo = max(t(2 * f_mm, EFF["fused_fp32"][1]), qkvo_bytes / BW, kv_l2 / L2) + launch
    hi = max(t(2 * f_mm, EFF["fused_fp32"][0]), qkvo_bytes / BW, kv_l2 / L2) + launch
    pred["V3"] = (lo, hi)
    # B1: library attention on all N (cuBLAS-class efficiency; the math backend also round-trips N x N)
    lo = t(2 * f_mm, EFF["library"][1]) + launch
    hi = t(2 * f_mm, EFF["library"][0]) + 2 * s_bytes / BW + 4 * launch
    pred["B1"] = (lo, hi)

    # Gated kernels at keep-rate k = M / N. Blocks = H * ceil(M / 64) query tiles, each sweeping all N keys.
    # Optimistic end: the work divides perfectly over the GPU (no wave quantisation, a lone block on an SM
    # runs as fast as the SM's share). Pessimistic end: whole waves of SMs * per-SM blocks, and a partly filled
    # wave costs as much as a full one. The truth lies between; split-KV would move it toward the optimistic end.
    def tiles(m):
        return H * math.ceil(max(m, 1) / BLOCK_ROWS)

    def rows_time(m, v3, per_sm, quantised):
        dense_tiles = tiles(N)
        if quantised:
            per_wave = (v3 - launch) / math.ceil(dense_tiles / (sms * per_sm))
            return math.ceil(tiles(m) / (sms * per_sm)) * per_wave
        return (v3 - launch) * tiles(m) / dense_tiles

    def gated_time(k, kind, side):
        m = max(1, round(k * N))
        v3, per_sm, fixed, quant = ((pred["V3"][0], BLOCKS_PER_SM[1], G1_FIXED_US[0], False) if side == 0 else
                                    (pred["V3"][1], BLOCKS_PER_SM[0], G1_FIXED_US[1], True))
        if kind == "B2":
            return (B2_FIXED_US[side] * 1e-6) + k * pred["B1"][side]
        work = rows_time(m, v3, per_sm, quant)
        if kind == "G1_exact_update":
            # unchanged rows: remove and re-add the changed keys' terms, about 4d multiply-adds per
            # (unchanged row, changed key) against 2d per (row, key) for a recomputed row: 2k(1 - k) of dense
            work += (v3 - launch) * 2 * (m / N) * (1 - m / N)
        return launch + fixed * 1e-6 + work

    kinds = ("G1_rows_only", "G1_exact_update", "B2")
    keeps = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    gated = [{"keep": k, "M": max(1, round(k * N)), **{n: (gated_time(k, n, 0), gated_time(k, n, 1)) for n in kinds}}
             for k in keeps]

    def breakeven(kind, ref):
        """Largest keep-rate (1% steps) at which the gated version beats ref: (optimistic, pessimistic)."""
        out = []
        for side in (0, 1):
            ok = [k / 100 for k in range(1, 101) if gated_time(k / 100, kind, side) < pred[ref][side]]
            out.append(max(ok) if ok else None)
        return out

    encoder_layer_flops = 2 * (12 * N * D_MODEL * D_MODEL) + 2 * f_mm  # token-wise ops + attention core
    result = {
        "shape": {"N": N, "head_dim": D, "heads": H, "dtype": "fp32"},
        "measured_limits": {"fp32_tflops_sustained": P / 1e12, "fp32_tflops_burst": pk["fma"]["tflops"], "dram_gb_per_s": BW / 1e9, "l2_gb_per_s": L2 / 1e9,
                            "launch_back_to_back_us": launch * 1e6, "launch_roundtrip_us": roundtrip * 1e6,
                            "source": "results/cuda/peak.json"},
        "assumptions": {"efficiency_share_of_peak": EFF, "v1_l1_reuse": L1_REUSE, "block_rows": BLOCK_ROWS,
                        "blocks_per_sm": BLOCKS_PER_SM,
                        "g1_fixed_us": G1_FIXED_US, "b2_fixed_us": B2_FIXED_US},
        "work": {"flops_qk_plus_pv": 2 * f_mm, "nxn_matrix_mb": s_bytes / 1e6, "qkvo_mb": qkvo_bytes / 1e6,
                 "attention_share_of_encoder_layer_flops": 2 * f_mm / encoder_layer_flops},
        "dense_ms": {k: [round(v[0] * 1e3, 3), round(v[1] * 1e3, 3)] for k, v in pred.items()},
        "gated_ms": [{"keep": r["keep"], "M": r["M"],
                      **{n: [round(r[n][0] * 1e3, 3), round(r[n][1] * 1e3, 3)]
                         for n in ("G1_rows_only", "G1_exact_update", "B2")}} for r in gated],
        "breakeven_keep_vs_V3": {"G1_rows_only": breakeven("G1_rows_only", "V3"),
                                 "G1_exact_update": breakeven("G1_exact_update", "V3")},
        "breakeven_keep_vs_each_other": "rows-only G1 and B2 are like for like (both leave unchanged rows stale); "
                                        "their ranges overlap at every keep-rate, so B1's measured time decides",
        "breakeven_keep_vs_B1": {"B2": breakeven("B2", "B1")},
        "notes": [
            "Launch costs are a few microseconds against dense times near a millisecond, so H1 expects dense "
            "attention at N = 1024 to be limited by compute and memory traffic, not fixed costs.",
            "V2 cannot avoid four N x N passes through DRAM (about 1 ms here); V3 removes them, so fusion is "
            "predicted to matter about as much as tiling.",
            "G1 work is quantised into waves of thread blocks: at small M one partly filled wave sets the time "
            "(split-KV would lower it).",
            "Attention is only about a fifth of an encoder layer's FLOPs at this shape, so gating attention "
            "alone bounds the encoder-level saving; the token-wise layers need gating too.",
            "Break-even keep-rates are against the dense kernel's own time, not against running dense at a "
            "lower frame rate.",
        ],
    }
    out = ROOT / "results" / "cuda" / "predicted_times.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("dense_ms", "breakeven_keep_vs_V3", "breakeven_keep_vs_B1", "work")},
                     indent=1))
    print("gated (ms):")
    for r in result["gated_ms"]:
        print(f"  k={r['keep']:<5} M={r['M']:<5} G1 rows {r['G1_rows_only']}  G1 exact {r['G1_exact_update']}  B2 {r['B2']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

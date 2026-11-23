"""Inputs and a float64 reference for the attention kernels' tests (V1 now; B1 and later versions read
the same files, so every implementation is checked against the same data).

Each case is a directory under kernels/attention/build/testdata/ (git-ignored, regenerated on demand):
    q.f32, k.f32, v.f32   [H][N][D] float32, little endian
    rows.i32              the query rows that have a reference (the same rows in every head)
    oref.f64              [H][len(rows)][D] float64 reference output for those rows
    meta.json             H, N, D, seed, description

Example:
    .venv/bin/python -I bench/attn_testdata.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kernels" / "attention" / "build" / "testdata"
D = 64

CASES = [
    # name, heads, n, q scale, rows ("all" or a list), description
    ("rand_h12_n1024", 12, 1024, 1.0, "all", "the encoder's shape, every row"),
    ("rand_h2_n1000", 2, 1000, 1.0, "all", "N not a multiple of 16 (partial tiles), every row"),
    ("peaky_h2_n256", 2, 256, 40.0, "all", "Q scaled x40: scores up to ~100, rows near one-hot "
                                           "(a softmax without max subtraction overflows)"),
    ("rand_h3_n4100", 3, 4100, 1.0, "sampled", "partial tiles at larger N, sampled rows in every head"),
    ("rand_h1_n16384", 1, 16384, 1.0, "sampled", "the largest single-head size V1 fits easily, sampled rows"),
]


def sampled_rows(n: int, rng) -> np.ndarray:
    fixed = [0, 1, 15, 16, 17, n // 2 - 1, n // 2, n - 17, n - 16, n - 2, n - 1]
    extra = rng.choice(n, size=21, replace=False)
    return np.unique(np.array([r for r in fixed if 0 <= r < n] + list(extra), dtype=np.int32))


def reference(q, k, v, rows):
    """float64 softmax(q k^T / sqrt(D)) v for the given rows of every head."""
    q, k, v = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
    out = np.empty((q.shape[0], len(rows), D))
    for h in range(q.shape[0]):
        s = q[h, rows] @ k[h].T / np.sqrt(D)
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        out[h] = p @ v[h]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (name, h, n, qs, rows_kind, desc) in enumerate(CASES):
        d = OUT / name
        if (d / "meta.json").exists() and "--force" not in sys.argv:
            print(f"{name}: exists")
            continue
        d.mkdir(exist_ok=True)
        rng = np.random.default_rng(1000 + i)
        q = (rng.standard_normal((h, n, D)) * qs).astype(np.float32)
        k = rng.standard_normal((h, n, D)).astype(np.float32)
        v = rng.standard_normal((h, n, D)).astype(np.float32)
        rows = np.arange(n, dtype=np.int32) if rows_kind == "all" else sampled_rows(n, rng)
        oref = reference(q, k, v, rows)
        for arr, fn in ((q, "q.f32"), (k, "k.f32"), (v, "v.f32"), (rows, "rows.i32"), (oref, "oref.f64")):
            arr.astype(arr.dtype.newbyteorder("<")).tofile(d / fn)
        (d / "meta.json").write_text(json.dumps({"heads": h, "n": n, "d": D, "seed": 1000 + i, "q_scale": qs,
                                                 "rows": int(len(rows)), "description": desc}, indent=1) + "\n")
        print(f"{name}: H={h} N={n} rows={len(rows)}")


if __name__ == "__main__":
    main()

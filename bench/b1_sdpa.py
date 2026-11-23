"""B1: PyTorch scaled_dot_product_attention on all N tokens, FP32, each backend pinned (PLAN.md section 3.5).

Same shapes as V1 (kernels/attention/v1_bench.cu): group "h12_n1024" (12 heads, N = 1024, D = 64) and
groups "x1_n<N>" (one head, N = 128 ... 32,768). Backends: MATH and EFFICIENT_ATTENTION run in FP32 on this
build; FLASH_ATTENTION and CUDNN_ATTENTION have no FP32 kernel and are attempted once (h12 group) so the
result file records that. The backend is selected per call with torch.nn.attention.sdpa_kernel, so the
Python overhead of that context manager is part of B1's time. TF32 is off for matmuls and cuDNN.

Checks before timing (same criterion as V1, max over rows of ||o - o_ref|| / ||o_ref|| below 1e-4):
- h12_n1024 and x1_n16384 use the shared inputs and float64 reference (bench/attn_testdata.py), the
  same files V1's test reads;
- other sizes: random inputs, sampled rows (row 0, tile edges, middle, last) against float64 on the GPU.

The CUDA Graphs variant in the plan is deferred.

Example:
    .venv/bin/python -I bench/b1_sdpa.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import harness as H  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

D = 64
TESTDATA = ROOT / "kernels" / "attention" / "build" / "testdata"
SHARED = {(12, 1024): "rand_h12_n1024", (1, 16384): "rand_h1_n16384"}


def check_rows(n: int) -> list[int]:
    return sorted({r for r in (0, 1, 15, 16, n // 2, n - 17, n - 16, n - 1) if 0 <= r < n})


def load_case(name: str):
    d = TESTDATA / name
    meta = json.loads((d / "meta.json").read_text())
    h, n = meta["heads"], meta["n"]
    arr = lambda f, dt, shape: np.fromfile(d / f, dtype=dt).reshape(shape)  # noqa: E731
    q, k, v = (torch.from_numpy(arr(f, "<f4", (1, h, n, D)).copy()) for f in ("q.f32", "k.f32", "v.f32"))
    rows = np.fromfile(d / "rows.i32", dtype="<i4")
    oref = torch.from_numpy(arr("oref.f64", "<f8", (h, len(rows), D)).copy())
    return q, k, v, torch.from_numpy(rows.astype(np.int64)), oref


def rel_errors(o: torch.Tensor, ref: torch.Tensor) -> float:
    """max over (head, row) of ||o - ref|| / ||ref||, inputs [h, rows, D]."""
    o, ref = o.double(), ref.double()
    err = (o - ref).norm(dim=-1) / ref.norm(dim=-1)
    return float(err.max()) if torch.isfinite(err).all() else math.inf


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "attention" / "b1.json")
    if not (TESTDATA / "rand_h12_n1024" / "meta.json").exists():
        sys.exit("shared test data missing: run .venv/bin/python -I bench/attn_testdata.py")
    run = H.Runner("attention_b1_sdpa", H.read_limits(ROOT / "results" / "cuda" / "peak.json"))
    shapes = [("h12_n1024", 12, 1024)] + [(f"x1_n{n}", 1, n) for n in
                                          (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768)]
    for group, h, n in shapes:
        tensors = {}

        def setup(h=h, n=n, tensors=tensors):
            if tensors:
                return
            if (h, n) in SHARED:
                q, k, v, rows, oref = load_case(SHARED[(h, n)])
                tensors.update(q=q.cuda(), k=k.cuda(), v=v.cuda(), rows=rows.cuda(), oref=oref.cuda(),
                               source=f"shared test data {SHARED[(h, n)]}")
            else:
                g = torch.Generator(device="cuda").manual_seed(17)
                q, k, v = (torch.randn(1, h, n, D, device="cuda", generator=g) for _ in range(3))
                rows = torch.tensor(check_rows(n), device="cuda")
                q64, k64, v64 = q[0].double(), k[0].double(), v[0].double()
                s = q64[:, rows] @ k64.transpose(-1, -2) / math.sqrt(D)
                oref = torch.softmax(s, dim=-1) @ v64
                tensors.update(q=q, k=k, v=v, rows=rows, oref=oref, source="random inputs, float64 reference rows")

        def teardown(tensors=tensors):
            tensors.clear()

        backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
        if group == "h12_n1024":
            backends += [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]
        for i, be in enumerate(backends):
            def run_once(be=be, tensors=tensors):
                with sdpa_kernel(be):
                    return F.scaled_dot_product_attention(tensors["q"], tensors["k"], tensors["v"])

            def check(run_once=run_once, tensors=tensors):
                o = run_once()
                torch.cuda.synchronize()
                err = rel_errors(o[0][:, tensors["rows"]], tensors["oref"])
                return {"ok": err <= 1e-4, "max_rel_err": err, "max_abs_err": None,
                        "note": f"{tensors['source']}; max over rows of ||o - o_ref|| / ||o_ref||"}

            run.add(H.Op(name=f"b1_{be.name.lower()}_h{h}_n{n}", run=run_once, group=group,
                         params={"version": "B1", "backend": be.name, "heads": h, "n": n, "d": D, "dtype": "fp32",
                                 "tf32": "off", "dispatch": "sdpa_kernel context per call (Python overhead included)"},
                         flops=4.0 * h * n * n * D, setup=setup if i == 0 else None,
                         teardown=teardown if i == len(backends) - 1 else None, check=check))
    run.run_all()
    run.write_json(out)
    print(f"{'op':34s} {'status':14s} {'event ms':>10s} {'host ms':>10s} {'TFLOPS':>7s} {'peak MiB':>9s}")
    for s in run.ops:
        ev = s.clocks.get("event_batch_ms", {}).get("median")
        hs = s.clocks.get("host_sync_ms", {}).get("median")
        f = lambda x, w=10, p=4: f"{x:{w}.{p}f}" if x is not None and not (isinstance(x, float) and math.isnan(x)) else " " * (w - 3) + "nan"  # noqa: E731
        print(f"{s.op.name:34s} {s.status:14s} {f(ev)} {f(hs)} {f(s.tflops, 7, 3)} {f(s.mem_peak_mib, 9, 1)}"
              f"{'  ' + s.error[:70] if s.error else ''}")
    print("wrote", out)


if __name__ == "__main__":
    main()

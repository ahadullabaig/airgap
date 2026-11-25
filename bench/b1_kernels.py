"""Which CUDA kernels PyTorch's FP32 attention backends launch at 12 heads, N = 1024, D = 64 (B1's settings:
TF32 off for matmuls and cuDNN, highest matmul precision), recorded with the PyTorch profiler.

Why: on compute capability 8.0 and newer, PyTorch's FP32 "memory-efficient" backend is a CUTLASS kernel
built for Sm80, which in PyTorch's source multiplies FP32 operands on tensor cores by splitting them into TF32
parts (OpMultiplyAddFastF32). The math backend uses cuBLAS SGEMM on CUDA cores. Our kernels use CUDA cores
only, so the paper must say which kind of kernel each comparison is against. The tensor-pipe activity of the
memory-efficient kernel is measured with Nsight Compute (command below; output in
results/attention/b1_efficient_ncu.txt).

Example:
    .venv/bin/python -I bench/b1_kernels.py
    ncu --kernel-name regex:fmha --launch-skip 2 --launch-count 1 --clock-control none \\
        --metrics sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \\
        .venv/bin/python -I bench/b1_kernels.py --ncu-target
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def main() -> None:
    q, k, v = (torch.randn(1, 12, 1024, 64, device="cuda") for _ in range(3))
    if "--ncu-target" in sys.argv:  # three calls of the memory-efficient backend for Nsight Compute
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for _ in range(3):
                F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        return
    out = {"settings": "fp32, 12 heads, N = 1024, D = 64; allow_tf32 False (matmul, cuDNN); matmul precision highest",
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "backends": {}}
    for be in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
        out["backends"][be.name] = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    path = ROOT / "results" / "attention" / "b1_kernels.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    for name, ks in out["backends"].items():
        print(name, [k[:80] for k in ks])
    print("wrote", path)


if __name__ == "__main__":
    main()

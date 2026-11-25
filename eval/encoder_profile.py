"""Where the vision encoder's GPU time goes in E1's configuration (bf16, four frames, no image splitting): the
CUDA kernels of one get_image_features call, recorded with the PyTorch profiler and grouped by kind.

Groups, by kernel name: attention (flash / fused multi-head attention / SDPA kernels), matmul (GEMM kernels),
convolution (the patch embedding), and other (layer norms, activations, elementwise, copies). The attention
share is the measured counterpart of the FLOP-share estimate in results/cuda/predicted_times.json. Clocks are not
pinned: this is the deployed configuration.

Example:
    .venv/bin/python -I eval/encoder_profile.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import capera  # noqa: E402
import e1  # noqa: E402

REPEATS = 5
OUT = ROOT / "results" / "e1" / "encoder_profile.json"
GROUPS = [("attention", re.compile(r"flash|fmha|attention|sdpa|scaled_dot", re.I)),
          ("convolution", re.compile(r"conv|implicit_gemm|winograd", re.I)),
          ("matmul", re.compile(r"gemm|gemv|xmma|cutlass|ampere_.*(bf16|h16|s16)|sm\d+_", re.I))]


def kind(name: str) -> str:
    for g, rx in GROUPS:
        if rx.search(name):
            return g
    return "other"


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vlm import Captioner
    prompt = json.loads(e1.CONFIG.read_text())["prompt"]
    cap = Captioner()
    model = cap.model.model
    c = capera.stratified_order(capera.load("test"), e1.SEED)[0]
    inp = cap.inputs(cap.frames(c.path, c.duration_s, e1.FRAMES), prompt, e1.SPLIT_IMAGES)
    kw = {k: inp[k] for k in ("pixel_attention_mask",) if k in inp}
    with torch.inference_mode():
        for _ in range(3):  # warm-up
            model.get_image_features(inp["pixel_values"], return_dict=True, **kw)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(REPEATS):
                model.get_image_features(inp["pixel_values"], return_dict=True, **kw)
            torch.cuda.synchronize()
    per_kernel: dict[str, float] = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            per_kernel[e.name] = per_kernel.get(e.name, 0.0) + e.device_time / 1e3 / REPEATS  # ms per call
    totals: dict[str, float] = {}
    for n, t in per_kernel.items():
        totals[kind(n)] = totals.get(kind(n), 0.0) + t
    gpu = sum(totals.values())
    top = sorted(per_kernel.items(), key=lambda x: -x[1])[:12]
    out = {"what": "GPU kernel time of one vision-encoder call (4 frames, bf16), grouped by kind "
                   "(definitions: eval/encoder_profile.py)",
           "attn_implementation": getattr(cap.model.config.vision_config, "_attn_implementation", None),
           "repeats": REPEATS, "gpu": torch.cuda.get_device_name(),
           "gpu_ms_per_call": gpu, "ms_by_kind": totals, "share_by_kind": {k: v / gpu for k, v in totals.items()},
           "top_kernels_ms": [[n[:120], t] for n, t in top]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"GPU time per encoder call {gpu:.2f} ms;", {k: f"{v:.2f} ms ({100 * v / gpu:.0f}%)" for k, v in totals.items()})
    for n, t in top[:8]:
        print(f"  {t:7.3f} ms  [{kind(n)}] {n[:100]}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

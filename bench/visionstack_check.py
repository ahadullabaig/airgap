"""Checks for bench/visionstack.py before any token-trace run. Writes results/tokens/checks.json.

1. Path: the standalone stack reproduces the full model's image features.
2. Pixel round trip: error from feeding the uint8 view instead of the processor's floats.
3. Index mapping: changing one 16x16 block changes only that token at the embedding.
4. Row-only compute: full layer then select rows equals computing only those rows.
5. Batch size: one frame alone vs inside a batch of 8.
6. BF16 floor: bf16 vs fp32 on the same pixels.
7. Timing per frame on this GPU.

Example:
    .venv/bin/python -I bench/visionstack_check.py
"""

import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from visionstack import GRID, TOKENS, VisionStack  # noqa: E402
from x0core import model_view  # noqa: E402

MODEL = ROOT / "models" / "SmolVLM2-500M-Video-Instruct"
CLIP = ROOT / "data" / "era" / "Videos" / "Tra" / "Soccer" / "Soccer_030 .mp4"


def rel(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-token relative error of a against reference b (last dim is the feature)."""
    e = (a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1)
    return {"mean": float(e.mean()), "p95": float(e.flatten().quantile(0.95)), "max": float(e.max())}


def main() -> None:
    from transformers import AutoImageProcessor
    ip = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    mean, std = np.array(ip.image_mean, np.float32), np.array(ip.image_std, np.float32)
    with av.open(str(CLIP)) as c:
        imgs = [f.to_image() for i, f in zip(range(9), c.decode(c.streams.video[0]))]
    views = np.stack([model_view(ip, im, mean, std) for im in imgs])
    out = {"clip": str(CLIP.relative_to(ROOT)), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    vs = VisionStack(str(MODEL), keep_full=True)
    full = vs.full.to("cuda")

    # 1. path: processor float pixels through the full model vs through the stack
    pv = torch.from_numpy(np.asarray(ip(images=[[imgs[0]]], do_image_splitting=False,
                                        return_tensors="np")["pixel_values"])).to("cuda")
    with torch.no_grad():
        ref = full.model.get_image_features(pixel_values=pv).pooler_output
        ref = ref[0] if isinstance(ref, (list, tuple)) else ref
    enc_f, con_f, _ = vs.forward(pv[:, 0])
    out["path_max_abs_diff"] = float((con_f.reshape(ref.shape) - ref).abs().max())
    full.model.text_model.to("cpu")  # the vision tower and connector are shared with vs and stay on the GPU
    full.lm_head.to("cpu")
    vs.full = None
    del full
    torch.cuda.empty_cache()

    # 2. round trip: uint8 view vs processor floats
    enc_u, con_u, _ = vs.forward(vs.pixels(views[0]))
    out["roundtrip_rel_error"] = {"encoder": rel(enc_u, enc_f), "connector": rel(con_u, con_f)}

    # 3. index mapping
    by, bx = 5, 7
    v2 = views[0].copy()
    v2[by * 16:(by + 1) * 16, bx * 16:(bx + 1) * 16] = 255 - v2[by * 16:(by + 1) * 16, bx * 16:(bx + 1) * 16]
    d = (vs.embed(vs.pixels(v2)) - vs.embed(vs.pixels(views[0]))).abs().amax(dim=-1)[0]
    changed = torch.nonzero(d > 0).flatten().tolist()
    out["index_mapping"] = {"block": [by, bx], "expected_token": by * GRID + bx, "changed_tokens": changed}
    assert changed == [by * GRID + bx], out["index_mapping"]

    # 4. row-only compute on a mixed input (embedding of frame 8 for half the tokens, frame 0 for the rest)
    g = torch.Generator().manual_seed(0)
    rows = torch.randperm(TOKENS, generator=g)[:300].sort().values.to("cuda")
    h = vs.embed(vs.pixels(views[0]))
    h8 = vs.embed(vs.pixels(views[8]))
    mix = torch.where((torch.rand(TOKENS, generator=g) < 0.5).to("cuda")[None, :, None], h8, h)
    for layer in (0, 5):
        full_sel = vs.layer(layer, mix)[:, rows]
        only = vs.layer_rows(layer, mix, rows)
        out.setdefault("row_only_rel_error", {})[str(layer)] = rel(only, full_sel)
    assert all(v["max"] < 1e-4 for v in out["row_only_rel_error"].values()), out["row_only_rel_error"]

    # 5. batch size
    px = vs.pixels(views[:8])
    e8, c8, _ = vs.forward(px)
    e1, c1, _ = vs.forward(px[3:4])
    out["batch8_vs_batch1_rel_error"] = {"encoder": rel(e8[3:4], e1), "connector": rel(c8[3:4], c1)}

    # 7. timing (before loading the bf16 copy)
    for b in (1, 4, 8):
        x = vs.pixels(views[:b])
        vs.forward(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            vs.forward(x)
        torch.cuda.synchronize()
        out.setdefault("ms_per_frame_fp32", {})[str(b)] = round((time.perf_counter() - t) / 5 / b * 1000, 2)
    out["peak_mem_mib_fp32"] = round(torch.cuda.max_memory_allocated() / 2**20)

    # 6. bf16 floor
    vb = VisionStack(str(MODEL), dtype=torch.bfloat16)
    eb, cb, _ = vb.forward(vb.pixels(views[0]))
    out["bf16_vs_fp32_rel_error"] = {"encoder": rel(eb, enc_u), "connector": rel(cb, con_u)}

    path = ROOT / "results" / "tokens" / "checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

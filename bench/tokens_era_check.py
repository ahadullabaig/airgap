"""Checks for bench/tokens_era.py on a short stretch of one clip. Writes results/tokens/gate_checks.json.

1. Endpoints of the gate simulation: refreshing every block (threshold below 0)
   reproduces the dense outputs in both tiers; refreshing none (threshold above
   255) reproduces the first frame's output held, in both tiers.
2. Sensitivity yardstick: how much the encoder and connector outputs move when
   the input changes by tiny amounts that carry no content: the same frame
   again (must be 0), +-1 grey level of random noise, and a repeated frame from
   the clip (consecutive frames with MAD < 0.5, if any).

Example:
    .venv/bin/python -I bench/tokens_era_check.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import tokens_era as te  # noqa: E402
from k0lib import K0  # noqa: E402
from visionstack import VisionStack  # noqa: E402

CLIPS = [("Basketball", "Basketball_016 .mp4"), ("Constructing", "Constructing_028 .mp4")]
FRAMES = 16


def rel_stats(a, b) -> dict:
    e = te.tok_rel(a, b)
    return {"mean": float(e.mean()), "max": float(e.max())}


def main() -> None:
    from transformers import AutoImageProcessor
    ip = AutoImageProcessor.from_pretrained(te.MODEL, local_files_only=True)
    mean, std = np.array(ip.image_mean, np.float32), np.array(ip.image_std, np.float32)
    vs = VisionStack(str(te.MODEL))
    k0 = K0()
    out = {"clips": [c[1] for c in CLIPS], "frames": FRAMES}

    cls, name = CLIPS[0]
    views = te.load_views(ROOT / "data" / "era" / "Videos" / "Tra" / cls / name, ip, mean, std)[:FRAMES]
    enc_all, con_all, *_ = te.dense_pass(vs, k0, views)

    # 1. endpoints
    for label, thr in (("refresh_all", -1.0), ("refresh_none", 1e9)):
        for s in (1, 4):
            cell = te.gate_cell(vs, k0, views, enc_all, con_all, s, thr)
            hold_first = te.hold_stream(con_all, 10**6)  # first frame held for the whole stretch
            out.setdefault("endpoints", {})[f"{label}/stride{s}"] = {
                "keep": float(np.mean(cell["keep_counts"])) / te.TOKENS,
                "A_con_mean": cell["samples"]["A"]["con_mean"], "B_con_mean": cell["samples"]["B"]["con_mean"],
                "A_stream": cell["stream"]["A"]["mean"], "B_stream": cell["stream"]["B"]["mean"],
                "hold_stride_stream": te.hold_stream(con_all, s)["mean"], "hold_first_stream": hold_first["mean"],
            }
    ep = out["endpoints"]
    for s in (1, 4):
        a = ep[f"refresh_all/stride{s}"]
        assert a["keep"] == 1.0 and a["A_con_mean"] < 1e-4 and a["B_con_mean"] < 1e-4, a
        assert abs(a["A_stream"] - a["hold_stride_stream"]) < 1e-4, a
        n = ep[f"refresh_none/stride{s}"]
        assert n["keep"] == 0.0 and abs(n["B_stream"] - n["hold_first_stream"]) < 1e-4, n
        assert abs(n["A_stream"] - n["hold_first_stream"]) < 1e-4, n

    # 2. sensitivity
    rng = np.random.default_rng(0)
    sens = {}
    for cls, name in CLIPS:
        v = te.load_views(ROOT / "data" / "era" / "Videos" / "Tra" / cls / name, ip, mean, std)
        e0, c0, _ = vs.forward(vs.pixels(v[0]))
        e1, c1, _ = vs.forward(vs.pixels(v[0]))
        noisy = np.clip(v[0].astype(np.int16) + rng.integers(-1, 2, v[0].shape), 0, 255).astype(np.uint8)
        en, cn, _ = vs.forward(vs.pixels(noisy))
        e_next, c_next, _ = vs.forward(vs.pixels(v[1]))
        r = {"same_frame": {"encoder": rel_stats(e1, e0), "connector": rel_stats(c1, c0)},
             "noise_pm1": {"encoder": rel_stats(en, e0), "connector": rel_stats(cn, c0),
                           "pixel_mad": float(np.abs(noisy.astype(np.int16) - v[0]).mean())},
             "next_frame": {"encoder": rel_stats(e_next, e0), "connector": rel_stats(c_next, c0),
                            "pixel_mad": float(np.abs(v[1].astype(np.int16) - v[0]).mean())}}
        mads = np.abs(np.diff(v.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
        rep = np.flatnonzero(mads < 0.5)
        if len(rep):
            j = int(rep[0])
            ea, ca, _ = vs.forward(vs.pixels(v[j]))
            eb, cb, _ = vs.forward(vs.pixels(v[j + 1]))
            r["repeated_frame"] = {"frames": [j, j + 1], "pixel_mad": float(mads[j]),
                                   "encoder": rel_stats(eb, ea), "connector": rel_stats(cb, ca)}
        assert r["same_frame"]["connector"]["max"] == 0.0, r["same_frame"]
        sens[name.replace(" .mp4", ".mp4")] = r
        torch.cuda.empty_cache()
    out["sensitivity"] = sens

    path = ROOT / "results" / "tokens" / "gate_checks.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

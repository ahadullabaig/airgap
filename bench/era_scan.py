"""Scan every ERA clip for the properties that matter before using it in X0.

Per clip: timestamp health (missing 1/24 s slots, broken pts), repeated frames,
keyframe pulses, static blocks and dark bars (overlays and letterboxing that
flatter a gate), camera motion (shotmotion: patch phase correlation + fitted
similarity transform, independent of K0), and possible shot changes
(colour-histogram events, validated on the hand-labelled Sheboygan clip with
--validate).

Writes results/era/scan.json (numbers and file names only; ERA itself is
academic-use, no redistribution, so no frames leave data/) and per-clip
signals to results/era/raw/signals.npz (git-ignored).

Example:
    .venv/bin/python -I bench/era_scan.py --era data/era
    .venv/bin/python -I bench/era_scan.py --validate
"""

import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import shotmotion as sm  # noqa: E402

PAIR_S = 0.25  # frame spacing for motion and shot signals
MOTION_STEP = 2  # every 2nd pair for motion (shot signals use every pair)
RELIABLE_INLIERS = 0.5  # a motion estimate counts only if this share of patches fit one transform
DUP_MAD = (0.25, 0.5, 1.0)  # consecutive-frame grey MAD below these: candidate repeated frame
STATIC_BLOCK_MAD = 1.5  # block never moves more than this (grey, 0-255) from frame 0
TEXTURE_STD = 6.0  # a static block with this much spatial detail (grey std) is not plain sky or water
BLOCKS = 32  # 32 x 32 blocks, the encoder's patch grid
BAR_MEAN, BAR_STD = 24.0, 4.0  # dark, flat edge rows/columns


def era_files(era: Path) -> list[tuple[str, str, Path]]:
    out = []
    for split in ("Tra", "Test"):
        for p in sorted((era / "Videos" / split).glob("*/*.mp4")):
            out.append((split, p.parent.name, p))
    return out


def bar_px(mean_img: np.ndarray, std_img: np.ndarray) -> dict:
    """Dark, flat rows/columns from each edge, in working pixels (SIZE scale)."""
    def run(means, stds):
        n = 0
        for m, s in zip(means, stds):
            if m < BAR_MEAN and s < BAR_STD:
                n += 1
            else:
                break
        return n
    rm, rs = mean_img.mean(1), std_img.mean(1)
    cm, cs = mean_img.mean(0), std_img.mean(0)
    return {"top": run(rm, rs), "bottom": run(rm[::-1], rs[::-1]),
            "left": run(cm, cs), "right": run(cm[::-1], cs[::-1])}


def scan_one(job):
    split, cls, path = job
    rec = {"split": split, "cls": cls, "file": path.name, "id": path.name.replace(" .mp4", ".mp4")}
    try:
        dec = sm.decode(str(path))
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:200]
        return rec, None
    g = dec.grey
    n = len(g)
    fps = dec.fps
    rec.update(frames=n, fps=fps, width=dec.width, height=dec.height)

    # timestamps, in slots of 1/fps
    pts = dec.pts_s
    dt = np.diff(pts) * fps
    slots = np.rint(dt).astype(int)
    rec["pts_span_s"] = float(pts[-1] - pts[0]) if n > 1 else 0.0
    rec["pts_nonincreasing"] = int((dt <= 0).sum())
    rec["missing_slots"] = int(np.clip(slots - 1, 0, None).sum())
    rec["max_gap_slots"] = int(slots.max()) if n > 1 else 0
    rec["pts_broken"] = bool(rec["pts_span_s"] > 2 * n / fps or rec["pts_nonincreasing"] > 0)

    # consecutive-frame difference, repeated frames, keyframe pulses
    mad1 = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    rec["mad1_median"] = float(np.median(mad1))
    for thr in DUP_MAD:
        rec[f"dup_lt_{thr:g}"] = int((mad1 < thr).sum())
    key_idx = np.flatnonzero(dec.key)
    rec["keyframes"] = int(len(key_idx))
    kp = key_idx[key_idx > 0] - 1  # pair (i-1, i) entering a keyframe
    rec["key_pulse"] = float(np.median(mad1[kp]) / max(np.median(mad1), 1e-6)) if len(kp) else None

    # static blocks and bars
    bs = sm.SIZE // BLOCKS
    dev = np.abs(g - g[0]).reshape(n, BLOCKS, bs, BLOCKS, bs).mean(axis=(2, 4)).max(axis=0)
    static = dev < STATIC_BLOCK_MAD
    texture = g[0].reshape(BLOCKS, bs, BLOCKS, bs).std(axis=(1, 3))
    rec["static_block_frac"] = float(static.mean())
    rec["static_textured_frac"] = float((static & (texture >= TEXTURE_STD)).mean())
    rec["bars_px"] = bar_px(g.mean(0), g.std(0))

    # camera motion and shot signals
    k = max(1, round(PAIR_S * fps))
    ps_all = sm.pair_signals(dec, k=k, step=1) if n > k else None
    if ps_all is None:
        rec["error"] = "too few frames"
        return rec, None
    sel = slice(None, None, MOTION_STEP)
    mot, inl = ps_all.motion_fw_per_s[sel], ps_all.inlier_frac[sel]
    ok = (inl >= RELIABLE_INLIERS) & np.isfinite(mot)
    rec["motion"] = {
        "k": k, "pairs": int(len(mot)), "reliable_frac": float(ok.mean()),
        "median_fw_per_s": float(np.median(mot[ok])) if ok.any() else None,
        "p90_fw_per_s": float(np.percentile(mot[ok], 90)) if ok.any() else None,
    }
    ev = sm.shot_events(ps_all, fps)
    rec["shot"] = {"hist_max": float(ps_all.hist_l1.max()), "hist_median": float(np.median(ps_all.hist_l1)),
                   "events": [[round(a, 2), round(b, 2), round(p, 3)] for a, b, p in ev]}
    raw = {"mad1": mad1.astype(np.float32), "pts": pts, "key": dec.key, "hist_l1": ps_all.hist_l1.astype(np.float32),
           "inlier": ps_all.inlier_frac.astype(np.float32), "motion": ps_all.motion_fw_per_s.astype(np.float32)}
    return rec, raw


def validate() -> None:
    """Shot-change recall on the hand-labelled Sheboygan boundaries (cuts and dissolves)."""
    clip = ROOT / "data" / "clips" / "sheboygan_falls.webm"
    seg = json.loads((ROOT / "bench" / "x0_segments" / "sheboygan_falls.json").read_text())
    dec = sm.decode(str(clip))
    k = max(1, round(PAIR_S * dec.fps))
    ps = sm.pair_signals(dec, k=k, step=1)
    first_aerial = seg["segments"][1]["start"]
    bounds = [s["start"] for s in seg["segments"][1:]]
    ev = [e for e in sm.shot_events(ps, dec.fps) if e[1] >= first_aerial - 0.5]
    near = lambda e, b: e[0] - 0.3 <= b <= e[1] + 0.3  # noqa: E731
    hit = [b for b in bounds if any(near(e, b) for e in ev)]
    fa = [e for e in ev if not any(near(e, b) for b in bounds)]
    out = {"clip": clip.name, "threshold": sm.HIST_EVENT, "pair_s": PAIR_S, "k": k,
           "boundaries": len(bounds), "found": len(hit), "missed": [b for b in bounds if b not in hit],
           "false_events": [[round(a, 2), round(b, 2), round(p, 3)] for a, b, p in fa],
           "aerial_seconds": round(seg["segments"][-1]["end"] - first_aerial, 1)}
    print(json.dumps(out, indent=1))
    path = ROOT / "results" / "era" / "shot_detector_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", default=str(ROOT / "data" / "era"))
    ap.add_argument("--out", default=str(ROOT / "results" / "era"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N clips only (testing)")
    ap.add_argument("--validate", action="store_true", help="check shot detector on the labelled clip and exit")
    args = ap.parse_args()
    if args.validate:
        validate()
        return

    jobs = era_files(Path(args.era))
    if args.limit:
        jobs = jobs[:: max(1, len(jobs) // args.limit)][: args.limit]
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recs, raws = [], {}
    with Pool(args.workers) as pool:
        for i, (rec, raw) in enumerate(pool.imap(scan_one, jobs, chunksize=4)):
            recs.append(rec)
            if raw is not None:
                for key, v in raw.items():
                    raws[f"{rec['split']}/{rec['cls']}/{rec['id']}/{key}"] = v
            if (i + 1) % 250 == 0:
                print(f"{i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    meta = {"clips": len(recs), "errors": sum(1 for r in recs if "error" in r), "seconds": round(time.time() - t0, 1),
            "params": {"SIZE": sm.SIZE, "PATCH": sm.PATCH, "GRID": sm.GRID, "PEAK_MIN": sm.PEAK_MIN,
                       "TOL_PX": sm.TOL_PX, "HIST_EVENT": sm.HIST_EVENT, "PAIR_S": PAIR_S,
                       "MOTION_STEP": MOTION_STEP, "RELIABLE_INLIERS": RELIABLE_INLIERS,
                       "STATIC_BLOCK_MAD": STATIC_BLOCK_MAD, "TEXTURE_STD": TEXTURE_STD, "BLOCKS": BLOCKS, "BAR_MEAN": BAR_MEAN,
                       "BAR_STD": BAR_STD},
            "env": {"python": platform.python_version(), "numpy": np.__version__}}
    (out / "scan.json").write_text(json.dumps({"meta": meta, "clips": recs}, indent=0) + "\n")
    np.savez_compressed(out / "raw" / "signals.npz", **raws)
    print(json.dumps(meta, indent=1))
    print(f"wrote {out / 'scan.json'}")


if __name__ == "__main__":
    main()

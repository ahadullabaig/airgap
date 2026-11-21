"""X0 on ERA: reference-mode keep-rates for every clip of one split, as the encoder sees it.

Each frame goes through the SmolVLM2 image processor (512x512, no splitting) and
into the reference-mode gate (bench/x0core.py) at strides 1, 2, 4, 8, 24, which
are 24, 12, 6, 3 and 1 fps at ERA's 24 fps. Frame i counts as time i/24: frames
are used as decoded, so a missing timestamp slot is a slightly longer step.

Conventions, because ERA clips are short (about 5.75 s):
  steady      mean keep over the counted samples; the first sample, which only
              initialises the reference, is left out. Understates a long stream:
              slow change has had less time to build up.
  with_first  the first sample counted as a full frame (keep 1). Overstates a
              long stream. The long-stream value lies between the two; at 1 fps
              a clip has only 5 counted samples, so the gap is wide.
  half1/half2 steady keep in the first and second half of each clip, to show
              how much the fresh reference flatters the first half.

Clips are pooled with equal weight; spreads are across clips (ERA clips may share
source videos, so no confidence intervals). Slices: hover (results/era/hover_tra.json),
hover at cut-offs 0.010 and 0.015, moving and unreliable (each with and without a
shot event), hover by class, and the 100 hand-labelled clips by eye label.

Per-clip results go to <out>/raw/per_clip.jsonl as they finish (git-ignored);
--resume skips clips already there. The summary goes to <out>/summary.json.

Example:
    .venv/bin/python -I bench/x0_era.py --workers 8
    .venv/bin/python -I bench/x0_era.py --aggregate-only
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import av  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import era_report as er  # noqa: E402
from k0lib import K0, changemap_numpy  # noqa: E402
from x0core import RefGate, model_view  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "SmolVLM2-500M-Video-Instruct"
NATIVE_FPS = 24.0
STRIDES = (1, 2, 4, 8, 24)
THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
BLOCKS = 1024  # 32 x 32 blocks per frame
HIST_BINS = np.linspace(0.0, 1.0, 21)

_W: dict = {}


def init_worker(model: str) -> None:
    import torch
    from transformers import AutoImageProcessor
    torch.set_num_threads(1)
    ip = AutoImageProcessor.from_pretrained(model, local_files_only=True)
    _W.update(ip=ip, mean=np.array(ip.image_mean, np.float32), std=np.array(ip.image_std, np.float32),
              k0=K0(threads=1), checked=False)


def run_clip(job: dict) -> dict:
    k0 = _W["k0"]
    gate = RefGate(k0, STRIDES, THRESHOLDS)
    check, prev, n = None, None, 0
    t0 = time.perf_counter()
    with av.open(job["path"]) as c:
        st = c.streams.video[0]
        for i, fr in enumerate(c.decode(st)):
            f = model_view(_W["ip"], fr.to_image(), _W["mean"], _W["std"])
            if not _W["checked"] and prev is not None:  # once per worker: ctypes hand-off and serial == omp
                m = k0.changemap(f, prev)
                check = {"max_abs_diff_vs_numpy": float(np.abs(changemap_numpy(f, prev) - m).max()),
                         "serial_equals_omp": bool(np.array_equal(k0.changemap(f, prev, impl="serial"), m))}
                _W["checked"] = True
            gate.push(i, f)
            prev, n = f, n + 1
    if n != job["frames"]:
        raise RuntimeError(f'{job["id"]}: decoded {n} frames, scan.json has {job["frames"]}')
    keep = {f"{s}/{t:g}": [round(v * BLOCKS) for v in gate.keep[(s, t)]] for s in STRIDES for t in THRESHOLDS}
    return {"split": job["split"], "cls": job["cls"], "id": job["id"], "frames": n,
            "index": {str(s): gate.index[s] for s in STRIDES}, "keep_blocks": keep, "check": check,
            "seconds": round(time.perf_counter() - t0, 2)}


def clip_stats(rec: dict, s: int, t: float) -> dict:
    k = np.asarray(rec["keep_blocks"][f"{s}/{t:g}"], dtype=np.float64) / BLOCKS
    idx = np.asarray(rec["index"][str(s)])
    half = rec["frames"] / 2
    return {"values": k, "steady": float(k.mean()) if len(k) else np.nan,
            "with_first": float((1.0 + k.sum()) / (len(k) + 1)),
            "half1": float(k[idx < half].mean()) if (idx < half).any() else np.nan,
            "half2": float(k[idx >= half].mean()) if (idx >= half).any() else np.nan}


def pooled(stats: list[dict], fps: float) -> dict:
    steady = np.array([x["steady"] for x in stats])
    steady = steady[np.isfinite(steady)]
    wf = np.array([x["with_first"] for x in stats])
    h = np.array([(x["half1"], x["half2"]) for x in stats])
    h = h[np.isfinite(h).all(axis=1)]
    vals = np.concatenate([x["values"] for x in stats]) if stats else np.zeros(0)
    if len(steady) == 0:
        return {"clips": 0}
    r4 = lambda v: round(float(v), 4)  # noqa: E731
    return {
        "clips": len(steady),
        "steady_mean": r4(steady.mean()), "steady_median": r4(np.median(steady)),
        "steady_p10": r4(np.percentile(steady, 10)), "steady_p90": r4(np.percentile(steady, 90)),
        "with_first_mean": r4(wf.mean()),
        "cost_steady": r4(steady.mean() * fps), "cost_with_first": r4(wf.mean() * fps),
        "half1_mean": r4(h[:, 0].mean()) if len(h) else None, "half2_mean": r4(h[:, 1].mean()) if len(h) else None,
        "samples": int(len(vals)),
        "sample_hist": np.histogram(vals, bins=HIST_BINS)[0].tolist(),
    }


def aggregate(out: Path, split: str) -> dict:
    scan = {er.clip_key(r): r for r in json.loads((ROOT / "results" / "era" / "scan.json").read_text())["clips"]}
    recs = {}
    for line in (out / "raw" / "per_clip.jsonl").read_text().splitlines():
        r = json.loads(line)
        recs[(r["split"], r["cls"], r["id"])] = r
    hover = {tuple(k) for k in json.loads((ROOT / "results" / "era" / "hover_tra.json").read_text())["clips"]}
    labels = json.loads((ROOT / "bench" / "era_labels" / "tra_sample_seed0.json").read_text())["clips"]

    slices: dict[str, list] = defaultdict(list)
    for key, rec in recs.items():
        sr = scan[key]
        b = er.bucket(sr)
        fl = er.shot_flagged(sr)
        if key in hover:
            slices["hover"].append(key)
            slices[f"hover_by_class/{key[1]}"].append(key)
        for c in (0.010, 0.015):
            if er.bucket(sr, c) == "hover" and not fl:
                slices[f"hover_cut_{c:g}"].append(key)
        if b != "hover":
            slices[b].append(key)
            slices[f"{b}_{'flagged' if fl else 'unflagged'}"].append(key)
    for lab in labels:
        key = (lab["split"], lab["cls"], lab["id"])
        if key in recs:
            slices[f"eye/{lab['camera']}"].append(key)
    eye_s_rejected = [(l["split"], l["cls"], l["id"]) for l in labels
                      if l["camera"] == "S" and (l["split"], l["cls"], l["id"]) not in hover]

    cells = [(s, t) for s in STRIDES for t in THRESHOLDS]
    summary = {}
    for name, keys in sorted(slices.items()):
        summary[name] = {}
        for s, t in cells:
            summary[name][f"{s}/{t:g}"] = pooled([clip_stats(recs[k], s, t) for k in keys], NATIVE_FPS / s)
    per_clip_eye_s = {k[2]: {f"{s}/{t:g}": round(clip_stats(recs[k], s, t)["steady"], 4) for s, t in cells}
                      for k in eye_s_rejected if k in recs}
    checks = [r["check"] for r in recs.values() if r.get("check")]
    return {"slices": summary, "eye_still_rejected_by_rule": per_clip_eye_s,
            "slice_sizes": {k: len(v) for k, v in sorted(slices.items())},
            "crosscheck": {"workers_checked": len(checks),
                           "max_abs_diff_vs_numpy": max((c["max_abs_diff_vs_numpy"] for c in checks), default=None),
                           "serial_equals_omp": all(c["serial_equals_omp"] for c in checks) if checks else None},
            "clips_processed": len(recs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="Tra")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--era", default=str(ROOT / "data" / "era"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="every k-th clip, N clips in total (timing runs)")
    ap.add_argument("--resume", action="store_true", help="skip clips already in raw/per_clip.jsonl")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out", help="default results/x0/era_<split>")
    args = ap.parse_args()
    if args.split != "Tra":
        sys.exit("X0 uses the training split only; the test split is reserved for E1-E4")
    out = Path(args.out) if args.out else ROOT / "results" / "x0" / f"era_{args.split.lower()}"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    jsonl = out / "raw" / "per_clip.jsonl"

    t_run = None
    if args.aggregate_only and (out / "summary.json").exists():  # keep the timing of the run that made the data
        t_run = json.loads((out / "summary.json").read_text()).get("timing_s")
    if not args.aggregate_only:
        scan = [r for r in json.loads((ROOT / "results" / "era" / "scan.json").read_text())["clips"]
                if r["split"] == args.split and "error" not in r]
        if args.limit:
            scan = scan[:: max(1, len(scan) // args.limit)][: args.limit]
        done = set()
        if args.resume and jsonl.exists():
            done = {(r["split"], r["cls"], r["id"]) for r in map(json.loads, jsonl.read_text().splitlines())}
        elif jsonl.exists():
            jsonl.unlink()
        jobs = [{"split": r["split"], "cls": r["cls"], "id": r["id"], "frames": r["frames"],
                 "path": str(Path(args.era) / "Videos" / r["split"] / r["cls"] / r["file"])}
                for r in scan if er.clip_key(r) not in done]
        t0 = time.perf_counter()
        with Pool(args.workers, initializer=init_worker, initargs=(args.model,)) as pool, jsonl.open("a") as fh:
            for i, rec in enumerate(pool.imap_unordered(run_clip, jobs, chunksize=1)):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                    el = time.perf_counter() - t0
                    print(f"{i + 1}/{len(jobs)} clips  {el:.0f} s  ({el / (i + 1):.2f} s/clip)", flush=True)
        t_run = round(time.perf_counter() - t0, 1)

    agg = aggregate(out, args.split)
    cx = agg["crosscheck"]
    if cx["max_abs_diff_vs_numpy"] is None or cx["max_abs_diff_vs_numpy"] > 1e-3 or not cx["serial_equals_omp"]:
        sys.exit(f"K0 cross-check failed or missing: {cx}")
    import transformers
    summary = {
        "experiment": "X0 on ERA",
        "split": args.split,
        "view": "SmolVLM2 image processor output (512x512, no splitting), rounded back to uint8; "
                "K0 with 16x16 blocks = the encoder's 32x32 patch grid",
        "mode": "reference (bench/x0core.py RefGate)",
        "fps": {str(s): NATIVE_FPS / s for s in STRIDES},
        "thresholds_mad": list(THRESHOLDS),
        "cell_key": "<stride>/<threshold>",
        "conventions": {
            "time": "frame index / 24; frames used as decoded (missing timestamp slots not filled)",
            "steady": "mean keep over counted samples, first sample (reference init) excluded; clips weighted equally",
            "with_first": "first sample counted as a full frame",
            "cost": "full-frame equivalents per second of video = mean keep x fps",
            "sample_hist": "per-sample keep, pooled over clips, 20 bins of 0.05 on [0, 1]",
            "caveats": ["repeated frames lower the 24 fps per-sample mean slightly: dropping repeated samples raises "
                        "moving from 42.7% to 45.0% (threshold 8; 5.1% of pairs repeat) and hover from 27.4% to 27.7% "
                        "in the 32 hover clips where repeats are measurable (1.3% repeat); lower rates are less affected",
                        "thresholds 1-2 are mostly codec noise on still footage",
                        "pixel proxy: no link to encoder output error yet (needs token-level traces)"],
        },
        **agg,
        "timing_s": t_run,
        "env": {"transformers": transformers.__version__, "numpy": np.__version__, "pyav": av.__version__,
                "python": platform.python_version()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out / 'summary.json'}  ({agg['clips_processed']} clips)")


if __name__ == "__main__":
    main()

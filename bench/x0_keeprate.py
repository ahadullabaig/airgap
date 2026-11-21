"""X0: how much does real aerial footage change between frames, as the encoder sees it?

Every frame goes through the SmolVLM2 image processor (512x512, no splitting),
so K0's 16x16 blocks line up one-to-one with the encoder's patch tokens.

Two modes, for each sampling stride (native frames between samples) and each
threshold on K0's block mean absolute difference (0-255 scale):

  pair  K0(frame_t, frame_{t - stride}). Change between consecutive samples.
  ref   K0(frame_t, reference). The reference holds each block as it was when
        that block was last refreshed; blocks over the threshold are refreshed.
        This is what a gate actually pays: slow drift accumulates until it
        crosses the threshold, so high frame rates are not flattered.

Reported per mode, stride and threshold: keep-rate per sample (fraction of
blocks over threshold), and for ref mode the cost in full-frame equivalents
per second of video (mean keep-rate x sampling fps).

Edited footage has cuts; they are flagged from the native-rate global
difference and every statistic is given with and without samples that span
a cut. Repeated (duplicate) frames are counted.

Example:
    .venv/bin/python bench/x0_keeprate.py --video data/clips/sheboygan_falls.webm
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import platform
import sys
import time
from collections import deque
from pathlib import Path

import av
import numpy as np
import transformers
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from k0lib import K0, changemap_numpy  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "SmolVLM2-500M-Video-Instruct"
BLOCK = 16


def model_view(ip, img, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """The 512x512 RGB image the encoder sees, as C-contiguous HWC uint8."""
    pv = ip(images=[[img]], do_image_splitting=False, return_tensors="np")["pixel_values"]
    x = np.asarray(pv)[0, 0].astype(np.float32)  # CHW, normalised
    x = (x * std[:, None, None] + mean[:, None, None]) * 255.0
    x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(x.transpose(1, 2, 0))


def refresh_blocks(ref: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> None:
    """Copy the masked blocks of frame into ref, in place."""
    h, w, c = ref.shape
    rv = ref.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, c).transpose(0, 2, 1, 3, 4)
    fv = frame.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, c).transpose(0, 2, 1, 3, 4)
    rv[mask] = fv[mask]


def summarise(values) -> dict | None:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", help="clip name for output files (default: file stem)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--target-fps", type=float, nargs="+", default=[30, 15, 5, 1],
                    help="sampling rates to approximate; actual rates are native/stride")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--cut-min", type=float, default=30.0, help="cut if global MAD exceeds this ...")
    ap.add_argument("--cut-ratio", type=float, default=5.0, help="... and this multiple of the clip's median")
    ap.add_argument("--max-seconds", type=float, help="only process the first N seconds")
    ap.add_argument("--out", default=str(ROOT / "results" / "x0"))
    ap.add_argument("--threads", type=int, help="OpenMP threads for K0 (default: all)")
    ap.add_argument("--segments", help="hand-labelled shots JSON (default: bench/x0_segments/<name>.json if present)")
    args = ap.parse_args()

    name = args.name or Path(args.video).stem
    seg_path = Path(args.segments) if args.segments else ROOT / "bench" / "x0_segments" / f"{name}.json"
    segments = json.loads(seg_path.read_text()) if seg_path.exists() else None
    out_dir = Path(args.out) / name
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    k0 = K0(threads=args.threads)
    ip = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    mean, std = np.array(ip.image_mean, dtype=np.float32), np.array(ip.image_std, dtype=np.float32)
    thresholds = sorted(float(t) for t in args.thresholds)

    with av.open(args.video) as probe:
        vs = probe.streams.video[0]
        native_fps = float(vs.average_rate)
        src = {"path": args.video, "width": vs.width, "height": vs.height, "codec": vs.codec_context.name,
               "native_fps": round(native_fps, 4)}
    strides = sorted({max(1, round(native_fps / f)) for f in args.target_fps})

    ring: deque[np.ndarray] = deque(maxlen=max(strides) + 1)
    times: list[float] = []
    global_mad: list[float] = []   # stride-1 pair (i-1, i), stored at index i-1
    max_mad: list[float] = []
    pair_maps = {s: [] for s in strides}
    pair_idx = {s: [] for s in strides}
    refs: dict[tuple[int, float], np.ndarray] = {}
    ref_keep = {(s, t): [] for s in strides for t in thresholds}
    ref_idx = {s: [] for s in strides}
    crosscheck = {"pairs": 0, "max_abs_diff_vs_numpy": 0.0, "serial_equals_omp": None}
    refresh_checked = False

    t_start = time.perf_counter()
    t_pre = 0.0
    with av.open(args.video) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else i / native_fps
            if args.max_seconds is not None and t > args.max_seconds:
                break
            s0 = time.perf_counter()
            f = model_view(ip, frame.to_image(), mean, std)
            t_pre += time.perf_counter() - s0
            ring.append(f)
            times.append(t)

            if i >= 1:
                m1 = k0.changemap(f, ring[-2])
                global_mad.append(float(m1.mean()))
                max_mad.append(float(m1.max()))
                if crosscheck["pairs"] < 20:  # verify the ctypes hand-off on real frames
                    ref_np = changemap_numpy(f, ring[-2])
                    crosscheck["max_abs_diff_vs_numpy"] = max(crosscheck["max_abs_diff_vs_numpy"],
                                                              float(np.abs(ref_np - m1).max()))
                    if crosscheck["pairs"] == 0:
                        crosscheck["serial_equals_omp"] = bool(
                            np.array_equal(k0.changemap(f, ring[-2], impl="serial"), m1))
                    crosscheck["pairs"] += 1

            for s in strides:
                if i % s:
                    continue
                if i >= s:
                    pair_maps[s].append(m1 if s == 1 else k0.changemap(f, ring[-1 - s]))
                    pair_idx[s].append(i)
                first = (s, thresholds[0]) not in refs
                if not first:
                    ref_idx[s].append(i)
                for thr in thresholds:
                    key = (s, thr)
                    if key not in refs:
                        refs[key] = f.copy()
                        continue
                    m = k0.changemap(f, refs[key])
                    mask = m > thr
                    if mask.any():
                        refresh_blocks(refs[key], f, mask)
                        if not refresh_checked:  # the strided view assignment really writes into ref
                            jy, jx = np.argwhere(mask)[0]
                            blk = (slice(jy * BLOCK, (jy + 1) * BLOCK), slice(jx * BLOCK, (jx + 1) * BLOCK))
                            assert np.array_equal(refs[key][blk], f[blk]), "refresh_blocks did not write"
                            refresh_checked = True
                    ref_keep[key].append(float(mask.mean()))

    elapsed = time.perf_counter() - t_start
    n_frames = len(times)
    if n_frames < 2:
        sys.exit("too few frames")
    if crosscheck["max_abs_diff_vs_numpy"] > 1e-3 or crosscheck["serial_equals_omp"] is False:
        sys.exit(f"K0 cross-check failed: {crosscheck}")

    # Cuts: a stride-1 pair whose global difference is both large and far above the clip's median.
    g = np.asarray(global_mad)
    cut_level = max(args.cut_min, args.cut_ratio * float(np.median(g)))
    cut_frames = [k + 1 for k in np.flatnonzero(g > cut_level)]  # frame index after the cut
    cut_set = np.zeros(n_frames + 1, dtype=np.int64)
    for c in cut_frames:
        cut_set[c] = 1
    cum = np.cumsum(cut_set)

    def spans_cut(i: int, s: int) -> bool:  # any cut in (i - s, i]
        return bool(cum[i] - cum[i - s] > 0)

    dup_pairs = [k + 1 for k, v in enumerate(max_mad) if v < 0.5]

    # Hand-labelled shots: a sample covering (t[i - s], t[i]] takes its shot's label, unless the
    # interval comes within the margin of a shot boundary, in which case it is a "transition".
    if segments:
        margin = float(segments.get("boundary_margin_s", 0.5))
        segs = segments["segments"]
        boundaries = sorted({float(x["start"]) for x in segs} | {float(x["end"]) for x in segs})
        boundaries = [b for b in boundaries if times[0] < b < times[-1]]

        def label_of(i: int, s: int) -> str:
            t0, t1 = times[i - s], times[i]
            if any(t0 - margin < b < t1 + margin for b in boundaries):
                return "transition"
            for x in segs:
                if float(x["start"]) <= t1 < float(x["end"]):
                    return x["label"]
            return "unlabelled"
    else:
        def label_of(i: int, s: int) -> str:
            return "unlabelled"

    def with_cost(stats: dict | None, fps: float) -> dict | None:
        if stats:
            stats["frame_equiv_per_s"] = round(stats["mean"] * fps, 4)
        return stats

    pair_out, ref_out = {}, {}
    for s in strides:
        fps = native_fps / s
        maps = np.stack(pair_maps[s]) if pair_maps[s] else np.zeros((0, 32, 32), np.float32)
        clean = np.array([not spans_cut(i, s) for i in pair_idx[s]], dtype=bool)
        ref_clean = np.array([not spans_cut(i, s) for i in ref_idx[s]], dtype=bool)
        pair_lab = np.array([label_of(i, s) for i in pair_idx[s]])
        ref_lab = np.array([label_of(i, s) for i in ref_idx[s]])
        pair_out[str(s)] = {"fps": round(fps, 3), "by_threshold": {}}
        ref_out[str(s)] = {"fps": round(fps, 3), "by_threshold": {}}
        for thr in thresholds:
            keep = (maps > thr).mean(axis=(1, 2)) if len(maps) else np.zeros(0)
            pair_out[str(s)]["by_threshold"][f"{thr:g}"] = {
                "all": summarise(keep),
                "no_cuts": summarise(keep[clean]) if len(keep) else None,
                "by_label": {lab: summarise(keep[pair_lab == lab]) for lab in sorted(set(pair_lab))},
            }
            rk = np.asarray(ref_keep[(s, thr)])
            ref_out[str(s)]["by_threshold"][f"{thr:g}"] = {
                "all": with_cost(summarise(rk), fps),
                "no_cuts": with_cost(summarise(rk[ref_clean]), fps) if len(rk) else None,
                "by_label": {lab: with_cost(summarise(rk[ref_lab == lab]), fps) for lab in sorted(set(ref_lab))},
            }
        np.savez_compressed(out_dir / "raw" / f"pair_maps_stride{s}.npz",
                            maps=maps.astype(np.float16), frame_index=np.asarray(pair_idx[s]))
    np.savez_compressed(out_dir / "raw" / "stride1_global.npz", global_mad=g, max_mad=np.asarray(max_mad),
                        times=np.asarray(times))
    np.savez_compressed(out_dir / "raw" / "ref_keep.npz",
                        **{f"s{s}_t{t:g}": np.asarray(v) for (s, t), v in ref_keep.items()},
                        **{f"s{s}_index": np.asarray(ref_idx[s]) for s in strides})

    summary = {
        "experiment": "X0",
        "status": "pipeline check on one clip; not gate evidence",
        "clip": name,
        "source": src,
        "frames_processed": n_frames,
        "seconds_processed": round(times[-1] - times[0], 3),
        "view": "SmolVLM2 image processor output (512x512, no splitting), rounded back to uint8; "
                "K0 with 16x16 blocks = the encoder's 32x32 patch grid",
        "thresholds_mad": thresholds,
        "strides": [{"stride": s, "fps": round(native_fps / s, 3)} for s in strides],
        "cuts": {
            "rule": f"stride-1 global MAD > max({args.cut_min}, {args.cut_ratio} x clip median)",
            "level": round(cut_level, 3),
            "count": len(cut_frames),
            "times_s": [round(times[c], 3) for c in cut_frames],
            "note": "gradual transitions (dissolves, fades) are not detected",
        },
        "duplicates": {
            "rule": "stride-1 pairs whose largest block MAD is below 0.5",
            "count": len(dup_pairs),
            "first_times_s": [round(times[d], 3) for d in dup_pairs[:20]],
        },
        "segments": None if not segments else {
            "path": str(seg_path.relative_to(ROOT)) if seg_path.is_relative_to(ROOT) else str(seg_path),
            "boundary_margin_s": margin,
            "seconds_by_label": {
                lab: round(sum(min(float(x["end"]), times[-1]) - float(x["start"])
                               for x in segments["segments"] if x["label"] == lab), 2)
                for lab in sorted({x["label"] for x in segments["segments"]})
            },
        },
        "crosscheck": crosscheck,
        "pair": pair_out,
        "ref": ref_out,
        "timing_s": {"total": round(elapsed, 1), "preprocess": round(t_pre, 1)},
        "env": {
            "k0_threads": k0.threads,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pyav": av.__version__,
            "python": platform.python_version(),
            "image_processor": type(ip).__name__,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_dir / 'summary.json'} ({n_frames} frames, {elapsed:.0f} s)")


if __name__ == "__main__":
    main()

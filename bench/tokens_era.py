"""Token-level traces on ERA: how a pixel gate's choices show up in the encoder.

For a seeded sample of training clips (camera still, moving, untrackable, plus
the hand-labelled still clips the hover rule rejected), every frame's encoder
view (bench/x0core.model_view) goes through the standalone vision tower and
connector (bench/visionstack.py, fp32, TF32 off).

1. Dense trace (exported to traces/, git-ignored). For each frame and stride s in
   1, 2, 4, 8: per-token relative change ||h_i - h_{i-s}|| / ||h_{i-s}|| at the
   embedding, after each of the 12 layers and after the post-layernorm, the
   connector's 64 output tokens, and K0's pixel block MAD for the same pair.
   Change against the previous sample understates what a gate pays (as pair
   mode did in X0).

2. Gate error. The reference-mode pixel gate (same rule as X0) at strides 1 and
   4 (24 and 6 fps) and thresholds 1-32, two tiers:
     A  stale input: the dense encoder on the gate's reference image. A floor:
        it can rule a threshold out, never in.
     B  stale rows (G1, design option 1): the pixel mask is held through every
        layer; refreshed rows are computed against all keys and values, the
        other rows keep their cached output from when they were last refreshed.
   Errors are per-token relative errors against the dense output of the true
   frame, at the encoder output (refreshed and unrefreshed tokens separately)
   and at the connector output (what the language model sees).

3. Equal-cost yardsticks, on the connector output at 24 fps (every method holds
   its last output between updates): dense at a lower rate with the last output
   held, and the bf16-vs-fp32 floor from results/tokens/checks.json.

Needs results/x0/era_tra/raw/per_clip.jsonl (bench/x0_era.py) and results/tokens/
checks.json (bench/visionstack_check.py). About 3 minutes per clip on an RTX 3050
Laptop, 39 clips; per-clip results are appended as they finish and a rerun skips
clips already done.

Example:
    .venv/bin/python -I bench/tokens_era.py
    .venv/bin/python -I bench/tokens_era.py --aggregate-only
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import era_report as er  # noqa: E402
from k0lib import K0  # noqa: E402
from visionstack import TOKENS, VisionStack  # noqa: E402
from x0core import model_view, refresh_blocks  # noqa: E402

MODEL = ROOT / "models" / "SmolVLM2-500M-Video-Instruct"
TRACE_STRIDES = (1, 2, 4, 8)
GATE_STRIDES = (1, 4)
HOLD_STRIDES = (1, 2, 4, 8, 24)
THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
EPS = (0.01, 0.03, 0.1, 0.3)  # token-level change thresholds for the dense trace
PIXEL_MASK_THR = 8.0  # inside/outside split of the dense trace
PER_GROUP = 12
SEED = 2
NATIVE_FPS = 24.0


def pick_clips() -> list[dict]:
    scan = [r for r in json.loads((ROOT / "results" / "era" / "scan.json").read_text())["clips"]
            if r["split"] == "Tra" and "error" not in r]
    hover = {tuple(k) for k in json.loads((ROOT / "results" / "era" / "hover_tra.json").read_text())["clips"]}
    groups = {"hover": [r for r in scan if er.clip_key(r) in hover],
              "moving": [r for r in scan if er.bucket(r) == "moving"],
              "unreliable": [r for r in scan if er.bucket(r) == "unreliable"]}
    rng = random.Random(SEED)
    out = []
    for g, rows in groups.items():
        for r in sorted(rng.sample(rows, PER_GROUP), key=er.clip_key):
            out.append({**r, "group": g})
    labels = json.loads((ROOT / "bench" / "era_labels" / "tra_sample_seed0.json").read_text())["clips"]
    by = {er.clip_key(r): r for r in scan}
    for lab in labels:
        key = (lab["split"], lab["cls"], lab["id"])
        if lab["camera"] == "S" and key not in hover:
            out.append({**by[key], "group": "eye_still_rejected"})
    return out


def load_views(path: Path, ip, mean, std) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.stack([model_view(ip, f.to_image(), mean, std) for f in c.decode(c.streams.video[0])])


def tok_rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token relative change/error of a against b, shape (tokens,)."""
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).reshape(-1)


def dense_pass(vs: VisionStack, k0: K0, views: np.ndarray):
    """One encoder pass per frame. Returns dense encoder and connector outputs for every frame, and
    the dense trace arrays."""
    n = len(views)
    enc_all = torch.empty((n, TOKENS, 768), device=vs.device)
    con_all = torch.empty((n, 64, 960), device=vs.device)
    ring: dict[int, list[torch.Tensor]] = {}
    tr = np.full((n, len(TRACE_STRIDES), 14, TOKENS), np.nan, np.float16)
    trc = np.full((n, len(TRACE_STRIDES), 64), np.nan, np.float16)
    mad = np.full((n, len(TRACE_STRIDES), TOKENS), np.nan, np.float16)
    for i in range(n):
        enc, con, hs = vs.forward(vs.pixels(views[i]), keep_layers=True)
        enc_all[i], con_all[i] = enc[0], con[0]
        ring[i] = hs + [enc]
        for j, s in enumerate(TRACE_STRIDES):
            if i - s < 0:
                continue
            prev = ring[i - s]
            for layer in range(14):
                tr[i, j, layer] = tok_rel(ring[i][layer], prev[layer]).cpu().numpy()
            trc[i, j] = tok_rel(con_all[i], con_all[i - s]).cpu().numpy()
            mad[i, j] = k0.changemap(views[i], views[i - s]).reshape(-1)
        ring.pop(i - max(TRACE_STRIDES), None)
    return enc_all, con_all, tr, trc, mad


def gate_cell(vs, k0, views, enc_all, con_all, s: int, thr: float) -> dict:
    """Reference-mode pixel gate at stride s and threshold thr; tiers A and B."""
    n = len(views)
    ref = None
    cache: list[torch.Tensor] = []
    keep_counts, per = [], {"A": [], "B": []}
    con_out = {"A": {}, "B": {}}
    idx = list(range(0, n, s))
    for i in idx:
        f = views[i]
        if ref is None:  # first sample initialises the reference: dense, not counted
            ref = f.copy()
            enc, con, hs = vs.forward(vs.pixels(f), keep_layers=True)
            cache = hs
            con_out["A"][i] = con_out["B"][i] = con[0]
            continue
        mask = k0.changemap(f, ref) > thr
        refresh_blocks(ref, f, mask)
        keep_counts.append(int(mask.sum()))
        px = vs.pixels(ref)
        m = torch.from_numpy(mask.reshape(-1)).to(vs.device)
        # tier A: dense encoder on the gate's reference image
        encA, conA, _ = vs.forward(px)
        # tier B: mask held through the layers; unrefreshed rows keep their cached output
        h = torch.where(m[None, :, None], vs.embed(px), cache[0])
        cache[0] = h
        for layer in range(len(vs.layers)):
            h = torch.where(m[None, :, None], vs.layer(layer, h), cache[layer + 1])
            cache[layer + 1] = h
        encB, conB = vs.head(h)
        for tier, (e, c) in (("A", (encA, conA)), ("B", (encB, conB))):
            te = tok_rel(e[0], enc_all[i])
            tc = tok_rel(c[0], con_all[i])
            per[tier].append({
                "enc_mean": float(te.mean()), "enc_p95": float(te.quantile(0.95)),
                "enc_refreshed": float(te[m].mean()) if m.any() else None,
                "enc_unrefreshed": float(te[~m].mean()) if (~m).any() else None,
                "con_mean": float(tc.mean()), "con_p95": float(tc.quantile(0.95)),
            })
            con_out[tier][i] = c[0]
    # output stream at 24 fps: each frame gets the latest sample's output
    stream = {}
    for tier in ("A", "B"):
        errs = np.array([float(tok_rel(con_out[tier][(j // s) * s], con_all[j]).mean()) for j in range(n)])
        stream[tier] = {"mean": float(errs.mean()), "half1": float(errs[: n // 2].mean()),
                        "half2": float(errs[n // 2:].mean())}
    def avg(v, k):
        xs = [x[k] for x in v if x[k] is not None]
        return float(np.mean(xs)) if xs else None
    summ = {t: {k: avg(v, k) for k in v[0]} if v else None for t, v in per.items()}
    return {"keep_counts": keep_counts, "samples": summ, "stream": stream}


def hold_stream(con_all: torch.Tensor, s: int) -> dict:
    n = con_all.shape[0]
    errs = np.array([float(tok_rel(con_all[(j // s) * s], con_all[j]).mean()) for j in range(n)])
    return {"mean": float(errs.mean()), "half1": float(errs[: n // 2].mean()), "half2": float(errs[n // 2:].mean())}


def trace_summary(tr, trc, mad) -> dict:
    out = {}
    for j, s in enumerate(TRACE_STRIDES):
        t = tr[:, j].astype(np.float32)  # (n, 14, tokens)
        ok = np.isfinite(t[:, 0, 0])
        t, mk = t[ok], mad[ok, j].astype(np.float32) > PIXEL_MASK_THR
        layers = {}
        for layer in range(14):
            x = t[:, layer]
            layers[str(layer)] = {
                "inside_mean": float(x[mk].mean()) if mk.any() else None,
                "outside_mean": float(x[~mk].mean()) if (~mk).any() else None,
                "frac_gt": {f"{e:g}": float((x > e).mean()) for e in EPS},
            }
        c = trc[ok, j].astype(np.float32)
        out[str(s)] = {"pixel_keep": float(mk.mean()), "layers": layers,
                       "connector": {"mean": float(c.mean()), "frac_gt": {f"{e:g}": float((c > e).mean()) for e in EPS}}}
    return out


def run(args) -> None:
    from transformers import AutoImageProcessor
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out / "raw" / "per_clip.jsonl"
    done = set()
    if jsonl.exists():
        done = {(r["split"], r["cls"], r["id"]) for r in map(json.loads, jsonl.read_text().splitlines())}
    clips = [c for c in pick_clips() if er.clip_key(c) not in done]
    if args.limit:
        clips = clips[: args.limit]
    x0_raw = ROOT / "results" / "x0" / "era_tra" / "raw" / "per_clip.jsonl"
    if not x0_raw.exists():
        sys.exit(f"{x0_raw} missing: run bench/x0_era.py first (keep counts are checked against it)")
    x0 = {}
    for line in x0_raw.read_text().splitlines():
        r = json.loads(line)
        x0[(r["split"], r["cls"], r["id"])] = r["keep_blocks"]

    ip = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    mean, std = np.array(ip.image_mean, np.float32), np.array(ip.image_std, np.float32)
    vs = VisionStack(str(MODEL))
    k0 = K0()
    for ci, c in enumerate(clips):
        t0 = time.perf_counter()
        key = er.clip_key(c)
        views = load_views(Path(args.era) / "Videos" / c["split"] / c["cls"] / c["file"], ip, mean, std)
        assert len(views) == c["frames"], (key, len(views), c["frames"])
        enc_all, con_all, tr, trc, mad = dense_pass(vs, k0, views)
        np.savez_compressed(trace_dir / f'{c["cls"]}__{c["id"].replace(".mp4", "")}.npz',
                            token_change=tr, connector_change=trc, pixel_mad=mad,
                            strides=np.array(TRACE_STRIDES), layers=np.array(["embedding"] + [f"layer{i}" for i in
                                                                               range(1, 13)] + ["post_layernorm"]))
        rec = {"split": c["split"], "cls": c["cls"], "id": c["id"], "group": c["group"], "frames": len(views),
               "trace": trace_summary(tr, trc, mad), "cells": {}, "hold": {}}
        for s in GATE_STRIDES:
            for thr in THRESHOLDS:
                cell = gate_cell(vs, k0, views, enc_all, con_all, s, thr)
                expect = x0[key][f"{s}/{thr:g}"]
                assert cell["keep_counts"] == expect, (key, s, thr, "keep counts differ from X0")
                cell["keep"] = float(np.mean(cell.pop("keep_counts"))) / TOKENS
                rec["cells"][f"{s}/{thr:g}"] = cell
        for s in HOLD_STRIDES:
            rec["hold"][str(s)] = hold_stream(con_all, s)
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        with jsonl.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{ci + 1}/{len(clips)} {c['group']:18s} {c['id']:28s} {rec['seconds']:.0f} s", flush=True)
        del enc_all, con_all
        torch.cuda.empty_cache()


def aggregate(out: Path) -> dict:
    recs = [json.loads(l) for l in (out / "raw" / "per_clip.jsonl").read_text().splitlines()]
    checks = json.loads((ROOT / "results" / "tokens" / "checks.json").read_text())
    groups = sorted({r["group"] for r in recs})
    pct = lambda a, p: float(np.percentile(a, p)) if len(a) else None  # noqa: E731

    def pool(vals):
        a = np.array([v for v in vals if v is not None], dtype=float)
        return {"mean": float(a.mean()), "p10": pct(a, 10), "p90": pct(a, 90), "n": int(len(a))} if len(a) else None

    summ = {}
    for g in groups:
        R = [r for r in recs if r["group"] == g]
        cells = {}
        for key in R[0]["cells"]:
            s = int(key.split("/")[0])
            cl = [r["cells"][key] for r in R]
            keep = pool([c["keep"] for c in cl])
            cells[key] = {
                "keep": keep, "cost": keep["mean"] * NATIVE_FPS / s,
                **{f"{t}_{m}": pool([c["samples"][t][m] if c["samples"][t] else None for c in cl])
                   for t in ("A", "B") for m in ("enc_mean", "enc_p95", "enc_refreshed", "enc_unrefreshed",
                                                 "con_mean", "con_p95")},
                **{f"{t}_stream": pool([c["stream"][t]["mean"] for c in cl]) for t in ("A", "B")},
                **{f"{t}_stream_halves": [pool([c["stream"][t]["half1"] for c in cl])["mean"],
                                          pool([c["stream"][t]["half2"] for c in cl])["mean"]] for t in ("A", "B")},
            }
        hold = {s: {"cost": NATIVE_FPS / int(s), "stream": pool([r["hold"][s]["mean"] for r in R])} for s in R[0]["hold"]}
        trace = {}
        for s in R[0]["trace"]:
            trace[s] = {"pixel_keep": pool([r["trace"][s]["pixel_keep"] for r in R])["mean"], "layers": {}}
            for layer in R[0]["trace"][s]["layers"]:
                L = [r["trace"][s]["layers"][layer] for r in R]
                trace[s]["layers"][layer] = {
                    "inside_mean": pool([x["inside_mean"] for x in L])["mean"] if any(x["inside_mean"] is not None for x in L) else None,
                    "outside_mean": pool([x["outside_mean"] for x in L])["mean"] if any(x["outside_mean"] is not None for x in L) else None,
                    "frac_gt": {e: pool([x["frac_gt"][e] for x in L])["mean"] for e in L[0]["frac_gt"]},
                }
            trace[s]["connector"] = {"mean": pool([r["trace"][s]["connector"]["mean"] for r in R])["mean"],
                                     "frac_gt": {e: pool([r["trace"][s]["connector"]["frac_gt"][e] for r in R])["mean"]
                                                 for e in R[0]["trace"][s]["connector"]["frac_gt"]}}
        summ[g] = {"clips": [r["id"] for r in R], "cells": cells, "hold": hold, "trace": trace}
    return {"groups": summ,
            "floors": {"bf16_vs_fp32": checks["bf16_vs_fp32_rel_error"],
                       "uint8_roundtrip": checks["roundtrip_rel_error"],
                       "batch_size": checks["batch8_vs_batch1_rel_error"]},
            "clips_processed": len(recs), "seconds": round(sum(r["seconds"] for r in recs), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", default=str(ROOT / "data" / "era"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tokens" / "era_tra"))
    ap.add_argument("--trace-dir", default=str(ROOT / "traces" / "era_tra"))
    ap.add_argument("--limit", type=int, help="first N clips of the sample not yet done (smoke tests)")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        run(args)
    agg = aggregate(Path(args.out))
    summary = {
        "experiment": "token traces and gate error on ERA (training split)",
        "view": "encoder view as in X0 (uint8 512x512), fp32 vision tower + connector, TF32 off",
        "gate": "reference-mode pixel gate as in X0 (keep counts asserted equal to X0's per clip)",
        "tiers": {"A": "stale input: dense encoder on the gate's reference image (a floor)",
                  "B": "stale rows (G1 option 1): pixel mask held through all layers, unrefreshed rows cached"},
        "error": "per-token relative error ||x - x_dense|| / ||x_dense||, mean over tokens then samples then clips",
        "stream": "connector output at 24 fps, each method holding its latest output; mean over frames",
        "cost": "full-frame equivalents per second of video (keep x sampling fps); hold baseline costs its fps",
        "trace": "per-token relative change vs the previous sample (understates a gate); layer 0 = embedding, "
                 "1-12 = encoder layers, 13 = post-layernorm; inside/outside = pixel block MAD above/below 8",
        "gate_strides": list(GATE_STRIDES), "trace_strides": list(TRACE_STRIDES), "thresholds_mad": list(THRESHOLDS),
        "sample": f"{PER_GROUP} clips per group, random.Random({SEED}), plus hand-labelled still clips the hover rule rejected",
        **agg,
    }
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {Path(args.out) / 'summary.json'} ({agg['clips_processed']} clips)")


if __name__ == "__main__":
    main()

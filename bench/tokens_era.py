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
   Tier C is separate from the pixel gate: token-delta gates before the q/k/v
   projections and before the MLP at every layer, with exact attention for every
   row (G2-style gating with the exact update, design option 3), thresholds 0.05-0.5.
   Errors are per-token relative errors against the dense output of the true
   frame, at the encoder output (refreshed and unrefreshed tokens separately)
   and at the connector output (what the language model sees).

3. Equal-cost yardsticks, on the connector output at 24 fps (every method holds
   its last output between updates): dense at a lower rate with the last output
   held, and the bf16-vs-fp32 floor from results/tokens/checks.json.

Needs results/x0/era_tra/raw/per_clip.jsonl (bench/x0_era.py) and results/tokens/
checks.json (bench/visionstack_check.py). About 4 minutes per clip on an RTX 3050
Laptop, 39 clips (about 2.7 hours); per-clip results are appended as they finish and a rerun skips
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
EPS_C = (0.05, 0.1, 0.2, 0.3, 0.5)  # tier C token-delta thresholds; grid chosen by cost span only (two clips)
# Tier C cost model, per layer, in units of N*d^2 (N = 1024 tokens, d = 768): q/k/v projections 3 (gated by
# gate 1), attention 4N/d = 5.33 (changed rows plus the exact update of the other rows, about 2 x the gate-1
# share, capped at dense), output projection 1 (counted dense: every row's attention output can change),
# MLP 8 (gated by gate 2). Embedding, post-layernorm and connector are left out (small).
ATT_UNITS = 4 * 1024 / 768
DENSE_UNITS = 3 + ATT_UNITS + 1 + 8


def c_cost(k1: float, k2: float) -> float:
    """Estimated share of a dense layer's FLOPs for tier C given the shares through gates 1 and 2."""
    return (3 * k1 + ATT_UNITS * min(1.0, 2 * k1) + 1 + 8 * k2) / DENSE_UNITS
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


def pixel_gate(views: np.ndarray, k0: K0, s: int, thr: float):
    """Reference-mode pixel gate (same rule as X0). Yields (frame index, reference image, mask) per sample;
    the mask is None for the first sample, which initialises the reference. The reference is updated in
    place, so use it before advancing."""
    ref = None
    for i in range(0, len(views), s):
        f = views[i]
        if ref is None:
            ref = f.copy()
            yield i, ref, None
            continue
        mask = k0.changemap(f, ref) > thr
        refresh_blocks(ref, f, mask)
        yield i, ref, mask


def stream_errors(outputs: dict, con_all: torch.Tensor, s: int) -> tuple[list, list]:
    """Connector output at 24 fps, each frame taking the latest sample's output: per-frame mean and
    median over the 64 tokens of the relative error against dense."""
    mean, med = [], []
    for j in range(con_all.shape[0]):
        e = tok_rel(outputs[(j // s) * s], con_all[j])
        mean.append(round(float(e.mean()), 5))
        med.append(round(float(e.median()), 5))
    return mean, med


def stream_stats(mean: list, med: list) -> dict:
    a, m = np.asarray(mean), np.asarray(med)
    n = len(a)
    return {"mean": float(a.mean()), "token_median": float(m.mean()), "frame_p95": float(np.percentile(a, 95)),
            "half1": float(a[: n // 2].mean()), "half2": float(a[n // 2:].mean())}


def gate_cell(vs, k0, views, enc_all, con_all, s: int, thr: float) -> dict:
    """Reference-mode pixel gate at stride s and threshold thr; tiers A and B."""
    cache: list[torch.Tensor] = []
    keep_counts, per = [], {"A": [], "B": []}
    con_out = {"A": {}, "B": {}}
    for i, ref, mask in pixel_gate(views, k0, s, thr):
        if mask is None:  # first sample: dense, not counted
            enc, con, hs = vs.forward(vs.pixels(ref), keep_layers=True)
            cache = hs
            con_out["A"][i] = con_out["B"][i] = con[0]
            continue
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
                "i": i,
                "enc_mean": float(te.mean()), "enc_median": float(te.median()), "enc_p95": float(te.quantile(0.95)),
                "enc_refreshed": float(te[m].mean()) if m.any() else None,
                "enc_unrefreshed": float(te[~m].mean()) if (~m).any() else None,
                "con_mean": float(tc.mean()), "con_median": float(tc.median()), "con_p95": float(tc.quantile(0.95)),
            })
            con_out[tier][i] = c[0]
    stream, series = {}, {}
    for tier in ("A", "B"):
        mean, med = stream_errors(con_out[tier], con_all, s)
        stream[tier] = stream_stats(mean, med)
        series[tier] = {"stream_mean": mean,
                        "samples": {k: [None if x[k] is None else round(x[k], 5) for x in per[tier]]
                                    for k in ("i", "con_mean", "enc_mean", "enc_refreshed", "enc_unrefreshed")}}

    def avg(v, k):
        xs = [x[k] for x in v if x[k] is not None]
        return float(np.mean(xs)) if xs else None
    summ = {t: {k: avg(v, k) for k in v[0] if k != "i"} if v else None for t, v in per.items()}
    return {"keep_counts": keep_counts, "samples": summ, "stream": stream, "series": series}


def token_gate_cell(vs, views, enc_all, con_all, s: int, eps: float) -> dict:
    """Tier C: token-delta gates at every layer (VisionStack.layer_gated), on the true frames at stride s."""
    states = [dict() for _ in vs.layers]
    per, con_out, k_layers = [], {}, []
    for i in range(0, len(views), s):
        h = vs.embed(vs.pixels(views[i]))
        ks = []
        for layer in range(len(vs.layers)):
            h, k1, k2 = vs.layer_gated(layer, h, states[layer], eps)
            ks.append((k1, k2))
        enc, con = vs.head(h)
        con_out[i] = con[0]
        if i == 0:  # first sample: dense, references initialised, not counted
            continue
        k_layers.append(ks)
        te_ = tok_rel(enc[0], enc_all[i])
        tc = tok_rel(con[0], con_all[i])
        per.append({"i": i, "cost": float(np.mean([c_cost(a, b) for a, b in ks])),
                    "enc_mean": float(te_.mean()), "enc_median": float(te_.median()),
                    "con_mean": float(tc.mean()), "con_median": float(tc.median()), "con_p95": float(tc.quantile(0.95))})
    mean, med = stream_errors(con_out, con_all, s)
    kl = np.array(k_layers)  # samples, layers, 2
    return {"cost_fraction": float(np.mean([x["cost"] for x in per])),
            "k1_by_layer": kl[..., 0].mean(axis=0).round(4).tolist(),
            "k2_by_layer": kl[..., 1].mean(axis=0).round(4).tolist(),
            "samples": {k: float(np.mean([x[k] for x in per])) for k in per[0] if k != "i"},
            "stream": stream_stats(mean, med),
            "series": {"stream_mean": mean, "samples": {k: [round(x[k], 5) for x in per]
                                                        for k in ("i", "cost", "con_mean", "enc_mean")}}}


def hold_stream(con_all: torch.Tensor, s: int) -> dict:
    """Dense every s frames, output held in between. s >= frames means the first frame held throughout."""
    outs = {j: con_all[j] for j in range(0, con_all.shape[0], s)}
    mean, med = stream_errors(outs, con_all, s)
    return {**stream_stats(mean, med), "stream_mean": mean}


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


_JW: dict = {}


def _join_init() -> None:
    from transformers import AutoImageProcessor
    ip = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    _JW.update(ip=ip, mean=np.array(ip.image_mean, np.float32), std=np.array(ip.image_std, np.float32),
               k0=K0(threads=1))


def _join_one(job) -> tuple:
    c, era = job
    views = load_views(Path(era) / "Videos" / c["split"] / c["cls"] / c["file"], _JW["ip"], _JW["mean"], _JW["std"])
    counts = {f"{s}/{t:g}": [int(m.sum()) for _, _, m in pixel_gate(views, _JW["k0"], s, t) if m is not None]
              for s in GATE_STRIDES for t in THRESHOLDS}
    return er.clip_key(c), counts


def check_join(args) -> None:
    """CPU only: the pixel gate's keep counts for every sampled clip equal X0's (bench/x0_era.py)."""
    from multiprocessing import Pool
    x0 = {}
    for line in (ROOT / "results" / "x0" / "era_tra" / "raw" / "per_clip.jsonl").read_text().splitlines():
        r = json.loads(line)
        x0[(r["split"], r["cls"], r["id"])] = r["keep_blocks"]
    clips = pick_clips()
    bad = []
    with Pool(8, initializer=_join_init) as pool:
        for key, counts in pool.imap_unordered(_join_one, [(c, args.era) for c in clips]):
            bad += [(key, cell) for cell, v in counts.items() if v != x0[key][cell]]
    print(f"join check: {len(clips)} clips x {len(GATE_STRIDES) * len(THRESHOLDS)} cells, mismatches: {len(bad)}")
    if bad:
        sys.exit(f"keep counts differ from X0: {bad[:5]}")


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
    if args.clips:
        clips = [c for c in clips if c["id"] in args.clips]
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
    vb = VisionStack(str(MODEL), dtype=torch.bfloat16)  # for the per-clip bf16 floor
    k0 = K0()
    rng = np.random.default_rng(SEED)
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
        rec["token_cells"] = {}
        for s in GATE_STRIDES:
            for eps in EPS_C:
                rec["token_cells"][f"{s}/{eps:g}"] = token_gate_cell(vs, views, enc_all, con_all, s, eps)
        for s in HOLD_STRIDES:
            rec["hold"][str(s)] = hold_stream(con_all, s)
        rec["hold"]["first"] = hold_stream(con_all, len(views))
        # floors on this clip's first frame: bf16 vs fp32, and +-1 grey level of noise in fp32
        _, cb, _ = vb.forward(vb.pixels(views[0]))
        noisy = np.clip(views[0].astype(np.int16) + rng.integers(-1, 2, views[0].shape), 0, 255).astype(np.uint8)
        _, cn, _ = vs.forward(vs.pixels(noisy))
        eb, en = tok_rel(cb[0].float(), con_all[0]), tok_rel(cn[0], con_all[0])
        rec["floors"] = {"bf16_vs_fp32": {"mean": float(eb.mean()), "median": float(eb.median())},
                         "noise_pm1": {"mean": float(en.mean()), "median": float(en.median())}}
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
                   for t in ("A", "B") for m in ("enc_mean", "enc_median", "enc_p95", "enc_refreshed",
                                                 "enc_unrefreshed", "con_mean", "con_median", "con_p95")},
                **{f"{t}_stream": pool([c["stream"][t]["mean"] for c in cl]) for t in ("A", "B")},
                **{f"{t}_stream_token_median": pool([c["stream"][t]["token_median"] for c in cl])
                   for t in ("A", "B")},
                **{f"{t}_stream_halves": [pool([c["stream"][t]["half1"] for c in cl])["mean"],
                                          pool([c["stream"][t]["half2"] for c in cl])["mean"]] for t in ("A", "B")},
            }
        tcells = {}
        for key in R[0]["token_cells"]:
            s = int(key.split("/")[0])
            cl = [r["token_cells"][key] for r in R]
            frac = pool([c["cost_fraction"] for c in cl])
            tcells[key] = {
                "cost_fraction": frac, "cost": frac["mean"] * NATIVE_FPS / s,
                "k1_by_layer": np.mean([c["k1_by_layer"] for c in cl], axis=0).round(4).tolist(),
                "k2_by_layer": np.mean([c["k2_by_layer"] for c in cl], axis=0).round(4).tolist(),
                **{f"C_{m}": pool([c["samples"][m] for c in cl]) for m in ("enc_mean", "enc_median", "con_mean",
                                                                        "con_median", "con_p95")},
                "C_stream": pool([c["stream"]["mean"] for c in cl]),
                "C_stream_token_median": pool([c["stream"]["token_median"] for c in cl]),
                "C_stream_halves": [pool([c["stream"]["half1"] for c in cl])["mean"],
                                    pool([c["stream"]["half2"] for c in cl])["mean"]],
            }
        hold = {}
        for s in R[0]["hold"]:
            cost = (float(np.mean([NATIVE_FPS / r["frames"] for r in R])) if s == "first"
                    else NATIVE_FPS / int(s))
            hold[s] = {"cost": cost, "stream": pool([r["hold"][s]["mean"] for r in R]),
                       "stream_token_median": pool([r["hold"][s]["token_median"] for r in R]),
                       "halves": [pool([r["hold"][s]["half1"] for r in R])["mean"],
                                  pool([r["hold"][s]["half2"] for r in R])["mean"]]}
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
        floors = {f: {k: pool([r["floors"][f][k] for r in R]) for k in ("mean", "median")}
                  for f in ("bf16_vs_fp32", "noise_pm1")}
        summ[g] = {"clips": [r["id"] for r in R], "cells": cells, "token_cells": tcells, "hold": hold,
                   "trace": trace, "floors": floors}
    all_floors = {f: {k: pool([r["floors"][f][k] for r in recs]) for k in ("mean", "median")}
                  for f in ("bf16_vs_fp32", "noise_pm1")}
    return {"groups": summ,
            "floors_per_clip": all_floors,
            "floors_check_frame": {"bf16_vs_fp32": checks["bf16_vs_fp32_rel_error"],
                       "uint8_roundtrip": checks["roundtrip_rel_error"],
                       "batch_size": checks["batch8_vs_batch1_rel_error"]},
            "clips_processed": len(recs), "seconds": round(sum(r["seconds"] for r in recs), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", default=str(ROOT / "data" / "era"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tokens" / "era_tra"))
    ap.add_argument("--trace-dir", default=str(ROOT / "traces" / "era_tra"))
    ap.add_argument("--limit", type=int, help="first N clips of the sample not yet done (smoke tests)")
    ap.add_argument("--clips", nargs="+", help="only these clip ids from the sample (smoke tests)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--check-join", action="store_true", help="CPU-only keep-count check against X0, then exit")
    args = ap.parse_args()
    if args.check_join:
        check_join(args)
        return
    if not args.aggregate_only:
        run(args)
    agg = aggregate(Path(args.out))
    summary = {
        "experiment": "token traces and gate error on ERA (training split)",
        "view": "encoder view as in X0 (uint8 512x512), fp32 vision tower + connector, TF32 off",
        "gate": "reference-mode pixel gate as in X0 (keep counts asserted equal to X0's per clip)",
        "tiers": {"A": "stale input: dense encoder on the gate's reference image (a floor)",
                  "B": "stale rows (G1 option 1): pixel mask held through all layers, unrefreshed rows cached",
                  "C": "token-delta gates before q/k/v and before the MLP at every layer, exact attention for all "
                       "rows (G2-style gating with design option 3); true frames, no pixel gate"},
        "tier_c_thresholds": list(EPS_C),
        "tier_c_cost": "estimated FLOP share of the dense encoder layers (cost model in bench/tokens_era.py: "
                       "q/k/v 3, attention 5.33 at about 2 x the gate-1 share, output projection 1 dense, MLP 8, "
                       "in N*d^2 units); its minimum is 1/17.3 = 0.058 because the output projection stays dense",
        "error": "per-token relative error ||x - x_dense|| / ||x_dense||, mean over tokens then samples then clips",
        "stream": "connector output at 24 fps, each method holding its latest output; mean over frames of the "
                  "token-mean error (token_median: of the token-median error); hold 'first' = first frame held",
        "cost_note": "tier A is plotted at the pixel mask's cost, which no kernel achieves (it recomputes every "
                     "token on the reference image)",
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

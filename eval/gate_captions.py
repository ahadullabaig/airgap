"""Caption-level gating check (Part B decision gate, PLAN.md section 3.3): do the connector-output errors
measured in the token traces (bench/tokens_era.py) change the captions, and does any gate beat dense at a
lower frame rate with the output held, at equal cost, once captions are what is scored?

Clips: the first 100 of E1's class-stratified order on the CapERA test split (eval/e1.py, 4 per class).
Frames: E1's four frames per clip (demo/sampling.py rule, the clip's whole length as the window), located in
the full 24 fps stream, so gates and held outputs evolve over the real frame sequence up to each of them.
Only frames up to the fourth sampled one are processed (nothing after it can change the caption).
Language model: E1's frozen configuration (eval/e1_config.json: prompt P1, bf16, greedy, 48 new tokens).
Each condition replaces only the 4 x 64 connector outputs, through a patched SmolVLMModel.get_image_features.

Conditions. Vision tower and connector in fp32 on the encoder's uint8 view, TF32 off, as in the token traces,
unless noted; cost in full-frame equivalents per second of video, per clip:
  dense        every frame (cost 24); the reference
  e1_bf16      E1's own path, unpatched: vision tower in bf16 on the processor's pixels (a floor)
  noise_pm1    dense on the four frames with +-1 grey level of uniform noise (a floor)
  hold_S       dense every S frames from the clip's first, the last output held; S = 2, 4, 8, 24 (12, 6, 3,
               1 fps). At a sampled frame j the held output is j mod S frames old (0 when j is a multiple of
               S); the recorded connector error shows where that sits against the stream average.
  A_T, B_T     reference-mode pixel gate at 24 fps, block-MAD threshold T = 2, 8 (tokens_era.gate_cell):
               A stale input (a floor no kernel reaches), B stale rows (G1 option 1); cost = keep-rate x 24
  C_E          token-delta gates with exact attention at 24 fps, E = 0.05, 0.2 (tokens_era.token_gate_cell);
               cost = estimated FLOP share x 24
Thresholds were picked from the training-split traces to span the cost range, before any caption here.

Metrics, fixed before the run:
  primary    1. Per clip, CIDEr-D against the CapERA references minus dense's; mean over clips with a 95%
                bootstrap interval (clips resampled).
             2. At equal cost: per clip, a gate's difference from 1. minus hold's, hold taken on the chord
                between the hold rates at the gate's cost (as equal_cost in the token traces); mean and
                interval. Clips whose gate cost is below 1 fps are left out (n).
  secondary  Identical-caption rate and CIDEr-D against dense's caption (sensitivity only: a held output is an
             exact encoding of an earlier frame, so differing from dense is not an error in itself);
             connector error at the four frames, to compare gates and hold at matched error.
Outputs hold ids, our captions and scores, never CapERA's reference text.

Steps:
  check   validates the injection and the frame mapping on a few clips (results/gate_captions/check.json)
  run     captions every condition, resumable (results/gate_captions/predictions.jsonl)
  score   results/gate_captions/scores.json

Example:
    .venv/bin/python -I eval/gate_captions.py check
    .venv/bin/python -I eval/gate_captions.py run [--limit 2] [--out DIR]
    .venv/bin/python -I eval/gate_captions.py score
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("eval", "bench", "demo"):
    sys.path.insert(0, str(ROOT / sub))
import capera  # noqa: E402
import e1  # noqa: E402

OUT = ROOT / "results" / "gate_captions"
CLIPS = e1.SELECT_CLIPS
FPS = 24.0
HOLD_STRIDES = (2, 4, 8, 24)
PIXEL_THR = (2.0, 8.0)
TOKEN_EPS = (0.05, 0.2)
BOOT, BOOT_SEED = 10_000, 0
ERR_BINS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 10.0)  # connector-error bins for the matched-error table


def conditions() -> list[str]:
    return (["dense", "e1_bf16", "noise_pm1"] + [f"hold_{s}" for s in HOLD_STRIDES]
            + [f"{t}_{x:g}" for x in PIXEL_THR for t in ("A", "B")] + [f"C_{e:g}" for e in TOKEN_EPS])


def family(name: str) -> str:
    return name.split("_")[0]


def sample_indices(times: list, duration_s: float, k: int) -> list[int]:
    """demo/sampling.sample_window's rule from the start of the clip, as indices into the decoded stream:
    for each target (i + 0.5) * window / k, the first frame with a timestamp at or after it."""
    targets = [(i + 0.5) * duration_s / k for i in range(k)]
    out = []
    for j, t in enumerate(times):
        if t is None:
            continue
        if t >= targets[len(out)]:
            out.append(j)
            if len(out) == k:
                break
    if len(out) < k:
        raise ValueError(f"{len(out)} of {k} frames")
    return out


class Models:
    """The E1 captioner (bf16, its own vision tower) with a switchable feature injection, and the fp32 vision
    stack of the token traces."""

    def __init__(self):
        import numpy as np
        import torch  # noqa: F401
        from transformers import AutoImageProcessor
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        import tokens_era
        from k0lib import K0
        from visionstack import VisionStack
        from vlm import MODEL, Captioner
        self.te = tokens_era
        self.cap = Captioner()
        self.vs = VisionStack(str(MODEL))
        self.k0 = K0()
        self.ip = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
        self.mean = np.array(self.ip.image_mean, np.float32)
        self.std = np.array(self.ip.image_std, np.float32)
        self.prompt = json.loads(e1.CONFIG.read_text())["prompt"]
        inner = self.cap.model.model  # SmolVLMModel: its forward calls self.get_image_features
        self.own_features = inner.get_image_features  # the bound original
        self.feats, self.calls = None, 0

        def patched(pixel_values, pixel_attention_mask=None, **kw):
            self.calls += 1
            if self.feats is None:
                return self.own_features(pixel_values, pixel_attention_mask, **kw)
            return BaseModelOutputWithPooling(pooler_output=[self.feats])

        inner.get_image_features = patched

    def caption(self, inp, feats=None) -> dict:
        """Greedy caption with E1's settings; feats (4 * 64, 960) replaces the image features, None keeps the
        model's own. Asserts that the image features were requested exactly once."""
        self.feats, self.calls = feats, 0
        try:
            g = self.cap.generate(inp, e1.MAX_NEW_TOKENS)
        finally:
            self.feats = None
        assert self.calls == 1, f"get_image_features called {self.calls} times"
        return g

    def bf16_features(self, inp):
        import torch
        kw = {k: inp[k] for k in ("pixel_attention_mask",) if k in inp}
        with torch.inference_mode():
            return self.own_features(inp["pixel_values"], return_dict=True, **kw).pooler_output[0].clone()

    def decode(self, path: Path):
        """All frames (PIL) and their timestamps (None without a pts), in decode order."""
        import av
        imgs, times = [], []
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            for f in c.decode(st):
                imgs.append(f.to_image())
                times.append(None if f.pts is None else float(f.pts * st.time_base))
        return imgs, times

    def view(self, img):
        from x0core import model_view
        return model_view(self.ip, img, self.mean, self.std)


def clip_record(M: Models, c, e1_caption: str | None) -> dict:
    import numpy as np
    import torch
    te = M.te
    t0 = time.perf_counter()
    imgs, times = M.decode(c.path)
    idx = sample_indices(times, c.duration_s, e1.FRAMES)
    images = [imgs[j] for j in idx]
    n = idx[-1] + 1
    views = np.stack([M.view(im) for im in imgs[:n]])
    del imgs
    inp = M.cap.inputs(images, M.prompt, e1.SPLIT_IMAGES)
    vs = M.vs
    with torch.no_grad():
        enc_all = torch.empty((n, te.TOKENS, 768), device=vs.device)
        con_all = torch.empty((n, 64, 960), device=vs.device)
        for i in range(n):
            e, cc, _ = vs.forward(vs.pixels(views[i]))
            enc_all[i], con_all[i] = e[0], cc[0]
        feats, cost = {}, {}
        feats["dense"], cost["dense"] = [con_all[j] for j in idx], FPS
        for s in HOLD_STRIDES:
            feats[f"hold_{s}"], cost[f"hold_{s}"] = [con_all[(j // s) * s] for j in idx], FPS / s
        rng = np.random.default_rng([e1.SEED, zlib.crc32(c.id.encode())])
        feats["noise_pm1"], cost["noise_pm1"] = [], None
        for j in idx:
            noisy = np.clip(views[j].astype(np.int16) + rng.integers(-1, 2, views[j].shape), 0, 255).astype(np.uint8)
            feats["noise_pm1"].append(vs.forward(vs.pixels(noisy))[1][0])
        for thr in PIXEL_THR:
            out = {}
            cell = te.gate_cell(vs, M.k0, views, enc_all, con_all, 1, thr, outputs=out)
            keep = float(np.mean(cell["keep_counts"])) / te.TOKENS
            for t in ("A", "B"):
                feats[f"{t}_{thr:g}"], cost[f"{t}_{thr:g}"] = [out[t][j] for j in idx], keep * FPS
        for eps in TOKEN_EPS:
            out = {}
            cell = te.token_gate_cell(vs, views, enc_all, con_all, 1, eps, outputs=out)
            feats[f"C_{eps:g}"], cost[f"C_{eps:g}"] = [out[j] for j in idx], cell["cost_fraction"] * FPS
        own = M.bf16_features(inp).float().reshape(e1.FRAMES, 64, 960)
        feats["e1_bf16"], cost["e1_bf16"] = list(own), None
        err = {k: [float(te.tok_rel(f.float(), con_all[j]).mean()) for f, j in zip(v, idx)] for k, v in feats.items()}
    t1 = time.perf_counter()
    conds = {}
    for name in conditions():
        g = M.caption(inp, None if name == "e1_bf16" else torch.cat(feats[name]).reshape(-1, 960))
        conds[name] = {"caption_raw": g["caption"], "new_tokens": g["new_tokens"], "truncated": g["truncated"],
                       "cost": None if cost[name] is None else round(cost[name], 4),
                       "con_err": round(float(np.mean(err[name])), 5), "con_err_frames": [round(x, 5) for x in err[name]]}
    del enc_all, con_all
    torch.cuda.empty_cache()
    return {"split": c.split, "id": c.id, "cls": c.cls, "frames_total": len(times), "frames_used": n,
            "sample_idx": idx, "e1_match": None if e1_caption is None else conds["e1_bf16"]["caption_raw"] == e1_caption,
            "conditions": conds, "encode_s": round(t1 - t0, 1), "seconds": round(time.perf_counter() - t0, 1)}


def e1_captions() -> dict:
    path = ROOT / "results" / "e1" / "test_predictions.jsonl"
    return {r["id"]: r["caption_raw"] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def clips() -> list:
    return capera.stratified_order(capera.load("test"), e1.SEED)[:CLIPS]


# ---------------------------------------------------------------- check
def check(n: int, out: Path) -> None:
    import numpy as np
    import torch
    from sampling import sample_window
    M = Models()
    te = M.te
    E1 = e1_captions()
    res = {"clips": []}
    prev = None
    for c in clips()[:n]:
        imgs, times = M.decode(c.path)
        idx = sample_indices(times, c.duration_s, e1.FRAMES)
        ref = [img for _, img in sample_window(str(c.path), 0.0, c.duration_s, e1.FRAMES)]
        frames_equal = all(np.array_equal(M.view(a), M.view(imgs[j])) for a, j in zip(ref, idx))
        inp = M.cap.inputs(ref, M.prompt, e1.SPLIT_IMAGES)
        own = M.bf16_features(inp)
        cap_own = M.caption(inp)["caption"]
        cap_injected = M.caption(inp, own)["caption"]
        with torch.no_grad():
            dense = torch.stack([M.vs.forward(M.vs.pixels(M.view(imgs[j])))[1][0] for j in idx])
        own4 = own.float().reshape(e1.FRAMES, 64, 960)
        err_in_order = [float(te.tok_rel(own4[k], dense[k]).mean()) for k in range(e1.FRAMES)]
        err_reversed = [float(te.tok_rel(own4[k], dense[e1.FRAMES - 1 - k]).mean()) for k in range(e1.FRAMES)]
        cap_fp32 = M.caption(inp, dense.reshape(-1, 960))["caption"]
        row = {"id": c.id, "sample_idx": idx, "frames": len(times), "frames_equal_sample_window": frames_equal,
               "e1_stored_match": E1.get(c.id) == cap_own if c.id in E1 else None,
               "injected_own_features_match": cap_injected == cap_own,
               "fp32_vs_bf16_feature_error_in_order": [round(x, 5) for x in err_in_order],
               "fp32_vs_bf16_feature_error_frames_reversed": [round(x, 5) for x in err_reversed],
               "fp32_caption_matches_bf16": cap_fp32 == cap_own}
        if prev is not None:  # the previous clip's features in this clip's prompt give the previous clip's caption
            p_inp, p_own, p_cap = prev
            row["same_prompt_tokens_as_previous"] = bool(torch.equal(inp["input_ids"], p_inp["input_ids"]))
            row["previous_features_give_previous_caption"] = M.caption(inp, p_own)["caption"] == p_cap
            row["features_change_caption"] = M.caption(inp, p_own)["caption"] != cap_own
        prev = (inp, own, cap_own)
        res["clips"].append(row)
        print(json.dumps(row), flush=True)
    rows = res["clips"]
    res["passed"] = bool(all(r["frames_equal_sample_window"] and r["injected_own_features_match"]
                             and r["e1_stored_match"] is not False for r in rows)
                         and all(r.get("previous_features_give_previous_caption", True) for r in rows)
                         and all(max(r["fp32_vs_bf16_feature_error_in_order"]) < 0.05 for r in rows))
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(res, indent=1) + "\n")
    print("passed" if res["passed"] else "FAILED", out / "check.json")


# ---------------------------------------------------------------- run
def run(limit: int | None, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "predictions.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [c for c in clips() if c.id not in done]
    if limit:
        todo = todo[:limit]
    M = Models()
    E1 = e1_captions()
    t_start = time.perf_counter()
    with path.open("a") as fh:
        for i, c in enumerate(todo):
            rec = clip_record(M, c, E1.get(c.id))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            el = time.perf_counter() - t_start
            print(f"{i + 1}/{len(todo)} {c.id:28s} {rec['seconds']:.0f} s (encode {rec['encode_s']:.0f} s)  "
                  f"elapsed {el / 60:.0f} min, about {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left",
                  flush=True)


# ---------------------------------------------------------------- score
def chord(points: list, x: float) -> float | None:
    pts = sorted(points)
    for (c0, e0), (c1, e1_) in zip(pts, pts[1:]):
        if c0 <= x <= c1 and c1 > c0:
            return e0 + (x - c0) / (c1 - c0) * (e1_ - e0)
    return None


def interval(a, rng) -> dict:
    import numpy as np
    a = np.asarray(a, dtype=float)
    if not len(a):
        return None
    boots = a[rng.integers(0, len(a), (BOOT, len(a)))].mean(axis=1)
    return {"n": int(len(a)), "mean": float(a.mean()), "lo95": float(np.percentile(boots, 2.5)),
            "hi95": float(np.percentile(boots, 97.5))}


def score(out: Path) -> None:
    import numpy as np
    recs = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
    refs_all = {c.id: c.refs for c in capera.load("test")}
    ids = [r["id"] for r in recs]
    refs = {i: refs_all[i] for i in ids}
    names = conditions()
    cap = {k: {r["id"]: e1.sanitise(r["conditions"][k]["caption_raw"]) for r in recs} for k in names}
    corpus, per = {}, {}
    for k in names:
        corpus[k], per[k] = e1.coco_scores(refs, cap[k])
    vs_dense = {k: e1.coco_scores({i: [cap["dense"][i]] for i in ids}, cap[k])[0]["CIDEr"] for k in names}
    cider = {k: np.array([per[k][i]["CIDEr"] for i in ids]) for k in names}
    delta = {k: cider[k] - cider["dense"] for k in names}
    costs = {k: np.array([r["conditions"][k]["cost"] if r["conditions"][k]["cost"] is not None else np.nan
                          for r in recs]) for k in names}
    errs = {k: np.array([r["conditions"][k]["con_err"] for r in recs]) for k in names}
    same = {k: np.array([cap[k][i] == cap["dense"][i] for i in ids]) for k in names}
    rng = np.random.default_rng(BOOT_SEED)
    table = {}
    for k in names:
        table[k] = {"cost_mean": None if np.isnan(costs[k]).all() else float(np.nanmean(costs[k])),
                    "con_err_mean": float(errs[k].mean()), "corpus_vs_refs": corpus[k],
                    "cider_minus_dense": interval(delta[k], rng),
                    "identical_to_dense": float(same[k].mean()), "cider_vs_dense_caption": vs_dense[k],
                    "mean_words": float(np.mean([len(cap[k][i].split()) for i in ids])),
                    "truncated": int(sum(r["conditions"][k]["truncated"] for r in recs))}
    # equal cost: each gate against hold on the chord at the gate's per-clip cost
    eq = {}
    hold_pts = [("dense", FPS)] + [(f"hold_{s}", FPS / s) for s in HOLD_STRIDES]
    for k in names:
        if family(k) not in ("A", "B", "C"):
            continue
        d_cider, d_same, d_err = [], [], []
        for j in range(len(ids)):
            c = costs[k][j]
            h = chord([(cst, delta[h_][j]) for h_, cst in hold_pts], c)
            if h is None:
                continue
            d_cider.append(delta[k][j] - h)
            d_same.append(float(same[k][j]) - chord([(cst, float(same[h_][j])) for h_, cst in hold_pts], c))
            d_err.append(errs[k][j] - chord([(cst, errs[h_][j]) for h_, cst in hold_pts], c))
        eq[k] = {"cider_minus_dense_vs_hold": interval(d_cider, rng),
                 "identical_rate_vs_hold": interval(d_same, rng), "con_err_vs_hold": interval(d_err, rng)}
    # matched connector error: caption change by error bin, per family
    matched = {}
    for fam in ("hold", "A", "B", "C"):
        ks = [k for k in names if family(k) == fam]
        e = np.concatenate([errs[k] for k in ks])
        d = np.concatenate([delta[k] for k in ks])
        s = np.concatenate([same[k] for k in ks])
        rows = []
        for lo, hi in zip(ERR_BINS, ERR_BINS[1:]):
            m = (e >= lo) & (e < hi)
            if m.any():
                rows.append({"con_err": [lo, hi], "n": int(m.sum()), "cider_minus_dense": float(d[m].mean()),
                             "identical_to_dense": float(s[m].mean())})
        matched[fam] = rows
    res = {"experiment": "caption-level gating check (eval/gate_captions.py docstring: conditions, metrics fixed "
                         "before the run)",
           "clips": len(ids), "e1_match": int(sum(bool(r["e1_match"]) for r in recs)),
           "bootstrap": {"resamples": BOOT, "seed": BOOT_SEED, "interval": "95% percentile, clips resampled"},
           "cost": "full-frame equivalents per second of video; tier A's is unachievable, tier C's a FLOP share",
           "conditions": table, "equal_cost": eq, "matched_error": matched,
           "seconds": round(sum(r["seconds"] for r in recs), 1)}
    (out / "scores.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'condition':10s} {'cost':>6s} {'conerr':>7s} {'CIDEr':>6s} {'d vs dense [95%]':>22s} {'same':>5s}")
    for k in names:
        t = table[k]
        ci = t["cider_minus_dense"]
        print(f"{k:10s} {'' if t['cost_mean'] is None else f'{t['cost_mean']:6.2f}':>6s} {100 * t['con_err_mean']:6.1f}% "
              f"{t['corpus_vs_refs']['CIDEr']:6.3f} {ci['mean']:+7.3f} [{ci['lo95']:+.3f}, {ci['hi95']:+.3f}] "
              f"{t['identical_to_dense']:5.2f}")
    for k, v in eq.items():
        ci = v["cider_minus_dense_vs_hold"]
        if ci:
            print(f"equal cost {k:8s} n={ci['n']:3d} CIDEr vs hold {ci['mean']:+.3f} [{ci['lo95']:+.3f}, {ci['hi95']:+.3f}]")
    print("wrote", out / "scores.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("check", "run", "score"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--clips", type=int, default=3, help="check: number of clips")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.step == "check":
        check(a.clips, Path(a.out))
    elif a.step == "run":
        run(a.limit, Path(a.out))
    else:
        score(Path(a.out))


if __name__ == "__main__":
    main()

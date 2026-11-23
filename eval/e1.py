"""E1: caption quality on CapERA (PLAN.md section 2.5). Zero-shot as far as the model card shows: ERA and
CapERA are not among its named training datasets (whether they are inside its aggregate collections is
unknown).

Steps, in order:
  precheck   CPU only: 4 frames come back from every clip of a split (each clip's own length as the window).
  select     choose the prompt on the TRAIN split: the candidates and the metric below were fixed before any
             caption was generated; 100 train clips (4 per class), CIDEr. Writes eval/e1_config.json, which
             freezes the configuration for the test run.
  run        caption every test clip with the frozen configuration; resumable (results/e1/test_predictions.jsonl,
             our captions and timings only, committed so the scores can be regenerated).
  score      COCO metrics (BLEU-1..4, METEOR, ROUGE-L, CIDEr) on the whole test split, per-clip scores for E4,
             output statistics, latency, and three controls:
               shuffled   predictions scored against another clip's references (scores should collapse)
               constant   one fixed caption for every clip (a floor)
               human      reference 1 scored against references 2-5 (an anchor; the references are one
                          human caption plus paraphrases of it, which inflates this)

Committed outputs hold ids, our captions and scores, never CapERA's reference text.

Example:
    .venv/bin/python -I eval/e1.py precheck --split test
    .venv/bin/python -I eval/e1.py select
    .venv/bin/python -I eval/e1.py run [--limit 50]
    .venv/bin/python -I eval/e1.py memory
    .venv/bin/python -I eval/e1.py score
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import capera  # noqa: E402

OUT = ROOT / "results" / "e1"
CONFIG = ROOT / "eval" / "e1_config.json"
FRAMES, SPLIT_IMAGES, MAX_NEW_TOKENS = 4, False, 48
SEED = 0
SELECT_CLIPS = 100
SELECT_METRIC = "CIDEr"
CANDIDATE_PROMPTS = [
    # fixed before any caption was generated; P1 is the demo's default
    "These frames are from an aerial drone video, in time order. Describe in one or two sentences what is "
    "happening on the ground.",
    "These frames are from an aerial drone video, in time order. Describe the scene in one short sentence.",
    "Write one short sentence describing what this aerial video shows.",
    "Describe this aerial drone video in one short sentence, mentioning the main event and the setting.",
]
CONSTANT_CAPTION = "an aerial view of buildings, roads and trees"  # the floor control, fixed in advance


def sanitise(text: str) -> str:
    """The only post-editing: newlines and runs of whitespace become single spaces; ends stripped."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- precheck
def precheck(split: str) -> None:
    from vlm import Captioner
    clips = capera.load(split)
    bad = []
    t0 = time.perf_counter()
    for i, c in enumerate(clips):
        try:
            n = len(Captioner.frames(c.path, c.duration_s, FRAMES))
        except Exception as e:  # noqa: BLE001
            n = 0
            bad.append({"id": c.id, "error": repr(e)[:200]})
        if n != FRAMES and not any(b["id"] == c.id for b in bad):
            bad.append({"id": c.id, "frames": n})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(clips)} {time.perf_counter() - t0:.0f}s", flush=True)
    res = {"split": split, "clips": len(clips), "frames_per_clip": FRAMES, "failures": bad,
           "duration_s": {"min": min(c.duration_s for c in clips), "max": max(c.duration_s for c in clips)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"precheck_{split}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}), "failures:", len(bad))


# ---------------------------------------------------------------- captioning
def caption_clips(cap, clips, prompt: str, out_path: Path, label: str) -> None:
    """Captions clips not already in out_path (jsonl), appending one record per clip."""
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()}
    todo = [c for c in clips if c.id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    with out_path.open("a") as fh:
        for i, c in enumerate(todo):
            t0 = time.perf_counter()
            images = cap.frames(c.path, c.duration_s, FRAMES)
            t1 = time.perf_counter()
            inp = cap.inputs(images, prompt, SPLIT_IMAGES)
            t2 = time.perf_counter()
            g = cap.generate(inp, MAX_NEW_TOKENS)
            rec = {"split": c.split, "id": c.id, "cls": c.cls, "caption_raw": g["caption"],
                   "new_tokens": g["new_tokens"], "prompt_tokens": g["prompt_tokens"], "truncated": g["truncated"],
                   "decode_s": round(t1 - t0, 4), "preprocess_s": round(t2 - t1, 4),
                   "generate_s": round(g["generate_s"], 4), "first_call": not done and i == 0}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                el = time.perf_counter() - t_start
                print(f"{label} {i + 1}/{len(todo)}  {el:.0f}s  ({el / (i + 1):.2f} s/clip)", flush=True)


def read_preds(path: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


# ---------------------------------------------------------------- scoring
def coco_scores(refs: dict, preds: dict, cider_sigma: float = 6.0) -> tuple[dict, dict]:
    """refs: id -> [str]; preds: id -> str. Returns (corpus scores, per-clip scores). CIDEr is CIDEr-D, whose
    Gaussian length penalty has width cider_sigma words (6.0 is the standard; a huge value turns it off)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tok = PTBTokenizer()
    ids = sorted(preds)
    g = tok.tokenize({i: [{"caption": r} for r in refs[i]] for i in ids})
    r = tok.tokenize({i: [{"caption": preds[i]}] for i in ids})
    corpus, per = {}, {i: {} for i in ids}
    for name, scorer in (("BLEU", Bleu(4)), ("METEOR", Meteor()), ("ROUGE_L", Rouge()),
                         ("CIDEr", Cider(sigma=cider_sigma))):
        score, scores = scorer.compute_score(g, r)
        if name == "BLEU":
            for n in range(4):
                corpus[f"BLEU-{n + 1}"] = float(score[n])
            for i, v in zip(sorted(g), scores[3]):
                per[i]["BLEU-4"] = float(v)
        else:
            corpus[name] = float(score)
            for i, v in zip(sorted(g), scores):
                per[i][name] = float(v)
    return corpus, per


# ---------------------------------------------------------------- commands
def select() -> None:
    from vlm import Captioner
    if CONFIG.exists():
        sys.exit(f"{CONFIG} exists: the configuration is frozen")
    clips = capera.stratified_order(capera.load("train"), SEED)[:SELECT_CLIPS]
    cap = Captioner()
    results = []
    for k, prompt in enumerate(CANDIDATE_PROMPTS):
        path = OUT / f"select_train_p{k + 1}.jsonl"
        caption_clips(cap, clips, prompt, path, f"P{k + 1}")
        preds = {i: sanitise(r["caption_raw"]) for i, r in read_preds(path).items()}
        refs = {c.id: c.refs for c in clips}
        corpus, _ = coco_scores(refs, preds)
        lens = [len(p.split()) for p in preds.values()]
        results.append({"prompt": prompt, "id": f"P{k + 1}", "scores": corpus, "mean_words": statistics.mean(lens)})
        print(f"P{k + 1}: {SELECT_METRIC} {corpus[SELECT_METRIC]:.3f}  mean words {statistics.mean(lens):.1f}")
    best = max(results, key=lambda r: r["scores"][SELECT_METRIC])
    config = {"frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prompt": best["prompt"], "prompt_id": best["id"],
              "frames": FRAMES, "image_splitting": SPLIT_IMAGES, "max_new_tokens": MAX_NEW_TOKENS, "dtype": "bfloat16",
              "decoding": "greedy", "window": "each clip's whole length",
              "selection": {"split": "train", "clips": len(clips), "seed": SEED, "metric": SELECT_METRIC,
                            "ids": [c.id for c in clips], "candidates": results}}
    CONFIG.write_text(json.dumps(config, indent=1) + "\n")
    print(f"chose {best['id']}; wrote {CONFIG}")


def run(limit: int | None) -> None:
    from vlm import Captioner
    config = json.loads(CONFIG.read_text())
    clips = capera.stratified_order(capera.load("test"), SEED)
    if limit:
        clips = clips[:limit]
    cap = Captioner()
    caption_clips(cap, clips, config["prompt"], OUT / "test_predictions.jsonl", "test")


def memory(n: int = 20) -> None:
    """Peak GPU memory for captioning with the frozen configuration, on the first n test clips (not saved as
    predictions): PyTorch's peak allocation and reservation, and device memory used as the driver sees it."""
    import torch
    from vlm import Captioner
    config = json.loads(CONFIG.read_text())
    clips = capera.stratified_order(capera.load("test"), SEED)[:n]
    cap = Captioner()
    torch.cuda.reset_peak_memory_stats()
    for c in clips:
        cap.generate(cap.inputs(cap.frames(c.path, c.duration_s, FRAMES), config["prompt"], SPLIT_IMAGES),
                     MAX_NEW_TOKENS)
    free, total = torch.cuda.mem_get_info()
    res = {"clips": n, "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
           "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
           "device_used_mib": (total - free) / 2**20, "device_total_mib": total / 2**20,
           "note": "device_used includes the CUDA context and other processes (the desktop)"}
    (OUT / "memory_test.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def score() -> None:
    import torch
    config = json.loads(CONFIG.read_text())
    clips = {c.id: c for c in capera.load("test")}
    raw = read_preds(OUT / "test_predictions.jsonl")
    missing = sorted(set(clips) - set(raw))
    preds = {i: sanitise(r["caption_raw"]) for i, r in raw.items()}
    refs = {i: clips[i].refs for i in preds}
    corpus, per = coco_scores(refs, preds)

    rng = random.Random(SEED)
    ids = sorted(preds)
    perm = ids[:]
    rng.shuffle(perm)
    shuffled, _ = coco_scores({i: refs[j] for i, j in zip(ids, perm)}, preds)
    constant, _ = coco_scores(refs, {i: CONSTANT_CAPTION for i in ids})
    human, _ = coco_scores({i: refs[i][1:] for i in ids}, {i: refs[i][0] for i in ids})
    words = [len(p.split()) for p in preds.values()]
    ref_words = [len(r.split()) for i in ids for r in refs[i]]
    warm = [r for r in raw.values() if not r["first_call"]]
    gen = sorted(r["generate_s"] for r in warm)
    total = sorted(r["decode_s"] + r["preprocess_s"] + r["generate_s"] for r in warm)
    q = lambda v, p: v[min(len(v) - 1, round(p * (len(v) - 1)))]  # noqa: E731
    by_cls = {}
    for i in ids:
        by_cls.setdefault(clips[i].cls, []).append(per[i]["CIDEr"])
    summary = {
        "experiment": "E1 caption quality, CapERA test split, zero-shot (as far as the model card shows)",
        "config": {k: v for k, v in config.items() if k != "selection"}, "selection": {
            "split": "train", "clips": config["selection"]["clips"], "metric": config["selection"]["metric"],
            "candidates": [{k: c[k] for k in ("id", "prompt", "mean_words")} | {"CIDEr": c["scores"]["CIDEr"]}
                           for c in config["selection"]["candidates"]]},
        "clips_scored": len(ids), "clips_missing": missing,
        "references": "5 per clip: one human caption and four paraphrases of it, not five independent references",
        "scores": corpus,
        "controls": {"shuffled_references": shuffled, "constant_caption": {"caption": CONSTANT_CAPTION, **constant},
                     "human_vs_other_references": human},
        "outputs": {"mean_words": statistics.mean(words), "median_words": statistics.median(words),
                    "reference_mean_words": statistics.mean(ref_words),
                    "empty": sum(not p for p in preds.values()),
                    "truncated_at_max_new_tokens": sum(r["truncated"] for r in raw.values()),
                    "multi_sentence": sum(len(re.findall(r"[.!?](\s|$)", p)) > 1 for p in preds.values()),
                    "contained_newline": sum("\n" in r["caption_raw"] for r in raw.values())},
        "latency_s": {"generate_median": q(gen, 0.5), "generate_p90": q(gen, 0.9),
                      "per_clip_total_median": q(total, 0.5), "per_clip_total_p90": q(total, 0.9),
                      "note": "first call excluded; decode and preprocess on the CPU, generation on the GPU"},
        "cider_by_class": {k: statistics.mean(v) for k, v in sorted(by_cls.items())},
        "memory": (json.loads((OUT / "memory_test.json").read_text()) if (OUT / "memory_test.json").exists()
                   else "not measured: run `eval/e1.py memory`"),
        "env": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__},
    }
    (OUT / "scores_test.json").write_text(json.dumps(summary, indent=1) + "\n")
    (OUT / "per_clip_test.json").write_text(json.dumps(
        [{"id": i, "cls": clips[i].cls, "caption": preds[i], **per[i], "new_tokens": raw[i]["new_tokens"],
          "truncated": raw[i]["truncated"]} for i in ids], indent=0) + "\n")
    print(json.dumps({k: summary[k] for k in ("scores", "outputs", "latency_s")}, indent=1))
    print("controls:", json.dumps({k: {m: round(v, 3) for m, v in c.items() if isinstance(v, float)}
                                   for k, c in summary["controls"].items()}))


def diagnose() -> None:
    """How much of CIDEr is caption length? No GPU. Writes results/e1/diagnostics_test.json.
    - per-clip CIDEr against caption length and against the length gap to the references
    - per class: mean words, METEOR, CIDEr, and their ranks
    - CIDEr with the length penalty off (diagnostic only; the headline keeps CIDEr-D)
    - the first 100 clips of the test stratified order scored alone (4 per class, like the train selection)"""
    clips = {c.id: c for c in capera.load("test")}
    raw = read_preds(OUT / "test_predictions.jsonl")
    preds = {i: sanitise(r["caption_raw"]) for i, r in raw.items()}
    refs = {i: clips[i].refs for i in preds}
    corpus, per = coco_scores(refs, preds)
    no_pen, per_np = coco_scores(refs, preds, cider_sigma=1e9)
    ids = sorted(preds)
    words = {i: len(preds[i].split()) for i in ids}
    gap = {i: words[i] - statistics.mean(len(r.split()) for r in refs[i]) for i in ids}

    def corr(xs, ys):
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        return cov / ((sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5)

    cider = [per[i]["CIDEr"] for i in ids]
    by_cls = {}
    for i in ids:
        d = by_cls.setdefault(clips[i].cls, {"words": [], "METEOR": [], "CIDEr": [], "CIDEr_no_length_penalty": []})
        d["words"].append(words[i])
        d["METEOR"].append(per[i]["METEOR"])
        d["CIDEr"].append(per[i]["CIDEr"])
        d["CIDEr_no_length_penalty"].append(per_np[i]["CIDEr"])
    table = {k: {m: statistics.mean(v[m]) for m in v} for k, v in by_cls.items()}
    for m in ("METEOR", "CIDEr", "CIDEr_no_length_penalty"):
        for r, k in enumerate(sorted(table, key=lambda k: -table[k][m])):
            table[k][f"rank_{m}"] = r + 1
    first100 = [c.id for c in capera.stratified_order(capera.load("test"), SEED)[:SELECT_CLIPS]]
    sub, _ = coco_scores({i: refs[i] for i in first100}, {i: preds[i] for i in first100})
    res = {"cider_is": "CIDEr-D (pycocoevalcap): Gaussian length penalty, sigma 6 words",
           "corr_cider_vs_words": corr([words[i] for i in ids], cider),
           "corr_cider_vs_abs_length_gap": corr([abs(gap[i]) for i in ids], cider),
           "mean_length_gap_words": statistics.mean(gap.values()),
           "corpus": corpus, "corpus_cider_no_length_penalty": no_pen["CIDEr"],
           "first_100_stratified_test_clips": sub, "by_class": dict(sorted(table.items()))}
    (OUT / "diagnostics_test.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "by_class"}, indent=1))
    print(f"{'class':18s} {'words':>6s} {'METEOR':>7s} {'r':>3s} {'CIDEr':>6s} {'r':>3s} {'CIDEr-np':>8s} {'r':>3s}")
    for k, v in sorted(table.items(), key=lambda kv: kv[1]["rank_METEOR"]):
        print(f"{k:18s} {v['words']:6.1f} {v['METEOR']:7.3f} {v['rank_METEOR']:3d} {v['CIDEr']:6.3f} {v['rank_CIDEr']:3d} "
              f"{v['CIDEr_no_length_penalty']:8.3f} {v['rank_CIDEr_no_length_penalty']:3d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["precheck", "select", "run", "memory", "score", "diagnose"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    {"precheck": lambda: precheck(args.split), "select": select, "run": lambda: run(args.limit),
     "memory": memory, "score": score, "diagnose": diagnose}[args.command]()


if __name__ == "__main__":
    main()

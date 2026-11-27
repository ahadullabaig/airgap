"""E2: event recognition on CapERA (PLAN.md section 2.5). Zero-shot: the model is given the 25 class names and
asked to pick one per clip; nothing is trained.

Everything below was fixed before any E2 answer was generated.

Settings: E1's frozen captioning settings (eval/e1_config.json) except the prompt: 4 frames spread over each
clip's whole length, no image splitting, bf16, greedy decoding, at most 48 new tokens. E2 runs before the demo
is frozen (PLAN.md 2.6 runs it on the frozen demo); if the demo changes the model or these settings, rerun E2.

Class names shown to the model (ERA's class order, "no event" last):
  ERA's names split into words, except Constructing -> "construction", PostEarthquake -> "earthquake damage",
  ParadeProtest -> "parade or protest", NonEvent -> "no event". See LABELS.

Steps, in order:
  select   choose the prompt on the TRAIN split: the three candidates in CANDIDATES, the first 200 clips of the
           seed-0 stratified train order (8 per class; E1's 100 selection clips are the first half), metric
           macro-F1 under the primary rule; ties go to fewer unmatched answers, then the lower candidate number.
           Writes eval/e2_config.json, which freezes the prompt, the names and the matching rule.
  run      answer every test clip with the frozen prompt; resumable (results/e2/test_predictions.jsonl, ids, the
           model's raw answers and timings; the field is called caption_raw because E1's code writes it).
  score    results/e2/scores_test.json and per_clip_test.json.

Matching rule (primary, "strict"):
  1. Normalise the answer: split CamelCase words, lowercase, every run of non-letters becomes one space.
  2. Each class matches its shown name or one of its ALIASES as whole words, with an optional plural "s"/"es".
  3. Exactly one class matched: that class. No class matched: "unmatched". Two or more: "ambiguous".
     Unmatched and ambiguous answers count as wrong; both rates are reported.
  Sensitivity rule ("first"): the class whose match starts earliest; unmatched as before. Reported beside the
  primary numbers, never instead of them.

Metrics (whole test split, 1,391 clips; every clip counts, answered or not):
  - accuracy;
  - macro-F1 averaged over the 25 true classes. Unmatched and ambiguous are extra prediction columns, not
    classes. A class never predicted has precision 0 and F1 0. NonEvent is an ordinary class;
  - 95% bootstrap intervals for both (10,000 resamples of clips, seed 0, percentile);
  - per-class precision, recall and F1; the confusion matrix (25 true classes x 25 classes + unmatched +
    ambiguous); how often each name is predicted, with its position in the list (position bias);
  - baselines: always NonEvent (the largest class) and uniform random guessing (1,000 draws, seed 0);
  - latency (first call excluded) and peak GPU memory of the test run.

Committed outputs hold ids, class names (from ERA's file names) and the model's answers, never CapERA's
reference captions (E2 does not read them).

Example:
    .venv/bin/python -I eval/e2.py select
    .venv/bin/python -I eval/e2.py run [--limit 50]
    .venv/bin/python -I eval/e2.py score
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import capera  # noqa: E402
import e1  # noqa: E402

OUT = ROOT / "results" / "e2"
CONFIG = ROOT / "eval" / "e2_config.json"
SEED = 0
SELECT_CLIPS = 200
BOOT = 10_000
CHANCE_DRAWS = 1_000

# ERA class -> the name shown in the prompt. Order is the order of the list in the prompt.
LABELS = {
    "Baseball": "baseball", "Basketball": "basketball", "Boating": "boating", "CarRacing": "car racing",
    "Concert": "concert", "Conflict": "conflict", "Constructing": "construction", "Cycling": "cycling",
    "Fire": "fire", "Flood": "flood", "Harvesting": "harvesting", "Landslide": "landslide", "Mudslide": "mudslide",
    "ParadeProtest": "parade or protest", "Party": "party", "Ploughing": "ploughing", "PoliceChase": "police chase",
    "PostEarthquake": "earthquake damage", "ReligiousActivity": "religious activity", "Running": "running",
    "Soccer": "soccer", "Swimming": "swimming", "TrafficCollision": "traffic collision",
    "TrafficCongestion": "traffic congestion", "NonEvent": "no event",
}
# Other forms of the same names, fixed with the rule (not added after seeing test answers).
ALIASES = {
    "CarRacing": ["car race"], "Constructing": ["constructing"], "Flood": ["flooding", "flooded"],
    "Harvesting": ["harvest"], "NonEvent": ["non event", "nonevent", "none", "nothing"],
    "ParadeProtest": ["parade", "protest"], "Party": ["parties"], "Ploughing": ["plowing"],
    "PostEarthquake": ["earthquake", "post earthquake"], "ReligiousActivity": ["religious"],
    "Soccer": ["football"], "TrafficCongestion": ["traffic jam"],
}
UNMATCHED, AMBIGUOUS = "unmatched", "ambiguous"

_NAMES = list(LABELS.values())
CANDIDATES = [
    "These frames are from an aerial drone video, in time order. Which one of these events does the video show: "
    + ", ".join(_NAMES[:-1]) + ", or " + _NAMES[-1] + "? Answer with the event name only.",
    "Classify this aerial drone video into exactly one of these categories: " + ", ".join(_NAMES)
    + ". Answer with the category name only.",
    "These frames are from an aerial drone video, in time order.\nOptions:\n" + "\n".join(f"- {n}" for n in _NAMES)
    + "\nWhich option best describes the video? Answer with one option from the list.",
]


def normalise(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.sub(r"[^a-z]+", " ", text.lower()).strip()


def _pattern(form: str) -> re.Pattern:
    return re.compile(r"\b" + r"\s+".join(map(re.escape, form.split())) + r"(?:s|es)?\b")


PATTERNS = {cls: [_pattern(f) for f in [name] + ALIASES.get(cls, [])] for cls, name in LABELS.items()}


def parse(answer: str) -> dict:
    """{"strict": class | unmatched | ambiguous, "first": class | unmatched, "matched": [classes]}"""
    text = normalise(answer)
    start = {}
    for cls, pats in PATTERNS.items():
        for p in pats:
            m = p.search(text)
            if m and m.start() < start.get(cls, len(text) + 1):
                start[cls] = m.start()
    matched = sorted(start, key=lambda c: (start[c], c))
    strict = matched[0] if len(matched) == 1 else (UNMATCHED if not matched else AMBIGUOUS)
    return {"strict": strict, "first": matched[0] if matched else UNMATCHED, "matched": matched}


def check_classes(clips) -> None:
    found = sorted({c.cls for c in clips})
    if found != sorted(LABELS):
        sys.exit(f"class names differ from LABELS: {found}")


# ---------------------------------------------------------------- metrics
def confusion(true: list[str], pred: list[str]):
    """Counts [25 true classes] x [25 classes + unmatched + ambiguous], numpy int array."""
    import numpy as np
    cols = list(LABELS) + [UNMATCHED, AMBIGUOUS]
    ti = np.array([cols.index(t) for t in true])
    pi = np.array([cols.index(p) for p in pred])
    return np.bincount(ti * len(cols) + pi, minlength=len(LABELS) * len(cols)).reshape(len(LABELS), len(cols)), \
        ti, pi


def f1_table(cm) -> dict:
    """cm: [..., 25, 27] counts. Returns accuracy, per-class precision/recall/F1 and macro-F1 (arrays over any
    leading batch dimension)."""
    import numpy as np
    k = len(LABELS)
    tp = np.diagonal(cm[..., :, :k], axis1=-2, axis2=-1).astype(float)
    predicted = cm[..., :, :k].sum(axis=-2).astype(float)
    support = cm.sum(axis=-1).astype(float)
    p = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    r = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return {"accuracy": tp.sum(axis=-1) / support.sum(axis=-1), "precision": p, "recall": r, "f1": f,
            "macro_f1": f.mean(axis=-1)}


def metrics(true: list[str], pred: list[str], boot: bool) -> dict:
    import numpy as np
    cm, ti, pi = confusion(true, pred)
    t = f1_table(cm)
    res = {"accuracy": float(t["accuracy"]), "macro_f1": float(t["macro_f1"]),
           "unmatched": int(cm[:, -2].sum()), "ambiguous": int(cm[:, -1].sum())}
    if boot:
        rng = np.random.default_rng(SEED)
        n, ncols = len(ti), cm.shape[1]
        idx = rng.integers(0, n, (BOOT, n))
        codes = (ti[idx] * ncols + pi[idx]) + (np.arange(BOOT) * cm.size)[:, None]
        bcm = np.bincount(codes.ravel(), minlength=BOOT * cm.size).reshape(BOOT, *cm.shape)
        bt = f1_table(bcm)
        for m in ("accuracy", "macro_f1"):
            res[f"{m}_95"] = [float(np.percentile(bt[m], 2.5)), float(np.percentile(bt[m], 97.5))]
    return res


# ---------------------------------------------------------------- commands
def select() -> None:
    from vlm import Captioner
    if CONFIG.exists():
        sys.exit(f"{CONFIG} exists: the configuration is frozen")
    train = capera.load("train")
    check_classes(train)
    clips = capera.stratified_order(train, SEED)[:SELECT_CLIPS]
    cap = Captioner()
    results = []
    for k, prompt in enumerate(CANDIDATES):
        path = OUT / f"select_train_p{k + 1}.jsonl"
        e1.caption_clips(cap, clips, prompt, path, f"P{k + 1}")
        raw = e1.read_preds(path)
        true = [c.cls for c in clips]
        pred = [parse(raw[c.id]["caption_raw"])["strict"] for c in clips]
        m = metrics(true, pred, boot=False)
        results.append({"id": f"P{k + 1}", "prompt": prompt, **m})
        print(f"P{k + 1}: macro-F1 {m['macro_f1']:.3f}  accuracy {m['accuracy']:.3f}  "
              f"unmatched {m['unmatched']}  ambiguous {m['ambiguous']}")
    best = min(results, key=lambda r: (-r["macro_f1"], r["unmatched"], r["id"]))
    config = {"frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prompt": best["prompt"], "prompt_id": best["id"],
              "labels": LABELS, "aliases": ALIASES, "rule": "strict (eval/e2.py docstring); sensitivity: first",
              "frames": e1.FRAMES, "image_splitting": e1.SPLIT_IMAGES, "max_new_tokens": e1.MAX_NEW_TOKENS,
              "dtype": "bfloat16", "decoding": "greedy", "window": "each clip's whole length",
              "note": "run before the demo freeze with E1's settings; rerun if the demo changes them",
              "selection": {"split": "train", "clips": len(clips), "seed": SEED, "metric": "macro_f1",
                            "ids": [c.id for c in clips], "candidates": results}}
    CONFIG.write_text(json.dumps(config, indent=1) + "\n")
    print(f"chose {best['id']}; wrote {CONFIG}")


def frozen_config() -> dict:
    config = json.loads(CONFIG.read_text())
    if config["labels"] != LABELS or config["aliases"] != ALIASES:
        sys.exit("LABELS or ALIASES differ from the frozen eval/e2_config.json")
    return config


def run(limit: int | None) -> None:
    import torch
    from vlm import Captioner
    config = frozen_config()
    clips = capera.stratified_order(capera.load("test"), SEED)
    check_classes(clips)
    if limit:
        clips = clips[:limit]
    cap = Captioner()
    torch.cuda.reset_peak_memory_stats()
    e1.caption_clips(cap, clips, config["prompt"], OUT / "test_predictions.jsonl", "test")
    free, total = torch.cuda.mem_get_info()
    (OUT / "memory_test.json").write_text(json.dumps({
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "device_used_mib": (total - free) / 2**20, "device_total_mib": total / 2**20,
        "note": "the last run session only (the run is resumable); device_used includes the CUDA context and "
                "other processes"}, indent=1) + "\n")


def score() -> None:
    import numpy as np
    import torch
    config = frozen_config()
    clips = {c.id: c for c in capera.load("test")}
    check_classes(clips.values())
    raw = e1.read_preds(OUT / "test_predictions.jsonl")
    missing = sorted(set(clips) - set(raw))
    if missing:
        sys.exit(f"{len(missing)} test clips have no answer yet: run `eval/e2.py run`")
    ids = sorted(clips)
    parsed = {i: parse(raw[i]["caption_raw"]) for i in ids}
    true = [clips[i].cls for i in ids]
    strict = [parsed[i]["strict"] for i in ids]
    first = [parsed[i]["first"] for i in ids]

    cm, _, _ = confusion(true, strict)
    table = f1_table(cm)
    names = list(LABELS)
    predicted_counts = cm.sum(axis=0)
    cols = names + [UNMATCHED, AMBIGUOUS]

    majority = max(names, key=lambda c: true.count(c))
    rng = np.random.default_rng(SEED)
    draws = [metrics(true, list(rng.choice(names, len(true))), boot=False) for _ in range(CHANCE_DRAWS)]

    warm = [r for r in raw.values() if not r["first_call"]]
    gen = sorted(r["generate_s"] for r in warm)
    total = sorted(r["decode_s"] + r["preprocess_s"] + r["generate_s"] for r in warm)
    q = lambda v, p: v[min(len(v) - 1, round(p * (len(v) - 1)))]  # noqa: E731
    words = [len(e1.sanitise(raw[i]["caption_raw"]).split()) for i in ids]
    shown = {normalise(n) for n in LABELS.values()}
    summary = {
        "experiment": "E2 event recognition, CapERA test split (ERA classes), zero-shot",
        "config": {k: v for k, v in config.items() if k != "selection"},
        "selection": {"split": "train", "clips": config["selection"]["clips"], "metric": "macro_f1",
                      "candidates": [{k: c[k] for k in ("id", "macro_f1", "accuracy", "unmatched", "ambiguous")}
                                     for c in config["selection"]["candidates"]]},
        "clips": len(ids),
        "strict": metrics(true, strict, boot=True),
        "first_mention": metrics(true, first, boot=True),
        "baselines": {f"always_{majority}": metrics(true, [majority] * len(true), boot=False),
                      "uniform_random": {"draws": CHANCE_DRAWS,
                                         "accuracy": statistics.mean(d["accuracy"] for d in draws),
                                         "macro_f1": statistics.mean(d["macro_f1"] for d in draws)}},
        "per_class": {c: {"support": int(cm[k].sum()), "predicted": int(predicted_counts[k]),
                          "precision": float(table["precision"][k]), "recall": float(table["recall"][k]),
                          "f1": float(table["f1"][k])} for k, c in enumerate(names)},
        "predicted_by_list_position": [{"position": k + 1, "class": c, "name": LABELS[c],
                                        "predicted": int(predicted_counts[k]), "support": int(cm[k].sum())}
                                       for k, c in enumerate(names)],
        "confusion": {"rows": names, "columns": cols, "counts": cm.tolist()},
        "answers": {"mean_words": statistics.mean(words), "median_words": statistics.median(words),
                    "exactly_a_shown_name": sum(normalise(raw[i]["caption_raw"]) in shown for i in ids),
                    "truncated_at_max_new_tokens": sum(raw[i]["truncated"] for i in ids)},
        "latency_s": {"generate_median": q(gen, 0.5), "generate_p90": q(gen, 0.9),
                      "per_clip_total_median": q(total, 0.5), "per_clip_total_p90": q(total, 0.9),
                      "note": "first call excluded; decode and preprocess on the CPU, generation on the GPU"},
        "memory": (json.loads((OUT / "memory_test.json").read_text()) if (OUT / "memory_test.json").exists()
                   else None),
        "env": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__},
    }
    (OUT / "scores_test.json").write_text(json.dumps(summary, indent=1) + "\n")
    (OUT / "per_clip_test.json").write_text(json.dumps(
        [{"id": i, "cls": clips[i].cls, "answer": e1.sanitise(raw[i]["caption_raw"]), **parsed[i]} for i in ids],
        indent=0) + "\n")
    s, f = summary["strict"], summary["first_mention"]
    print(f"strict: accuracy {s['accuracy']:.3f} {s['accuracy_95']}  macro-F1 {s['macro_f1']:.3f} "
          f"{s['macro_f1_95']}  unmatched {s['unmatched']}  ambiguous {s['ambiguous']}")
    print(f"first:  accuracy {f['accuracy']:.3f}  macro-F1 {f['macro_f1']:.3f}")
    print("baselines:", json.dumps({k: {m: round(v, 3) for m, v in b.items() if isinstance(v, float)}
                                    for k, b in summary["baselines"].items()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["select", "run", "score"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    {"select": select, "run": lambda: run(args.limit), "score": score}[args.command]()


if __name__ == "__main__":
    main()

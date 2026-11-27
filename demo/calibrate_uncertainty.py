"""Cutoff for the demo report's "uncertain items", fixed before the demo runs on any demo video.

Score: a caption's mean token log-probability (eval/vlm.py, special tokens excluded). Rule: a window is uncertain
when its score is below the 10th percentile of the scores on E1's 100 prompt-selection clips (CapERA train,
eval/e1_config.json), captioned with E1's frozen settings. Those captions must equal E1's committed ones
(results/e1/select_train_p1.jsonl) word for word; the script stops if any differs, since that would mean the
captioner changed.

Writes results/demo/uncertainty_train.jsonl (ids, our captions, scores) and demo/uncertainty.json (the cutoff).

Example:
    .venv/bin/python -I demo/calibrate_uncertainty.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import capera  # noqa: E402

CONFIG = ROOT / "eval" / "e1_config.json"
E1_CAPTIONS = ROOT / "results" / "e1" / "select_train_p1.jsonl"
OUT = ROOT / "results" / "demo" / "uncertainty_train.jsonl"
CUTOFF = ROOT / "demo" / "uncertainty.json"
PERCENTILE = 10


def main() -> None:
    from vlm import Captioner
    if CUTOFF.exists():
        sys.exit(f"{CUTOFF} exists: the cutoff is fixed")
    cfg = json.loads(CONFIG.read_text())
    clips = {c.id: c for c in capera.load("train")}
    ids = cfg["selection"]["ids"]
    e1 = {r["id"]: r["caption_raw"] for r in map(json.loads, E1_CAPTIONS.read_text().splitlines())}
    cap = Captioner()
    recs = []
    for i in ids:
        c = clips[i]
        g = cap.generate(cap.inputs(cap.frames(c.path, c.duration_s, cfg["frames"]), cfg["prompt"],
                                    cfg["image_splitting"]), cfg["max_new_tokens"], logprobs=True)
        recs.append({"id": i, "caption_raw": g["caption"], "mean_logprob": g["mean_logprob"],
                     "new_tokens": g["new_tokens"], "same_as_e1": g["caption"] == e1[i]})
    differ = [r["id"] for r in recs if not r["same_as_e1"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in recs))
    if differ:
        sys.exit(f"{len(differ)} captions differ from E1's ({differ[:5]}...): the captioner changed; no cutoff written")
    scores = sorted(r["mean_logprob"] for r in recs)
    cutoff = statistics.quantiles(scores, n=100, method="inclusive")[PERCENTILE - 1]
    CUTOFF.write_text(json.dumps({
        "frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "score": "mean token log-probability of the caption",
        "rule": f"uncertain when below the {PERCENTILE}th percentile on E1's {len(ids)} train selection clips",
        "cutoff": cutoff, "clips": len(ids), "percentile": PERCENTILE,
        "train_scores": {"min": scores[0], "median": statistics.median(scores), "max": scores[-1]},
        "captions_identical_to_e1": True}, indent=1) + "\n")
    print(f"cutoff {cutoff:.4f} (scores {scores[0]:.3f} .. {scores[-1]:.3f}, median {statistics.median(scores):.3f});"
          f" all {len(ids)} captions identical to E1's; wrote {CUTOFF}")


if __name__ == "__main__":
    main()

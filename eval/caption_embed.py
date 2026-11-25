"""Embedding similarity for the caption-level gating check: a second, meaning-level caption metric beside CIDEr-D,
which can be blind to temporal input (TemporalBench, arXiv:2410.10818, found the same CIDEr from 1 and 64
frames). Reads the stored captions (results/gate_captions/predictions.jsonl) and CapERA's references (data/, never
written out); runs no captioner.

Added after the CIDEr results were known, so the paper labels it post hoc. The analysis below was fixed in this
docstring before any similarity was computed, and mirrors eval/gate_captions.py's CIDEr analysis:
- Embedding: sentence-transformers/all-mpnet-base-v2 from models/all-mpnet-base-v2, mean-pooled over tokens and
  L2-normalised, as the model's own sentence-transformers configuration declares (asserted at load). Captions
  are sanitised exactly as in E1.
- Score of a caption: its mean cosine similarity to the clip's reference captions.
- Primary: per clip, a condition's score minus dense's; the mean over clips with a 95% bootstrap interval
  (10,000 resamples, seed 0, clips resampled).
- Equal cost: per clip, a gate's difference minus held output's, held output taken on the chord between the
  hold rates at the gate's per-clip cost (as in eval/gate_captions.py).
- Resolution: dense captions scored against the next clip's references (the same derangement as CIDEr's shuffled
  control); score minus dense's, with its interval.
- Sensitivity (not a quality measure): a condition's cosine similarity to dense's caption, mean over clips.

The download (about 440 MB), by explicit file names (this CLI reads --include patterns as file names):
    .venv/bin/hf download sentence-transformers/all-mpnet-base-v2 model.safetensors vocab.txt config.json \\
        tokenizer.json tokenizer_config.json special_tokens_map.json modules.json sentence_bert_config.json \\
        config_sentence_transformers.json 1_Pooling/config.json --local-dir models/all-mpnet-base-v2

Example:
    .venv/bin/python -I eval/caption_embed.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import capera  # noqa: E402
import e1  # noqa: E402
import gate_captions as gc  # noqa: E402

MODEL = ROOT / "models" / "all-mpnet-base-v2"
OUT = ROOT / "results" / "gate_captions" / "embed_scores.json"


class Embedder:
    def __init__(self, path: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        pool = json.loads((path / "1_Pooling" / "config.json").read_text())
        assert pool.get("pooling_mode_mean_tokens") and not any(
            pool.get(k) for k in ("pooling_mode_cls_token", "pooling_mode_max_tokens")), pool
        modules = json.loads((path / "modules.json").read_text())
        assert any("Normalize" in m.get("type", "") for m in modules), modules
        self.max_len = json.loads((path / "sentence_bert_config.json").read_text())["max_seq_length"]
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval()
        self.torch = torch

    def __call__(self, texts: list[str], batch: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                t = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt")
                h = self.model(**t).last_hidden_state
                m = t["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
                out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
        return np.concatenate(out)


def main() -> None:
    if not (MODEL / "model.safetensors").exists():
        sys.exit(f"{MODEL} missing: download it first (see this script's docstring)")
    emb = Embedder(MODEL)
    same = emb(["A crowd watches a football match.", "A crowd watches a football match."])
    assert abs(float(same[0] @ same[1]) - 1.0) < 1e-5  # identical sentences embed identically

    recs = [json.loads(l) for l in (gc.OUT / "predictions.jsonl").read_text().splitlines()]
    ids = [r["id"] for r in recs]
    refs_all = {c.id: c.refs for c in capera.load("test")}
    names = gc.conditions()
    cap = {k: [e1.sanitise(r["conditions"][k]["caption_raw"]) for r in recs] for k in names}
    ref_lists = [[e1.sanitise(x) for x in refs_all[i]] for i in ids]
    ref_flat = [x for rs in ref_lists for x in rs]
    ref_emb = emb(ref_flat)
    ref_idx, at = [], 0
    for rs in ref_lists:
        ref_idx.append(slice(at, at + len(rs)))
        at += len(rs)
    E = {k: emb(cap[k]) for k in names}

    def score(vecs: np.ndarray, ref_of) -> np.ndarray:  # mean cosine to the references of clip ref_of(j)
        return np.array([float((ref_emb[ref_idx[ref_of(j)]] @ vecs[j]).mean()) for j in range(len(ids))])

    S = {k: score(E[k], lambda j: j) for k in names}
    delta = {k: S[k] - S["dense"] for k in names}
    rng = np.random.default_rng(gc.BOOT_SEED)
    table = {k: {"score_mean": float(S[k].mean()), "minus_dense": gc.interval(delta[k], rng),
                 "cos_to_dense_caption": float(np.mean(np.sum(E[k] * E["dense"], axis=1)))} for k in names}
    costs = {k: np.array([r["conditions"][k]["cost"] if r["conditions"][k]["cost"] is not None else np.nan
                          for r in recs]) for k in names}
    hold_pts = [("dense", gc.FPS)] + [(f"hold_{s}", gc.FPS / s) for s in gc.HOLD_STRIDES]
    eq = {}
    for k in names:
        if gc.family(k) not in ("A", "B", "C"):
            continue
        d = []
        for j in range(len(ids)):
            h = gc.chord([(c, delta[hk][j]) for hk, c in hold_pts], costs[k][j])
            if h is not None:
                d.append(delta[k][j] - h)
        eq[k] = gc.interval(d, rng)
    nxt = lambda j: (j + 1) % len(ids)  # noqa: E731  same derangement as CIDEr's shuffled control
    shuffled = score(E["dense"], nxt)
    out = {"experiment": "embedding similarity for the caption-level gating check (definitions: "
                         "eval/caption_embed.py; post hoc relative to the CIDEr analysis)",
           "model": "sentence-transformers/all-mpnet-base-v2 (mean pooling, L2-normalised)",
           "clips": len(ids), "bootstrap": {"resamples": gc.BOOT, "seed": gc.BOOT_SEED},
           "conditions": table, "equal_cost_vs_hold": eq,
           "shuffled_control": {"score_mean": float(shuffled.mean()),
                                "minus_dense": gc.interval(shuffled - S["dense"], rng)}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'condition':10s} {'score':>6s} {'minus dense [95%]':>26s} {'cos to dense':>12s}")
    for k in names:
        t = table[k]
        ci = t["minus_dense"]
        print(f"{k:10s} {t['score_mean']:6.3f} {ci['mean']:+8.4f} [{ci['lo95']:+.4f}, {ci['hi95']:+.4f}] "
              f"{t['cos_to_dense_caption']:12.3f}")
    sc = out["shuffled_control"]["minus_dense"]
    print(f"shuffled control: {sc['mean']:+.4f} [{sc['lo95']:+.4f}, {sc['hi95']:+.4f}]")
    for k, v in eq.items():
        if v:
            print(f"equal cost {k:8s} {v['mean']:+.4f} [{v['lo95']:+.4f}, {v['hi95']:+.4f}]")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

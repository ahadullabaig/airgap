"""Where a caption's time goes in E1's configuration (eval/e1_config.json: bf16, 4 frames, prompt P1, greedy,
up to 48 new tokens), split so the paper can state the vision encoder's share of end-to-end latency.

For the first CLIPS clips of E1's test order, each stage is timed separately, with a CUDA synchronise around every
GPU stage:
  decode      the four frames from the video (CPU, PyAV)
  preprocess  the processor: resize, normalise, tokenise the prompt (CPU)
  encoder     vision tower + connector on the four frames (the model's own get_image_features)
  prefill     a generate() call limited to one new token, minus the encoder (the language model reads the prompt
              and image tokens)
  decode_lm   the full generate() minus the one-token call (the remaining tokens, one at a time)
GPU stages use the median of REPEATS runs after one warm-up per clip. Clocks are not pinned: this is the
deployed configuration.

Example:
    .venv/bin/python -I eval/encoder_share.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import capera  # noqa: E402
import e1  # noqa: E402

CLIPS, REPEATS = 20, 3
OUT = ROOT / "results" / "e1" / "encoder_share.json"


def main() -> None:
    import torch
    from vlm import Captioner
    prompt = json.loads(e1.CONFIG.read_text())["prompt"]
    cap = Captioner()
    model = cap.model.model

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0, r

    rows = []
    for c in capera.stratified_order(capera.load("test"), e1.SEED)[:CLIPS]:
        t0 = time.perf_counter()
        images = cap.frames(c.path, c.duration_s, e1.FRAMES)
        t1 = time.perf_counter()
        inp = cap.inputs(images, prompt, e1.SPLIT_IMAGES)
        t2 = time.perf_counter()
        kw = {k: inp[k] for k in ("pixel_attention_mask",) if k in inp}
        with torch.inference_mode():
            enc = lambda: model.get_image_features(inp["pixel_values"], return_dict=True, **kw)  # noqa: E731
            one = lambda: cap.model.generate(**inp, max_new_tokens=1, do_sample=False)  # noqa: E731
            full = lambda: cap.model.generate(**inp, max_new_tokens=e1.MAX_NEW_TOKENS, do_sample=False)  # noqa: E731
            for f in (enc, one, full):  # warm-up
                f()
            t_enc = statistics.median(timed(enc)[0] for _ in range(REPEATS))
            t_one = statistics.median(timed(one)[0] for _ in range(REPEATS))
            runs = [timed(full) for _ in range(REPEATS)]
        t_full = statistics.median(t for t, _ in runs)
        new_tokens = int(runs[0][1].shape[1] - inp["input_ids"].shape[1])
        rows.append({"id": c.id, "decode_s": t1 - t0, "preprocess_s": t2 - t1, "encoder_s": t_enc,
                     "prefill_s": max(t_one - t_enc, 0.0), "decode_lm_s": max(t_full - t_one, 0.0),
                     "generate_s": t_full, "new_tokens": new_tokens})
        print(f"{c.id:28s} enc {1e3 * t_enc:6.1f} ms  prefill {1e3 * (t_one - t_enc):6.1f}  decode "
              f"{1e3 * (t_full - t_one):6.1f}  ({new_tokens} tokens)", flush=True)

    stages = ("decode_s", "preprocess_s", "encoder_s", "prefill_s", "decode_lm_s")
    med = {k: statistics.median(r[k] for r in rows) for k in stages + ("generate_s", "new_tokens")}
    total = [sum(r[k] for k in stages) for r in rows]
    shares = {k: statistics.median(r[k] / t for r, t in zip(rows, total)) for k in stages}
    out = {"what": "per-stage time of one caption in E1's configuration (definitions: eval/encoder_share.py)",
           "clips": len(rows), "repeats": REPEATS, "gpu": torch.cuda.get_device_name(),
           "median_s": med, "median_share_of_total": shares, "median_total_s": statistics.median(total),
           "rows": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("median (ms):", {k: round(1e3 * v, 1) if k != "new_tokens" else v for k, v in med.items()})
    print("median share of total:", {k: round(v, 3) for k, v in shares.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()

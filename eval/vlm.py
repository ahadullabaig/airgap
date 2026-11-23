"""SmolVLM2 captioning for the evaluation, with the demo's settings (demo/caption_clip.py): local weights
only, bf16, frames passed as a multi-image chat message, no image splitting, greedy decoding.

The prompt is a parameter, and preprocessing (frames to model inputs) is a separate step from
generation, so E2 and the gating check can reuse both.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from sampling import sample_window  # noqa: E402

MODEL = ROOT / "models" / "SmolVLM2-500M-Video-Instruct"


class Captioner:
    def __init__(self, model_dir: Path = MODEL):
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()

    @staticmethod
    def frames(path: Path, duration_s: float, k: int) -> list:
        """k frames spread evenly over the whole clip (PIL images)."""
        return [img for _, img in sample_window(str(path), 0.0, duration_s, k)]

    def inputs(self, images: list, prompt: str, split: bool = False):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inp = self.processor(text=text, images=[images], do_image_splitting=split, return_tensors="pt")
        inp = inp.to("cuda", dtype=torch.bfloat16)
        inp["input_ids"] = inp["input_ids"].long()
        return inp

    @torch.inference_mode()
    def generate(self, inp, max_new_tokens: int) -> dict:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        new = out[0, inp["input_ids"].shape[1]:]
        return {"caption": self.processor.decode(new, skip_special_tokens=True).strip(),
                "new_tokens": int(new.shape[0]), "prompt_tokens": int(inp["input_ids"].shape[1]),
                "truncated": int(new.shape[0]) >= max_new_tokens, "generate_s": t}

"""Caption one window of an aerial video with SmolVLM2, fully offline.

Week-1 smoke test for Part A: loads the model from a local folder, captions a
single window, and reports latency, peak VRAM and the vision encoder's
sequence length N (the number Part B builds on).

Example:
    unshare -rn .venv/bin/python demo/caption_clip.py --video data/clips/x.webm
"""

import os

# Never touch the network: set before transformers / huggingface_hub import.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampling import probe, sample_window  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "SmolVLM2-500M-Video-Instruct"
DEFAULT_PROMPT = (
    "These frames are from an aerial drone video, in time order. "
    "Describe in one or two sentences what is happening on the ground."
)


def driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def find_vision_encoder(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if name.endswith("vision_model"):
            return module
    raise RuntimeError("no module named *vision_model found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--start", type=float, default=0.0, help="window start, seconds")
    ap.add_argument("--window", type=float, default=8.0, help="window length, seconds")
    ap.add_argument("--frames", type=int, default=4, help="frames sampled per window")
    ap.add_argument("--split", action="store_true", help="enable image splitting (more tiles per frame)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--runs", type=int, default=3, help="generate this many times; the first call is cold")
    ap.add_argument("--json", help="also write the result record to this path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to PyTorch")
    if not Path(args.model, "config.json").exists():
        sys.exit(f"model not found at {args.model}")
    device = torch.device("cuda")

    info = probe(args.video)
    t0 = time.perf_counter()
    frames = sample_window(args.video, args.start, args.window, args.frames)
    t_decode = time.perf_counter() - t0

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to(device).eval()
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0

    # Record the vision encoder's output shape: (images, N, hidden).
    seen_shapes: list[tuple[int, ...]] = []
    encoder = find_vision_encoder(model)
    encoder.register_forward_hook(
        lambda _m, _i, out: seen_shapes.append(tuple((out[0] if isinstance(out, tuple) else out.last_hidden_state).shape))
    )

    content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": args.prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=[[img for _, img in frames]],
        do_image_splitting=args.split,
        return_tensors="pt",
    ).to(device, dtype=torch.bfloat16)
    inputs["input_ids"] = inputs["input_ids"].long()  # keep token ids integer after the dtype cast
    prompt_tokens = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    gen_times = []
    for _ in range(max(1, args.runs)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        gen_times.append(round(time.perf_counter() - t0, 3))
        del seen_shapes[1:]  # same input every run; keep the first record only
    free_b, total_b = torch.cuda.mem_get_info()

    new_ids = out_ids[0, prompt_tokens:]
    caption = processor.decode(new_ids, skip_special_tokens=True).strip()

    record = {
        "video": info.__dict__,
        "window": {"start_s": args.start, "length_s": args.window, "frame_times_s": [round(t, 3) for t, _ in frames]},
        "image_splitting": args.split,
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "vision_encoder_output_shapes": [list(s) for s in seen_shapes],
        "prompt_tokens": prompt_tokens,
        "new_tokens": int(new_ids.shape[0]),
        "caption": caption,
        "timing_s": {
            "decode": round(t_decode, 3),
            "model_load": round(t_load, 3),
            "generate_cold": gen_times[0],
            "generate_warm": gen_times[1:],
        },
        "memory_mib": {
            # PyTorch allocator only: excludes the CUDA context and other processes.
            "torch_peak_allocated": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "torch_peak_reserved": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            # Whole device as the driver sees it, after generation: includes the CUDA context.
            "device_used": round((total_b - free_b) / 2**20, 1),
            "device_total": round(total_b / 2**20, 1),
        },
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "driver": driver_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "hf_offline": os.environ["HF_HUB_OFFLINE"],
        },
    }
    print(json.dumps(record, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

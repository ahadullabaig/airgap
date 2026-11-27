"""Offline mission report for an aerial video (PLAN.md 2.1-2.4).

Splits the video into equal windows of about 6 s, captions each with E1's frozen settings (eval/e1_config.json:
prompt P1, 4 frames per window, no image splitting, bf16, greedy, 48 new tokens), and writes a Markdown report
from a fixed template: notable events, the timeline, uncertain items and a summary paragraph, which is the only
part the language model writes freely. A JSON record beside it keeps every window's times, caption, score and
timings, the settings and peak GPU memory.

Windows: n = round(duration / 6 s), at least 1, each duration / n long, so no short tail window is left over.
A window that fails to decode is reported as failed and the run goes on.

Notable events: every window whose caption contains a word from EVENTS, a fixed list of event words built from
ERA's 25 event classes before any demo video was run. Runs of consecutive windows with the same event become one
line, with the caption of its most confident window. Captions with no event word appear only in the timeline.

Uncertain items: windows whose caption's mean token log-probability is below the cutoff in demo/uncertainty.json
(fixed on CapERA train clips by demo/calibrate_uncertainty.py before any demo video was run).

Summary: the same model, text only, given the timeline's captions; at most 128 new tokens, cut back to the last
full sentence if it runs out. It is the only free text in the report and is marked as written by the model.

The model always loads from local files only. --offline also makes any internet connection opened through Python's
socket module fail with an error; native libraries could bypass that, so the guarantee is running with no network
at all, as the acceptance test does:
    unshare -rn .venv/bin/python -I demo/run_mission.py --video data/clips/x.mp4 --offline

Example:
    .venv/bin/python -I demo/run_mission.py --video data/clips/x.mp4 --offline [--out results/demo/x]
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "eval"))
from sampling import probe, sample_window  # noqa: E402

E1_CONFIG = ROOT / "eval" / "e1_config.json"
CUTOFF = ROOT / "demo" / "uncertainty.json"
WINDOW_S = 6.0
SUMMARY_TOKENS = 128
# Event words, from ERA's 25 event classes; fixed before any demo video was run. Matched as whole words in the
# lowercased caption. (label, regular expression)
EVENTS = [
    ("fire or smoke", r"fires?|burning|burnt|flames?|smoke|smoky"),
    ("flood", r"flood(?:s|ed|ing|water|waters)?"),
    ("landslide or mudslide", r"landslides?|mudslides?|mudflows?"),
    ("collapsed or damaged buildings", r"collapsed?|rubble|debris|ruins|destroyed|damaged"),
    ("traffic accident", r"accidents?|crash(?:ed|es)?|collisions?|collided"),
    ("traffic jam", r"traffic jams?|congest(?:ed|ion)|heavy traffic|gridlock"),
    ("police or emergency vehicles", r"police|ambulances?|fire (?:trucks?|engines?)|firefighters?|emergency"),
    ("crowd or gathering", r"crowds?|crowded|gathering|gathered|protest(?:s|ers)?|parades?|marching|marchers|rally|"
                           r"concerts?|festivals?|party|ceremony|worship(?:pers)?|prayers?|praying"),
    ("race", r"races?|racing|racetracks?"),
    ("sport", r"soccer|football|baseball|basketball|tennis|stadium|players?|match"),
    ("construction", r"construction|cranes?|excavators?|bulldozers?"),
    ("farming", r"harvest(?:ing|er|ers)?|tractors?|plough(?:ing|ed)?|plow(?:ing|ed)?"),
    ("boats or swimming", r"boats?|boating|ships?|kayaks?|swim|swimming|swimmers?"),
    ("cycling or running", r"cycling|cyclists?|bicycles?|bikes?|runners?|joggers?|jogging|marathon|"
                           r"(?:people|person|man|woman|men|women) (?:are |is )?running"),
]
EVENT_RES = [(label, re.compile(rf"\b(?:{rx})\b")) for label, rx in EVENTS]
SUMMARY_PROMPT = (
    "Below are short descriptions of consecutive parts of one aerial drone video, with their times.\n\n{lines}\n\n"
    "Write a summary of the whole video in two or three sentences. Mention only things that appear in the "
    "descriptions.")


def block_network() -> None:
    """Make internet (IPv4/IPv6) connections opened through Python's socket module fail."""
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(fn):
        def wrapped(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise OSError(f"--offline: blocked a network connection to {address}")
            return fn(self, address)
        return wrapped
    socket.socket.connect, socket.socket.connect_ex = guard(connect), guard(connect_ex)


def clock(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def plan_windows(duration_s: float) -> list[tuple[float, float]]:
    n = max(1, round(duration_s / WINDOW_S))
    w = duration_s / n
    return [(k * w, (k + 1) * w) for k in range(n)]


def notable(wins: list[dict]) -> list[dict]:
    """Runs of consecutive windows whose captions share an event word, in time order."""
    runs = []
    for label, rx in EVENT_RES:
        cur = None
        for w in wins:
            if "caption" in w and rx.search(w["caption"].lower()):
                if cur and cur["windows"][-1]["index"] == w["index"] - 1:
                    cur["windows"].append(w)
                else:
                    cur = {"event": label, "windows": [w]}
                    runs.append(cur)
            else:
                cur = None
    for r in runs:
        best = max(r["windows"], key=lambda w: w["mean_logprob"])
        r |= {"start_s": r["windows"][0]["start_s"], "end_s": r["windows"][-1]["end_s"], "caption": best["caption"],
              "uncertain": sum(w["uncertain"] for w in r["windows"]), "indices": [w["index"] for w in r["windows"]]}
        del r["windows"]
    merged: dict[tuple, dict] = {}  # one line per run of windows; a caption naming two events lists both
    for r in runs:
        key = tuple(r["indices"])
        if key in merged:
            merged[key]["event"] += "; " + r["event"]
        else:
            merged[key] = r
    return sorted(merged.values(), key=lambda r: (r["start_s"], r["end_s"]))


def trim_summary(text: str, truncated: bool) -> str:
    text = " ".join(text.split())
    if truncated:
        ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
        text = text[:ends[-1]] if ends else text + " ..."
    return text


def report_md(video: Path, info, rec: dict) -> str:
    wins = rec["windows"]
    ok = [w for w in wins if "caption" in w]
    unsure = [w for w in ok if w["uncertain"]]
    failed = [w for w in wins if "error" in w]
    t = rec["timing_s"]
    lines = [
        f"# Mission report: {video.name}",
        "",
        f"{clock(info.duration_s)} of video, {info.width}x{info.height} at {info.fps:.0f} fps, in {len(wins)} windows "
        f"of {wins[0]['end_s'] - wins[0]['start_s']:.1f} s. Generated offline by SmolVLM2-500M on "
        f"{rec['env']['gpu']} in {t['total']:.0f} s.",
        "",
        "## Summary",
        "",
        f"*Written by the model from the timeline below; check it against the timeline.*",
        "",
        rec["summary"]["text"] or "(no summary)",
        "",
        "## Notable events",
        "",
    ]
    if rec["notable"]:
        lines.append("Windows whose caption names an event, joined where the same event runs on. The caption shown "
                     "is the run's most confident one.")
        lines.append("")
        for r in rec["notable"]:
            n = len(r["indices"])
            extra = f" ({n} windows" + (f", {r['uncertain']} uncertain" if r["uncertain"] else "") + ")" \
                if n > 1 else (" (uncertain)" if r["uncertain"] else "")
            lines.append(f"- {clock(r['start_s'])}-{clock(r['end_s'])} **{r['event']}**{extra}: {r['caption']}")
    else:
        lines.append("No caption names an event; see the timeline.")
    lines += [
        "",
        "## Timeline",
        "",
        "| Time | What the model sees | |",
        "|---|---|---|",
    ]
    for w in wins:
        span = f"{clock(w['start_s'])}-{clock(w['end_s'])}"
        if "error" in w:
            lines.append(f"| {span} | *window could not be captioned* | failed |")
        else:
            lines.append(f"| {span} | {w['caption'].replace('|', '/')} | {'uncertain' if w['uncertain'] else ''} |")
    lines += ["", "## Uncertain items", ""]
    if unsure:
        lines.append(f"Windows where the model was least sure of its own words (mean token log-probability below "
                     f"{rec['uncertainty']['cutoff']:.2f}). Check these in the video.")
        lines.append("")
        lines += [f"- {clock(w['start_s'])}-{clock(w['end_s'])}: {w['caption']} (score {w['mean_logprob']:.2f})"
                  for w in unsure]
    else:
        lines.append("None: every caption scored above the cutoff.")
    if failed:
        lines += ["", f"{len(failed)} window(s) could not be captioned: "
                  + ", ".join(f"{clock(w['start_s'])}-{clock(w['end_s'])}" for w in failed) + "."]
    pw = rec["per_window_s"]
    per = (f"median {pw['median']:.2f} s per window: decode {pw['decode_median']:.2f} s, generate "
           f"{pw['generate_median']:.2f} s" if ok else "no window was captioned")
    lines += ["", "## Run", "",
              f"Model load {t['model_load']:.1f} s; captioning {t['captioning']:.1f} s ({per}); summary "
              f"{t['summary']:.1f} s. Peak GPU memory {rec['memory_mib']['torch_peak_allocated']:.0f} MiB allocated "
              f"by PyTorch.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--offline", action="store_true", help="make Python-level internet connections fail")
    ap.add_argument("--out", type=Path, help="output folder (default results/demo/<video name>)")
    args = ap.parse_args()
    if args.offline:
        block_network()
    import torch
    from vlm import Captioner

    t_start = time.perf_counter()
    cfg = json.loads(E1_CONFIG.read_text())
    cutoff = json.loads(CUTOFF.read_text())
    info = probe(str(args.video))
    windows = plan_windows(info.duration_s)
    out = args.out or ROOT / "results" / "demo" / args.video.stem
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cap = Captioner()
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()

    recs = []
    t_cap = time.perf_counter()
    for k, (a, b) in enumerate(windows):
        w = {"index": k, "start_s": round(a, 3), "end_s": round(b, 3)}
        try:
            t0 = time.perf_counter()
            frames = sample_window(str(args.video), a, b - a, cfg["frames"])
            t1 = time.perf_counter()
            inp = cap.inputs([img for _, img in frames], cfg["prompt"], cfg["image_splitting"])
            t2 = time.perf_counter()
            g = cap.generate(inp, cfg["max_new_tokens"], logprobs=True)
            w |= {"frame_times_s": [round(t, 3) for t, _ in frames], "caption": " ".join(g["caption"].split()),
                  "mean_logprob": g["mean_logprob"], "uncertain": g["mean_logprob"] < cutoff["cutoff"],
                  "new_tokens": g["new_tokens"], "truncated": g["truncated"], "decode_s": round(t1 - t0, 4),
                  "preprocess_s": round(t2 - t1, 4), "generate_s": round(g["generate_s"], 4)}
            print(f"[{k + 1}/{len(windows)}] {clock(a)}-{clock(b)}  {t2 - t0 + g['generate_s']:.2f} s"
                  f"{'  uncertain' if w['uncertain'] else ''}  {w['caption']}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad window must not end a live run
            w["error"] = repr(e)[:300]
            print(f"[{k + 1}/{len(windows)}] {clock(a)}-{clock(b)}  FAILED  {w['error']}", flush=True)
        recs.append(w)
    t_cap = time.perf_counter() - t_cap

    ok = [w for w in recs if "caption" in w]
    t0 = time.perf_counter()
    summary = {"prompt": SUMMARY_PROMPT, "max_new_tokens": SUMMARY_TOKENS, "text": ""}
    if ok:
        lines = "\n".join(f"{clock(w['start_s'])}-{clock(w['end_s'])}: {w['caption']}" for w in ok)
        g = cap.generate(cap.text_inputs(SUMMARY_PROMPT.format(lines=lines)), SUMMARY_TOKENS)
        summary |= {"text": trim_summary(g["caption"], g["truncated"]), "raw": g["caption"], "truncated": g["truncated"],
                    "prompt_tokens": g["prompt_tokens"]}
    t_sum = time.perf_counter() - t0

    per = [w["decode_s"] + w["preprocess_s"] + w["generate_s"] for w in ok]
    rec = {
        "video": {"path": str(args.video), **info.__dict__},
        "settings": {"window_s": windows[0][1] - windows[0][0], "windows": len(windows),
                     **{k: cfg[k] for k in ("prompt_id", "prompt", "frames", "image_splitting", "max_new_tokens",
                                            "dtype", "decoding")}},
        "offline_guard": args.offline, "uncertainty": {"cutoff": cutoff["cutoff"], "rule": cutoff["rule"]},
        "windows": recs, "notable": notable(recs), "summary": summary,
        "per_window_s": {"median": statistics.median(per) if per else None,
                         "max": max(per) if per else None,
                         "decode_median": statistics.median(w["decode_s"] for w in ok) if ok else None,
                         "generate_median": statistics.median(w["generate_s"] for w in ok) if ok else None},
        "timing_s": {"model_load": t_load, "captioning": t_cap, "summary": t_sum,
                     "total": time.perf_counter() - t_start},
        "memory_mib": {"torch_peak_allocated": torch.cuda.max_memory_allocated() / 2**20,
                       "torch_peak_reserved": torch.cuda.max_memory_reserved() / 2**20},
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
    }
    (out / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "report.md").write_text(report_md(args.video, info, rec))
    print(f"\nwrote {out / 'report.md'} and {out / 'run.json'} in {rec['timing_s']['total']:.1f} s")


if __name__ == "__main__":
    main()

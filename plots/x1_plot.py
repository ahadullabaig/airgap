"""X1 figure: attention time and memory against sequence length N, one head, D = 64, FP32.

Reads results/attention/v1.json (C++ harness) and results/attention/b1.json (Python harness). Left: median
time per call (CUDA events; whiskers 10th-90th percentile of samples); right: device memory the
operation needs (V1: its setup, which holds the N x N matrix; B1: peak PyTorch allocation during a call).
A cross marks the N at which a version could not run (out of memory). The 12-head numbers are in a note,
not on the single-head curves.

Example:
    .venv/bin/python plots/x1_plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = [  # (file, op-name prefix, label, colour: categorical slots 1-3)
    ("v1.json", "v1_h1_", "V1 (ours, naive, N x N stored)", "#2a78d6"),
    ("b1.json", "b1_math_h1_", "B1 PyTorch math backend", "#eb6834"),
    ("b1.json", "b1_efficient_attention_h1_", "B1 PyTorch memory-efficient backend", "#1baf7a"),
]


def load(name):
    return json.loads((ROOT / "results" / "attention" / name).read_text())


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200, facecolor=SURFACE)
    docs = {f: load(f) for f in {s[0] for s in SERIES}}
    device_mib = docs["v1.json"]["env"]["memory_mib"]
    ns_all = []
    t_max = max(o["clocks"]["event_batch_ms"]["median"] for d in docs.values() for o in d["ops"]
                if o["status"] == "ok" and "_h1_" in o["name"])
    for idx, (fname, prefix, label, color) in enumerate(SERIES):
        ops = [o for o in docs[fname]["ops"] if o["name"].startswith(prefix)]
        ops.sort(key=lambda o: int(o["params"]["n"]))
        ok = [o for o in ops if o["status"] == "ok"]
        failed = [o for o in ops if o["status"] == "failed"]
        ns = [int(o["params"]["n"]) for o in ok]
        ns_all += [int(o["params"]["n"]) for o in ops]
        t = [o["clocks"]["event_batch_ms"]["median"] for o in ok]
        lo = [o["clocks"]["event_batch_ms"]["p10"] for o in ok]
        hi = [o["clocks"]["event_batch_ms"]["p90"] for o in ok]
        mem = [o.get("mem_peak_mib") if o.get("mem_peak_mib") is not None else o["mem_mib"] for o in ok]
        style = dict(color=color, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax_t.plot(ns, t, **style)
        ax_t.vlines(ns, lo, hi, color=color, lw=1.2, alpha=0.6)
        ax_m.plot(ns, [max(m, 0.05) for m in mem], **style)
        for o in failed:  # one row of crosses per series above the data, so they never overlap
            n = int(o["params"]["n"])
            for ax, y in ((ax_t, t_max * 2.2 * 1.45 ** idx), (ax_m, device_mib * 1.5 * 1.45 ** idx)):
                ax.plot([n], [y], marker="x", ms=8, mew=2.5, color=color, ls="none")
        ax_t.annotate(f"{t[-1]:.0f} ms", (ns[-1], t[-1]), xytext=(6, -8), textcoords="offset points", fontsize=7,
                      color=INK_2)

    ax_m.axhline(device_mib, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax_m.annotate(f"device memory {device_mib} MiB", (128, device_mib), xytext=(0, 4), textcoords="offset points",
                  fontsize=7, color=MUTED)
    ticks = sorted(set(ns_all))
    for ax in (ax_t, ax_m):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{n // 1024}k" if n >= 1024 else str(n) for n in ticks], rotation=45, ha="right")
        ax.set_xlabel("Sequence length N (tokens), one head, D = 64")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    ax_t.set_ylabel("Time per call (ms)")
    ax_t.set_title("Time", loc="left", fontsize=10, color=INK)
    ax_m.set_ylabel("Device memory for the operation (MiB)")
    ax_m.set_title("Memory: N x N storage meets the 4 GB device", loc="left", fontsize=10, color=INK)

    v1 = {o["name"]: o for o in docs["v1.json"]["ops"]}["v1_h12_n1024"]
    b1 = {o["name"]: o for o in docs["b1.json"]["ops"]}
    note = (f"Encoder shape (12 heads, N = 1024): V1 {v1['clocks']['event_batch_ms']['median']:.1f} ms "
            f"(predicted 2.0-4.9); B1 math {b1['b1_math_h12_n1024']['clocks']['event_batch_ms']['median']:.2f} ms "
            f"(predicted 0.5-1.8), memory-efficient "
            f"{b1['b1_efficient_attention_h12_n1024']['clocks']['event_batch_ms']['median']:.2f} ms (predicted 0.5-1.2)")
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("X1: attention on an RTX 3050 Laptop GPU (FP32). Crosses above the curves: the version ran out of "
                 "memory at that N.",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    lines = [
        note + ". Each version repeats within 1% in a second process.",
        "Small N: B1's times are mostly Python dispatch (math is flat at 0.14 ms up to N = 512; the memory-efficient "
        "host time is twice its event time at N = 128), so V1 meeting B1 there is not parity.",
        "Memory at small N: V1's 2 MiB is allocation granularity (it needs 0.19 MiB at N = 128); B1 math's 8 MiB is a "
        "cuBLAS workspace allocated on every call.",
        "Out of memory, bracketed by the sizes run: V1 (one N x N matrix, 4N^2 bytes) between N = 24k and 32k, where "
        "3.69 GB free predicts about 31k; B1 math (scores and probabilities, 8N^2 bytes) between 16k and 24k, "
        "predicted about 22k.",
    ]
    for k, line in enumerate(lines):
        fig.text(0.01, 0.012 + 0.03 * (len(lines) - 1 - k), line, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.13, 1, 0.86))
    out = ROOT / "plots" / "x1_attention.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()

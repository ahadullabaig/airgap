"""X1 figure: attention time and memory against sequence length N, one head, D = 64, FP32.

Reads results/attention/v1.json, v2.json and v3.json (C++ harness) and results/attention/b1.json (Python
harness). Left: median
time per call (CUDA events; whiskers 10th-90th percentile of samples); right: device memory the
operation needs (V1: its setup, which holds the N x N matrix; B1: peak PyTorch allocation during a call).
A cross marks the N at which a version could not run (out of memory). The 12-head numbers are in a note,
not on the single-head curves.

Example:
    .venv/bin/python plots/x1_plot.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = [  # (file, op-name prefix, label, short label, colour: categorical slots 1-5, marker)
    ("v1.json", "v1_h1_", "V1 (ours, naive, N x N stored)", "V1", "#2a78d6", "o"),
    ("b1.json", "b1_math_h1_", "B1 PyTorch math backend", "B1 math", "#eb6834", "s"),
    ("b1.json", "b1_efficient_attention_h1_", "B1 PyTorch memory-efficient backend", "B1 mem-eff", "#1baf7a", "^"),
    ("v2.json", "v2_h1_", "V2 (ours, tiled, N x N stored)", "V2", "#f2a600", "D"),
    ("v3.json", "v3_h1_", "V3 (ours, fused, online softmax)", "V3", "#ee7fac", "v"),
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
    ends = []  # (last time, last N, label text, colour) for the end-label column
    for idx, (fname, prefix, label, short, color, marker) in enumerate(SERIES):
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
        style = dict(color=color, lw=2, marker=marker, ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax_t.plot(ns, t, **style)
        ax_t.vlines(ns, lo, hi, color=color, lw=1.2, alpha=0.6)
        ax_m.plot(ns, [max(m, 0.05) for m in mem], **style)
        for o in failed:  # one row of crosses per series above the data, so they never overlap
            n = int(o["params"]["n"])
            for ax, y in ((ax_t, t_max * 2.2 * 1.45 ** idx), (ax_m, device_mib * 1.5 * 1.45 ** idx)):
                ax.plot([n], [y], marker="x", ms=8, mew=2.5, color=color, ls="none")
        ends.append((t[-1], ns[-1], f"{short} {t[-1]:.0f} ms", color))

    # Direct name labels (yellow, aqua and pink sit below 3:1 contrast on the surface, so the name is in text,
    # not only in colour): a column right of the data, spread at least GAP decades apart, each joined to its
    # line's last point by a thin leader in the series colour.
    GAP, X_LAB = 0.38, 2 ** 16.0
    ends.sort()
    ys = [math.log10(e[0]) for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + GAP)
    shift = (sum(math.log10(e[0]) for e in ends) - sum(ys)) / len(ys)  # recentre on the points
    for i in range(len(ys)):
        ys[i] += shift
        if i:
            ys[i] = max(ys[i], ys[i - 1] + GAP)
    for (y, n_end, text, color), yl in zip(ends, ys):
        ax_t.plot([n_end * 1.08, X_LAB * 0.94], [y, 10 ** yl], color=color, lw=0.8)
        ax_t.annotate(text, (X_LAB, 10 ** yl), va="center", fontsize=7, color=INK_2)

    ax_m.axhline(device_mib, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax_m.annotate(f"device memory {device_mib} MiB", (128, device_mib), xytext=(0, 4), textcoords="offset points",
                  fontsize=7, color=MUTED)
    ax_m.annotate("V1 and V2 need the same memory\n(one N x N matrix): V1's curve lies\nunder V2's; up to N = 512 "
                  "both lie\nunder V3's (2 MiB)", (128, 1500),
                  fontsize=7, color=INK_2, va="top")
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
    ax_t.set_xlim(None, 2 ** 18.2)
    ax_t.set_ylabel("Time per call (ms)")
    ax_t.set_title("Time", loc="left", fontsize=10, color=INK)
    ax_m.set_ylabel("Device memory for the operation (MiB)")
    ax_m.set_title("Memory: N x N storage meets the 4 GB device", loc="left", fontsize=10, color=INK)

    v1 = {o["name"]: o for o in docs["v1.json"]["ops"]}["v1_h12_n1024"]
    v2 = {o["name"]: o for o in docs["v2.json"]["ops"]}["v2_h12_n1024"]
    v3 = {o["name"]: o for o in docs["v3.json"]["ops"]}["v3_h12_n1024"]
    # V3's prediction rescaled to the clock it ran at (bench/predict_times.py inputs; peak measured near 1.9 GHz)
    pk = json.loads((ROOT / "results" / "cuda" / "peak.json").read_text())["fma_sustained"]
    v3_mhz = [g["median_sm_mhz"] for g in docs["v3.json"]["groups"] if g["group"] == "h12_n1024"][0]
    peak_mhz = sum(pk["sm_mhz_per_second"]) / len(pk["sm_mhz_per_second"])
    flop = 4 * 1024 * 1024 * 64 * 12
    p_c = pk["tflops_settled"] * 1e12 * v3_mhz / peak_mhz
    v3_lo, v3_hi = flop / (0.65 * p_c) * 1e3, flop / (0.35 * p_c) * 1e3
    b1 = {o["name"]: o for o in docs["b1.json"]["ops"]}
    note = (f"Encoder shape (12 heads, N = 1024): V1 {v1['clocks']['event_batch_ms']['median']:.1f} ms "
            f"(predicted 2.0-4.9); V2 {v2['clocks']['event_batch_ms']['median']:.2f} ms (predicted 1.2-1.6); "
            f"V3 {v3['clocks']['event_batch_ms']['median']:.2f} ms at {v3_mhz / 1000:.2f} GHz (predicted 0.67-1.24; "
            f"{v3_lo:.2f}-{v3_hi:.2f} at that clock: a miss)")
    b1_note = (f"B1 at the encoder shape: math {b1['b1_math_h12_n1024']['clocks']['event_batch_ms']['median']:.2f} ms "
            f"(predicted 0.5-1.8), memory-efficient "
            f"{b1['b1_efficient_attention_h12_n1024']['clocks']['event_batch_ms']['median']:.2f} ms (predicted 0.5-1.2)")
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("X1: attention on an RTX 3050 Laptop GPU (FP32). Crosses above the curves: the version ran out of "
                 "memory at that N.",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    rep = []  # largest median difference between the run plotted and a repeat in a second process
    for name, label in (("v1", "V1"), ("v2", "V2"), ("v3", "V3"), ("b1", "B1")):
        a = {o["name"]: o for o in load(f"{name}.json")["ops"] if o["status"] == "ok"}
        b = {o["name"]: o for o in load(f"{name}_repeat.json")["ops"] if o["status"] == "ok"}
        worst = max(abs(b[k]["clocks"]["event_batch_ms"]["median"] / a[k]["clocks"]["event_batch_ms"]["median"] - 1)
                    for k in a if k in b)
        rep.append(f"{label} {100 * worst:.1f}%")
    v2ops = {o["name"]: o for o in docs["v2.json"]["ops"]}
    ph = lambda name, phase: v2ops[name]["clocks"][f"phase_{phase}_ms"]["median"]  # noqa: E731
    v2_note = (f"V2 at one head: above N = 2048 the softmax switches to its two-read path (from 2048 to 4096 its time "
               f"grows {ph('v2_h1_n4096', 'softmax') / ph('v2_h1_n2048', 'softmax'):.1f}x, not 4x); the output "
               f"matmul launches N / 64 blocks for 16 SMs, so at N = 1024 it takes "
               f"{ph('v2_h1_n1024', 'output') / ph('v2_h1_n1024', 'scores'):.1f}x the scores time (at 12 heads "
               f"{ph('v2_h12_n1024', 'output') / ph('v2_h12_n1024', 'scores'):.2f}x).")
    v3ops = {o["name"]: o for o in docs["v3.json"]["ops"]}
    r32 = lambda n: (v3ops[f"v3_h1_n{n}"]["clocks"]["event_batch_ms"]["median"]  # noqa: E731
                     / v2ops[f"v2_h1_n{n}"]["clocks"]["event_batch_ms"]["median"])
    small = [r32(n) for n in (128, 256, 512)]
    v3_note = (f"V3 at one head: its 128-row tile (tuned at 12 heads) launches N / 128 blocks for 16 SMs, so it takes "
               f"{min(small):.1f}-{max(small):.1f}x V2's time at N = 128-512, {r32(1024):.2f}x at 1024 and "
               f"{r32(2048):.2f}x at 2048.")
    lines = [
        note + ".",
        b1_note + ".",
        v2_note,
        v3_note,
        "Largest change of a median in a repeat run (second process): " + ", ".join(rep) + ".",
        "Small N: B1's times are mostly Python dispatch (math is flat at 0.14 ms up to N = 512; the memory-efficient "
        "host time is twice its event time at N = 128), so V1 meeting B1 there is not parity.",
        "Memory at small N: our versions' 2 MiB is allocation granularity (it needs 0.19 MiB at N = 128); B1 math's 8 MiB is a "
        "cuBLAS workspace allocated on every call.",
        "V3 stores nothing of size N x N and runs at every N (its memory is Q, K, V and O).",
        "Out of memory, bracketed by the sizes run: V1 and V2 (one N x N matrix, 4N^2 bytes) between N = 24k and 32k, where "
        "3.69 GB free predicts about 31k; B1 math (scores and probabilities, 8N^2 bytes) between 16k and 24k, "
        "predicted about 22k.",
    ]
    for k, line in enumerate(lines):
        fig.text(0.01, 0.012 + 0.03 * (len(lines) - 1 - k), line, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.28, 1, 0.86))
    out = ROOT / "plots" / "x1_attention.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Caption-level gating check figure from results/gate_captions/scores.json (eval/gate_captions.py).

Left: per-clip CIDEr-D against the CapERA references minus dense's (mean over 100 clips, whiskers the 95%
bootstrap interval) against the connector error at the four caption frames, one point per condition. The
dashed line is the scale of a large effect: dense captions scored against another clip's references.
Right: the same difference against cost, for the conditions that have one.

Example:
    .venv/bin/python plots/gate_captions_plot.py
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
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, as in the token-trace figures
FAMILIES = [  # (prefix, legend label, colour, marker)
    ("floor", "Floors: E1's bf16 path, ±1 grey level noise", INK_2, "D"),
    ("hold", "Dense at a lower rate, output held", MUTED, "o"),
    ("A", "Gate at 24 fps, stale input (floor)", BLUE, "o"),
    ("B", "Gate at 24 fps, stale rows", ORANGE, "o"),
    ("C", "Gate at 24 fps, token deltas + exact attention", AQUA, "o"),
]
LABEL = {"e1_bf16": "bf16", "noise_pm1": "±1 noise", "hold_2": "12 fps", "hold_4": "6 fps", "hold_8": "3 fps",
         "hold_24": "1 fps", "A_2": "2", "B_2": "2", "A_8": "8", "B_8": "8", "C_0.05": "0.05", "C_0.2": "0.2"}


def fam(k: str) -> str:
    return "floor" if k in ("e1_bf16", "noise_pm1") else k.split("_")[0]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def main() -> None:
    s = json.loads((ROOT / "results" / "gate_captions" / "scores.json").read_text())
    C = s["conditions"]
    shuffled = s["post_hoc"]["shuffled_control"]["cider_minus_dense"]["mean"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200, facecolor=SURFACE, sharey=True)
    for prefix, label, color, marker in FAMILIES:
        ks = [k for k in C if fam(k) == prefix]
        for ax, xkey in ((ax_e, "con_err_mean"), (ax_c, "cost_mean")):
            pts = [(C[k][xkey] * (100 if xkey == "con_err_mean" else 1), C[k]["cider_minus_dense"], k) for k in ks
                   if C[k][xkey] is not None]
            if not pts:
                continue
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1]["mean"] for p in pts]
            ax.vlines(xs, [p[1]["lo95"] for p in pts], [p[1]["hi95"] for p in pts], color=color, lw=1.4, alpha=0.7)
            ax.plot(xs, ys, color=color, lw=1.5 if prefix == "hold" else 0, ls=(0, (4, 3)), marker=marker, ms=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label if ax is ax_e else None)
            for x, y, k in zip(xs, ys, [p[2] for p in pts]):
                below = prefix == "hold" or k == "e1_bf16"  # keeps these clear of the gate labels nearby
                left = ax is ax_e and k == "hold_8"  # next to stale rows at threshold 8
                ax.annotate(LABEL[k], (x, y), xytext=(-6, 4) if left else (5, -11 if below else 3),
                            textcoords="offset points", ha="right" if left else "left", fontsize=7, color=INK_2)
    for ax in (ax_e, ax_c):
        style(ax)
        ax.axhline(0, color=AXIS, lw=1)
        ax.axhline(shuffled, color=MUTED, lw=1, ls=(0, (2, 2)))
    ax_e.annotate(f"captions scored against another clip's references ({shuffled:+.2f})", (0.5, shuffled),
                  xytext=(0, 4), textcoords="offset points", fontsize=7, color=MUTED)
    ax_e.set_xlabel("Connector error at the four caption frames (%)")
    ax_e.set_ylabel("CIDEr-D minus dense's (mean of 100 clips)")
    ax_e.set_title("Caption quality against connector error", loc="left", fontsize=10, color=INK)
    ax_e.set_xlim(0, None)
    ax_c.set_xscale("log")
    xt = [1, 3, 6, 12, 24]
    ax_c.xaxis.set_major_locator(FixedLocator(xt))
    ax_c.xaxis.set_minor_locator(NullLocator())
    ax_c.set_xticklabels([f"{x:g}" for x in xt])
    ax_c.set_xlim(0.8, 30)
    ax_c.set_xlabel("Cost (full-frame equivalents per second; held output: its fps)")
    ax_c.set_title("Caption quality against cost", loc="left", fontsize=10, color=INK)
    handles, labels = ax_e.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Caption-level gating check, {s['clips']} CapERA test clips, E1's frozen captioner. Whiskers: "
                 "95% bootstrap interval of the per-clip difference. Numbers on gates: pixel or token threshold.",
                 x=0.01, ha="left", fontsize=8.5, color=INK_2)
    fig.text(0.01, 0.015, "Stale input's cost is unachievable; token-delta cost is a FLOP estimate. Intervals are "
             "unadjusted for 12 comparisons. results/gate_captions/scores.json", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    out = ROOT / "plots" / "gate_captions.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()

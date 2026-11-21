"""Figure for X0 on ERA from results/x0/era_tra/summary.json.

Panels 1-2: share of blocks the gate refreshes per sample, vs sampling rate, for
camera-still (hover), moving, and untrackable clips, at two thresholds. Points
are the mean over clips; whiskers span the 10th to 90th percentile of clips.
Panel 3: cost per second of video at the first threshold, against dense; the
bar spans the two short-clip conventions (first sample excluded / counted as a
full frame), which bracket a long stream.

Example:
    .venv/bin/python plots/x0_era_plot.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill), light mode. Categorical slots 1-3 validate all-pairs.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GROUPS = [("hover", "Camera still", "#2a78d6"),
          ("moving", "Camera moving", "#eb6834"),
          ("unreliable", "Untrackable (water, smoke, fast)", "#1baf7a")]
SHORT = {"hover": "still", "moving": "moving", "unreliable": "untrackable"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(ROOT / "results" / "x0" / "era_tra" / "summary.json"))
    ap.add_argument("--thresholds", type=float, nargs=2, default=[8, 16])
    ap.add_argument("--out", default=str(ROOT / "plots" / "x0_era_tra.png"))
    args = ap.parse_args()
    r = json.loads(Path(args.summary).read_text())
    sl = r["slices"]
    strides = sorted((int(s) for s in r["fps"]), key=lambda s: r["fps"][str(s)])
    fps = [r["fps"][str(s)] for s in strides]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=200, facecolor=SURFACE)
    offsets = {"hover": 0.96, "moving": 1.0, "unreliable": 1.04}  # spread whiskers apart on the log axis

    for ax, thr in zip(axes[:2], args.thresholds):
        for key, label, color in GROUPS:
            cells = [sl[key][f"{s}/{thr:g}"] for s in strides]
            xs = [f * offsets[key] for f in fps]
            ys = [100 * c["steady_mean"] for c in cells]
            ax.vlines(xs, [100 * c["steady_p10"] for c in cells], [100 * c["steady_p90"] for c in cells],
                      color=color, lw=1.2, alpha=0.55)
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    solid_capstyle="round", label=f"{label} ({cells[0]['clips']} clips)")
            ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Blocks refreshed per sample (%)")
        ax.set_title(f"Refresh when block MAD > {thr:g}", loc="left", fontsize=10, color=INK)

    ax = axes[2]
    thr = args.thresholds[0]
    for key, label, color in GROUPS:
        cells = [sl[key][f"{s}/{thr:g}"] for s in strides]
        xs = [f * offsets[key] for f in fps]
        lo = [c["cost_steady"] for c in cells]
        hi = [c["cost_with_first"] for c in cells]
        ax.vlines(xs, lo, hi, color=color, lw=3, alpha=0.6)
        ax.plot(xs, lo, color=color, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=2)
        dy = {"hover": 0, "moving": -6, "unreliable": 6}[key]  # moving and untrackable end close together
        ax.annotate(f"{SHORT[key]} {lo[-1]:.1f}", (xs[-1], lo[-1]), xytext=(9, dy), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.plot(fps, fps, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.annotate("dense 24", (fps[-1], fps[-1]), xytext=(9, 0), textcoords="offset points", va="center",
                fontsize=8, color=MUTED)
    ax.set_yscale("log")
    yt = [0.3, 1, 3, 10, 24]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.set_ylim(0.2, 40)
    ax.set_ylabel("Full-frame equivalents per second of video")
    ax.set_title(f"Cost per second of video, MAD > {thr:g}", loc="left", fontsize=10, color=INK)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(fps))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{f:g}" for f in fps])
        ax.set_xlim(fps[0] / 1.5, fps[-1] * (3.2 if ax is axes[2] else 2.0))
        ax.set_xlabel("Sampling rate (frames per second)")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("X0 on ERA (training split): how much a reference-mode gate refreshes, pixel proxy. "
                 "Not yet linked to encoder error.", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.text(0.01, 0.005, "Points: mean over clips (first sample excluded). Whiskers: 10th-90th percentile of clips. "
             "Cost bars: first sample excluded (bottom) to counted as a full frame (top). "
             "Repeated frames lower 24 fps by about 2 points for moving clips.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Figure for X0 from results/x0/<clip>/summary.json.

Left: share of blocks a gate refreshes per sample. Right: what that costs per
second of video, against running every block on every sample (dense).
Reference mode, one shot label (default "moving"), one line per threshold.

Example:
    .venv/bin/python plots/x0_plot.py sheboygan_falls
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill), light mode. Ordinal blue ramp validated with --ordinal.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # steps 250, 400, 550, 700


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--label", default="moving", help="shot label to plot")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--out", help="output PNG (default plots/x0_<clip>.png)")
    args = ap.parse_args()
    if len(args.thresholds) > len(RAMP):
        raise SystemExit(f"at most {len(RAMP)} thresholds (one ramp step each)")

    r = json.loads((ROOT / "results" / "x0" / args.clip / "summary.json").read_text())
    strides = sorted(r["ref"], key=lambda s: -r["ref"][s]["fps"])[::-1]  # ascending fps
    fps = [r["ref"][s]["fps"] for s in strides]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK})
    fig, (ax_k, ax_c) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, facecolor=SURFACE)

    n_by_fps = {}
    for color, thr in zip(RAMP, args.thresholds):
        keep, cost = [], []
        for s in strides:
            e = r["ref"][s]["by_threshold"][f"{thr:g}"]["by_label"][args.label]
            keep.append(100 * e["mean"])
            cost.append(e["frame_equiv_per_s"])
            n_by_fps[r["ref"][s]["fps"]] = e["n"]
        style = dict(color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", marker="o", ms=8,
                     markeredgecolor=SURFACE, markeredgewidth=2, label=f"MAD > {thr:g}")
        ax_k.plot(fps, keep, **style)
        ax_c.plot(fps, cost, **style)
        for ax, ys in ((ax_k, keep), (ax_c, cost)):
            ax.annotate(f"> {thr:g}", (fps[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)

    dense = list(fps)
    ax_c.plot(fps, dense, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax_c.annotate("dense: every block,\nevery sample", (fps[-1], dense[-1]), xytext=(8, 0),
                  textcoords="offset points", va="center", fontsize=8, color=MUTED)

    for ax in (ax_k, ax_c):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(fps))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{f:.0f}" for f in fps])
        ax.set_xlim(fps[0] / 1.3, fps[-1] * 2.2)
        ax.set_xlabel("Sampling rate (frames per second)")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    ax_k.set_ylim(0, 100)
    ax_k.set_ylabel("Blocks refreshed per sample (%)")
    ax_k.set_title("Blocks over threshold per sample (pixel proxy for gating)", loc="left", fontsize=10, color=INK)

    ax_c.set_yscale("log")
    ticks = [0.5, 1, 2, 5, 10, 20, 30]
    ax_c.yaxis.set_major_locator(FixedLocator(ticks))
    ax_c.yaxis.set_minor_locator(NullLocator())
    ax_c.set_yticklabels([f"{t:g}" for t in ticks])
    ax_c.set_ylim(0.3, 40)
    ax_c.set_ylabel("Full-frame equivalents per second of video")
    ax_c.set_title("Cost per second of video, vs dense at the same rate", loc="left", fontsize=10, color=INK)

    handles, labels = ax_k.get_legend_handles_labels()
    leg = fig.legend(handles, labels, title="Refresh a block when its mean absolute difference (0-255 scale) is",
                     loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=len(labels), frameon=False,
                     fontsize=8, title_fontsize=8, alignment="left")
    leg.get_title().set_color(INK_2)

    n_txt = ", ".join(f"{n_by_fps[f]} at {f:.0f} fps" for f in fps)
    fig.suptitle(f"X0 pipeline check, {args.clip}: '{args.label}' shots only. One edited clip, no hovering footage; "
                 "not gate evidence.", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.text(0.01, 0.005, f"Reference mode: each block is compared with the version it was last refreshed from. "
             f"Samples: {n_txt}.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.84))
    out = Path(args.out) if args.out else ROOT / "plots" / f"x0_{args.clip}.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Figures for the ERA token traces from results/tokens/era_tra/summary.json.

tokens_era_error.png: error of the connector output (what the language model
sees, at 24 fps with each method holding its latest output) against cost, per
footage group. Dense at a lower frame rate with the output held is the
equal-cost baseline; the gate is shown for stale input (tier A, a floor) and
stale rows (tier B), at 24 and 6 fps, one point per pixel threshold.
Horizontal lines: bf16-vs-fp32 and +-1 grey-level noise.

tokens_era_spread.png: per-layer change between consecutive frames for tokens
inside and outside the pixel mask (block MAD > 8), at 24 fps.

Example:
    .venv/bin/python plots/tokens_era_plot.py
"""

import argparse
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
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
TITLES = {"hover": "Camera still", "moving": "Camera moving", "unreliable": "Untrackable",
          "eye_still_rejected": "Still by eye, rejected by rule"}


def style(ax, xlog=True):
    ax.set_facecolor(SURFACE)
    if xlog:
        ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def error_figure(r: dict, out: Path) -> None:
    groups = [g for g in ("hover", "moving", "unreliable") if g in r["groups"]]
    gates = r["gate_strides"]
    thr = r["thresholds_mad"]
    floors = r["floors"]
    checks = json.loads((ROOT / "results" / "tokens" / "gate_checks.json").read_text())
    noise = sum(v["noise_pm1"]["connector"]["mean"] for v in checks["sensitivity"].values()) / len(checks["sensitivity"])
    bf16 = floors["bf16_vs_fp32"]["connector"]["mean"]

    fig, axes = plt.subplots(1, len(groups), figsize=(max(12.0, 4.6 * len(groups)), 4.8), dpi=200,
                             facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        G = r["groups"][g]
        hold = sorted(((v["cost"], 100 * v["stream"]["mean"]) for v in G["hold"].values() if v["cost"] < 24))
        ax.plot([c for c, _ in hold], [e for _, e in hold], color=MUTED, lw=1.5, ls=(0, (4, 3)), marker="o", ms=5,
                label="Dense at a lower rate, output held")
        for c, e in hold:
            ax.annotate(f"{c:g} fps", (c, e), xytext=(-5, 4), textcoords="offset points", ha="right", fontsize=7,
                        color=MUTED)
        for tier, color in (("A", BLUE), ("B", ORANGE)):
            for s, ls, mk in zip(gates, ("-", (0, (2, 2))), ("o", "s")):
                cells = [G["cells"][f"{s}/{t:g}"] for t in thr]
                xs = [c["cost"] for c in cells]
                ys = [100 * c[f"{tier}_stream"]["mean"] for c in cells]
                name = "stale input (floor)" if tier == "A" else "stale rows"
                ax.plot(xs, ys, color=color, lw=2, ls=ls, marker=mk, ms=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=f"Gate at {24 / s:g} fps, {name}")
                if tier == "A" and s == gates[0]:
                    for t, x, y in zip(thr, xs, ys):
                        ax.annotate(f"{t:g}", (x, y), xytext=(4, -9), textcoords="offset points", fontsize=7,
                                    color=INK_2)
        for y, lab, x, dy in ((100 * bf16, "bf16 vs fp32", 28, 3), (100 * noise, "±1 grey level noise", 0.22, 3)):
            ax.axhline(y, color=AXIS, lw=1)
            ax.annotate(lab, (x, y), xytext=(0, dy), textcoords="offset points", fontsize=7, color=MUTED,
                        ha="right" if x > 1 else "left")
        style(ax)
        xt = [0.3, 1, 3, 6, 12, 24]
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{x:g}" for x in xt])
        ax.set_xlim(0.2, 30)
        ax.set_xlabel("Cost (full-frame equivalents per second)")
        ax.set_title(f"{TITLES[g]} ({len(G['clips'])} clips)", loc="left", fontsize=10, color=INK)
    axes[0][0].set_ylabel("Connector output error at 24 fps (%)")
    axes[0][0].set_ylim(0, None)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Gate error against cost on ERA, encoder + connector in fp32. Numbers on the blue line: "
                 "pixel threshold (block MAD).", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def spread_figure(r: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200, facecolor=SURFACE)
    for g, color in (("hover", BLUE), ("moving", ORANGE), ("unreliable", AQUA)):
        if g not in r["groups"]:
            continue
        t = r["groups"][g]["trace"]["1"]["layers"]
        layers = sorted(int(k) for k in t)
        for part, ls in (("inside", "-"), ("outside", (0, (3, 2)))):
            ys = [100 * t[str(i)][f"{part}_mean"] if t[str(i)][f"{part}_mean"] is not None else float("nan")
                  for i in layers]
            ax.plot(layers, ys, color=color, lw=2, ls=ls, marker="o", ms=4,
                    label=f"{TITLES[g]}, tokens {part} the pixel mask")
    style(ax, xlog=False)
    ax.set_xticks(range(14))
    ax.set_xticklabels(["emb"] + [str(i) for i in range(1, 13)] + ["post-LN"], fontsize=8)
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("Change from previous frame (%)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    ax.set_title("How far change spreads past the pixel mask (24 fps, mask = block MAD > 8)", loc="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(ROOT / "results" / "tokens" / "era_tra" / "summary.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "plots"))
    args = ap.parse_args()
    r = json.loads(Path(args.summary).read_text())
    out = Path(args.out_dir)
    error_figure(r, out / "tokens_era_error.png")
    spread_figure(r, out / "tokens_era_spread.png")


if __name__ == "__main__":
    main()

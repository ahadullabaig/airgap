"""Figures for the ERA token traces from results/tokens/era_tra/summary.json.

tokens_era_error.png: error of the connector output (what the language model
sees, at 24 fps with each method holding its latest output) against cost, per
footage group. Dense at a lower frame rate with the output held is the
equal-cost baseline (drawn as the chord between the strides run, linear in cost); the gate is shown for stale input (tier A, a floor) and
stale rows (tier B), at 24 and 6 fps, one point per pixel threshold.
Tier C (token-delta gates with exact attention) is plotted at its estimated FLOP
cost, one point per token threshold. Horizontal lines: bf16-vs-fp32 and +-1
grey-level noise.

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
    noise = r["floors_per_clip"]["noise_pm1"]["mean"]["mean"]
    bf16 = r["floors_per_clip"]["bf16_vs_fp32"]["mean"]["mean"]

    fig, axes = plt.subplots(1, len(groups), figsize=(max(12.0, 4.6 * len(groups)), 4.8), dpi=200,
                             facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        G = r["groups"][g]
        hold = sorted(((v["cost"], 100 * v["stream"]["mean"]) for v in G["hold"].values()))
        first_cost = G["hold"]["first"]["cost"]
        # the chord between the strides run (linear in cost, so curved on this axis): dropping a fraction of
        # frames or mixing two strides achieves it, and it is what equal_cost in the summary compares against
        cx, cy = [], []
        for (c0, e0), (c1, e1) in zip(hold, hold[1:]):
            for j in range(24):
                cx.append(c0 + (c1 - c0) * j / 24)
                cy.append(e0 + (e1 - e0) * j / 24)
        ax.plot(cx, cy, color=MUTED, lw=1.5, ls=(0, (4, 3)), label="Dense at a lower rate, output held (cost = its fps)")
        ax.plot([c for c, _ in hold if c < 24], [e for c, e in hold if c < 24], color=MUTED, ls="none", marker="o",
                ms=5)
        ax.annotate("first frame only", (first_cost, [e for c, e in hold if c == first_cost][0]), xytext=(-5, 4),
                    textcoords="offset points", ha="right", fontsize=7, color=MUTED)
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
        eps = r.get("tier_c_thresholds", [])
        if "token_cells" in G and eps:
            for s, ls, mk in zip(gates, ("-", (0, (2, 2))), ("o", "s")):
                cells = [G["token_cells"][f"{s}/{e:g}"] for e in eps]
                xs = [c["cost"] for c in cells]
                ys = [100 * c["C_stream"]["mean"] for c in cells]
                ax.plot(xs, ys, color=AQUA, lw=2, ls=ls, marker=mk, ms=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=f"Gate at {24 / s:g} fps, token deltas + exact attention")
                if s == gates[0]:
                    for e, x, y in zip(eps, xs, ys):
                        ax.annotate(f"{e:g}", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7,
                                    color=INK_2)
        for y in (100 * bf16, 100 * noise):
            ax.axhline(y, color=AXIS, lw=1)
        ax.annotate("floors: ±1 grey level noise (upper line), bf16 vs fp32 (lower)", (0.045, 100 * noise),
                    xytext=(0, 3), textcoords="offset points", fontsize=7, color=MUTED)
        style(ax)
        xt = [0.1, 0.3, 1, 3, 6, 12, 24]
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{x:g}" for x in xt])
        ax.set_xlim(0.04, 30)
        ax.set_xlabel("Cost (full-frame equivalents per second)")
        ax.set_title(f"{TITLES[g]} ({len(G['clips'])} clips)", loc="left", fontsize=10, color=INK)
    axes[0][0].set_ylabel("Connector output error at 24 fps (%)")
    axes[0][0].set_ylim(0, None)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.91), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Gate error against cost on ERA, encoder + connector in fp32. Numbers: pixel threshold (blue), "
                 "token-delta threshold (green).\nStale input is plotted at the mask's cost, which no kernel "
                 "achieves; token-delta cost is a FLOP estimate.", x=0.01, ha="left", fontsize=8.5, color=INK_2)
    parts = []
    for t, name in (("B", "stale rows"), ("C", "token deltas + exact attention"),
                    ("A", "stale input (the floor, at a cost no kernel achieves)")):
        lower = sum(v["gate_lower"] for g in groups for v in r["groups"][g]["equal_cost"][t].values() if v)
        n = sum(v["n"] for g in groups for v in r["groups"][g]["equal_cost"][t].values() if v)
        parts.append(f"{name} {lower} of {n}")
    fig.text(0.01, 0.042, "Per clip, against the dashed chord: settings (clip x threshold x gate rate, three groups "
             "above) where the gate's error is lower than dense-and-hold at equal cost:", fontsize=7.5, color=INK_2)
    fig.text(0.01, 0.012, "; ".join(parts) + ". results/tokens/era_tra/summary.json, key equal_cost.",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.80))
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

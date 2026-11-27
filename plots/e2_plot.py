"""E2 confusion matrix from results/e2/scores_test.json (eval/e2.py).

Rows: the true ERA class, columns: the class the model named, both in the order of the list in the prompt (P3),
plus a column for answers that match no class name (no answer named two classes, so that column is left out).
Cells are the share of the true class's clips, so classes of different sizes read alike; counts are printed only
where the share is at least 15%. The diagonal (correct answers) is outlined. The strip above counts how often
each name was given in all.

Example:
    .venv/bin/python plots/e2_plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAR = "#e1e0d9", "#c3c2b7", "#b9b8b0"
# sequential blue, steps 100-700 (the palette's single-hue ramp); zero cells stay the surface colour
RAMP = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
LABEL_AT = 0.15


def luminance(hex_or_rgb) -> float:
    r, g, b = to_rgb(hex_or_rgb)
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def main() -> None:
    s = json.loads((ROOT / "results" / "e2" / "scores_test.json").read_text())
    conf = s["confusion"]
    names = s["config"]["labels"]
    rows = conf["rows"]
    keep = [k for k, c in enumerate(conf["columns"]) if c != "ambiguous"]
    assert all(conf["counts"][r][conf["columns"].index("ambiguous")] == 0 for r in range(len(rows)))
    counts = np.array(conf["counts"])[:, keep]
    cols = [conf["columns"][k] for k in keep]
    col_names = [names.get(c, "matches no name") for c in cols]
    support = counts.sum(axis=1)
    share = counts / support[:, None]
    given = counts.sum(axis=0)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig = plt.figure(figsize=(10, 11.2), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 10], width_ratios=[40, 1], hspace=0.04, wspace=0.03)
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    cax = fig.add_subplot(gs[1, 1])

    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    cmap.set_bad(SURFACE)
    shown = np.ma.masked_where(counts == 0, share)
    ax.pcolormesh(np.arange(len(cols) + 1), np.arange(len(rows) + 1), shown, cmap=cmap, vmin=0, vmax=1,
                  edgecolors=SURFACE, linewidth=1.0)
    ax.set_facecolor(SURFACE)
    ax.invert_yaxis()
    for i in range(len(rows)):  # correct answers
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor=INK, lw=0.9))
    for i in range(len(rows)):
        for j in range(len(cols)):
            if share[i, j] >= LABEL_AT:
                dark = luminance(cmap(share[i, j])) < 0.3
                ax.text(j + 0.5, i + 0.5, str(counts[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if dark else INK)
    ax.axvline(len(cols) - 1, color=INK_2, lw=0.6)  # the no-match column is not a class
    ax.set_xticks(np.arange(len(cols)) + 0.5, col_names, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(rows)) + 0.5, [f"{names[r]} ({n})" for r, n in zip(rows, support)])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("The model's answer", labelpad=6)
    ax.set_ylabel("True class (test clips)", labelpad=6)

    top.bar(np.arange(len(cols)) + 0.5, given, width=0.72, color=BAR)
    for j, g in enumerate(given):
        if g >= 100:
            top.text(j + 0.5, g + 8, str(g), ha="center", va="bottom", fontsize=7, color=INK_2)
    top.set_ylim(0, given.max() * 1.25)
    top.set_ylabel("Times\ngiven", rotation=0, ha="right", va="center", labelpad=4)
    top.set_yticks([])
    top.tick_params(axis="x", length=0, labelbottom=False)
    for side in ("top", "right", "left"):
        top.spines[side].set_visible(False)
    top.set_facecolor(SURFACE)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "25%", "50%", "75%", "100%"])
    cb.set_label("Share of the true class's clips", color=INK_2)

    st = s["strict"]
    top.text(0, 1.62, "E2: which event the model names for each true class", transform=top.transAxes,
             ha="left", va="bottom", fontsize=12, fontweight="bold")
    top.text(0, 1.18,
             f"CapERA test split, {s['clips']:,} clips, zero-shot, prompt P3. Accuracy {st['accuracy']:.3f}, "
             f"macro-F1 {st['macro_f1']:.3f}.\nOutlined cells are correct answers; numbers are clip counts where "
             f"the share is at least {LABEL_AT:.0%}.",
             transform=top.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK_2, linespacing=1.4)
    out = ROOT / "plots" / "e2_confusion.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    print(out)


if __name__ == "__main__":
    main()

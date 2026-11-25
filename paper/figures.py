"""Paper figures at IEEE widths (3.5 in for one column, 7.16 in for two), as vector PDFs in paper/figures/, from
the committed result files. Explanatory notes live in the captions in main.tex, not in the figures. Do not edit
paper/figures/ by hand.

  fig_x1.pdf        time and device memory against N, one head (X1)
  fig_x2.pdf        where the time goes at the pinned clock, and what each kernel's warps wait on (X2)
  fig_tokens.pdf    gate error against cost on ERA, per footage group (token traces)
  fig_captions.pdf  caption quality against connector error (caption-level check)

Example:
    .venv/bin/python paper/figures.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
OUT = HERE / "figures"
sys.path.insert(0, str(HERE))
from derived import pinned_breakdown  # noqa: E402

ONE, TWO = 3.5, 7.16  # column widths, inches
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#f2a600", "#ee7fac"]  # categorical slots 1-5, as in plots/

# Liberation Serif: TrueType with Times metrics, so the figures match the text and embed as TrueType (Type 42)
# without the CFF mismatch OpenType Times clones cause, which PDF checkers can flag.
FONT_DIR = Path("/usr/share/fonts/liberation-serif-fonts")
for f in ("LiberationSerif-Regular.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-Bold.ttf"):
    if (FONT_DIR / f).exists():
        font_manager.fontManager.addfont(str(FONT_DIR / f))
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(rel: str) -> dict:
    return json.loads((RES / rel).read_text())


def style(ax, grid_axis="both") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, width=0.5)


def med(o: dict, k: str = "event_batch_ms") -> float:
    return o["clocks"][k]["median"]


# ---------------------------------------------------------------- X1
def fig_x1() -> None:
    series = [("v1.json", "v1_h1_", "V1 naive", SLOTS[0], "o"), ("v2.json", "v2_h1_", "V2 tiled", SLOTS[3], "D"),
              ("v3.json", "v3_h1_", "V3 fused", SLOTS[4], "v"),
              ("b1.json", "b1_math_h1_", "PyTorch math", SLOTS[1], "s"),
              ("b1.json", "b1_efficient_attention_h1_", "PyTorch mem.-eff.", SLOTS[2], "^")]
    docs = {f: load(f"attention/{f}") for f in {s[0] for s in series}}
    device = docs["v1.json"]["env"]["memory_mib"]
    fig, (at, am) = plt.subplots(1, 2, figsize=(TWO, 2.35))
    ns_all = set()
    for i, (f, prefix, label, c, mk) in enumerate(series):
        ops = sorted((o for o in docs[f]["ops"] if o["name"].startswith(prefix)), key=lambda o: int(o["params"]["n"]))
        ok = [o for o in ops if o["status"] == "ok"]
        ns = [int(o["params"]["n"]) for o in ok]
        ns_all |= {int(o["params"]["n"]) for o in ops}
        t = [med(o) for o in ok]
        mem = [o.get("mem_peak_mib") if o.get("mem_peak_mib") is not None else o["mem_mib"] for o in ok]
        kw = dict(color=c, marker=mk, markeredgecolor="white", markeredgewidth=0.4, label=label)
        at.plot(ns, t, **kw)
        am.plot(ns, [max(m, 0.05) for m in mem], **kw)
        for o in ops:
            if o["status"] == "failed":  # out of memory at this N: a cross above the data, one row per series
                n = int(o["params"]["n"])
                at.plot([n], [3e3 * 1.6 ** i], marker="x", color=c, ms=4, mew=1.2, ls="none")
                am.plot([n], [device * 2.2 * 1.6 ** i], marker="x", color=c, ms=4, mew=1.2, ls="none")
    am.axhline(device, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    am.text(128, device * 1.35, f"device memory ({device} MiB)", fontsize=6.5, color=INK_2)
    ticks = sorted(ns_all)
    for ax, ylab in ((at, "Time per call (ms)"), (am, "Device memory (MiB)")):
        style(ax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{n // 1024}k" if n >= 1024 else str(n) for n in ticks], rotation=45, ha="right")
        ax.set_xlabel("Sequence length $N$ (one head, $d$ = 64)")
        ax.set_ylabel(ylab)
    at.set_title("(a) Time", loc="left")
    am.set_title("(b) Memory", loc="left")
    h, lab = at.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.06), handlelength=1.6,
               columnspacing=1.2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_x1.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- X2
PHASE = {"scores": SLOTS[0], "softmax": SLOTS[1], "output": SLOTS[2], "fused": SLOTS[3]}
STALL_GROUPS = [("global memory", ("long_scoreboard", "lg_throttle"), SLOTS[0]),
                ("MIO (shared mem., shuffles, exp)", ("short_scoreboard", "mio_throttle"), SLOTS[1]),
                ("issue side", ("selected", "not_selected", "dispatch_stall", "math_pipe_throttle"), SLOTS[2]),
                ("fixed-latency dependency", ("wait",), SLOTS[3]),
                ("barrier", ("barrier",), SLOTS[4])]


def _stack(ax, y, v: dict) -> float:
    x = 0.0
    for r in v["phases"]:
        w = r["time_ms"]
        ax.barh(y, w, left=x, height=0.6, color=PHASE[r["phase"]], edgecolor="white", linewidth=0.6)
        f = x + r["floor_ms"]
        ax.plot([f, f], [y - 0.38, y + 0.38], color=INK, lw=1.0)
        x += w
    return x


def fig_x2() -> None:
    d = pinned_breakdown()
    V = d["versions"]
    pin = load("attention/clock1395/summary.json")
    fig = plt.figure(figsize=(TWO, 2.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1, 3.2], hspace=0.65, wspace=0.55)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[:, 1])
    x = _stack(a1, 0, V["V1"])
    a1.text(x, 0, f"  {V['V1']['time_ms']:.1f}", va="center", fontsize=6.5, color=INK_2)
    a1.set_yticks([0], ["V1"])
    a1.set_xlim(0, V["V1"]["time_ms"] * 1.18)
    a1.set_ylim(-0.6, 0.6)
    rows = [("V2", V["V2"]), ("V3", V["V3"])]
    for i, (name, v) in enumerate(rows):
        x = _stack(a2, -i, v)
        a2.text(x, -i, f"  {v['time_ms']:.2f}", va="center", fontsize=6.5, color=INK_2)
    for j, (k, name) in enumerate((("B1 memory-efficient", "PyTorch mem.-eff."), ("B1 math", "PyTorch math"))):
        y = -2 - j
        t = pin["ops"][k]["time_ms"]
        a2.barh(y, t, height=0.6, color=AXIS, edgecolor="white", linewidth=0.6)
        a2.text(t, y, f"  {t:.2f}", va="center", fontsize=6.5, color=INK_2)
    a2.set_yticks([0, -1, -2, -3], ["V2", "V3", "PyTorch mem.-eff.", "PyTorch math"])
    a2.set_xlim(0, 3.7)
    for ax in (a1, a2):
        style(ax, "x")
        ax.tick_params(axis="y", length=0)
    a2.set_xlabel(f"Time per call at {d['pinned_mhz']:.0f} MHz (ms)")
    a1.set_title("(a) Time by phase; black tick: the phase's floor", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE[p]) for p in ("scores", "softmax", "output", "fused")]
    a2.legend(handles, ["scores", "softmax", "output", "fused"], loc="upper center", frameon=False, ncol=4,
              handlelength=1.0, columnspacing=0.8, fontsize=6.5, bbox_to_anchor=(0.45, -0.36))
    kernels = [(f"{ver} {r['phase']}", r) for ver in ("V1", "V2", "V3") for r in V[ver]["phases"]]
    for i, (name, r) in enumerate(kernels):
        y = -i
        x = 0.0
        parts = [sum(r["stalls"].get(k, 0.0) for k in reasons) for _, reasons, _ in STALL_GROUPS]
        parts.append(max(0.0, 1.0 - sum(parts)))
        for share, c in zip(parts, [g[2] for g in STALL_GROUPS] + [AXIS]):
            if share > 0:
                a3.barh(y, 100 * share, left=100 * x, height=0.62, color=c, edgecolor="white", linewidth=0.5)
            x += share
    a3.set_yticks([-i for i in range(len(kernels))], [k for k, _ in kernels])
    a3.set_xlim(0, 100)
    style(a3, "x")
    a3.tick_params(axis="y", length=0)
    a3.set_xlabel("Share of warp stall cycles (%)")
    a3.set_title("(b) What each kernel's warps wait on", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=g[2]) for g in STALL_GROUPS] + [plt.Rectangle((0, 0), 1, 1, color=AXIS)]
    a3.legend(handles, [g[0] for g in STALL_GROUPS] + ["other"], loc="upper center", bbox_to_anchor=(0.45, -0.28),
              ncol=3, frameon=False, handlelength=1.0, columnspacing=0.8, fontsize=6.5)
    fig.savefig(OUT / "fig_x2.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- token traces
def fig_tokens() -> None:
    r = load("tokens/era_tra/summary.json")
    titles = {"hover": "(a) Camera still", "moving": "(b) Camera moving", "unreliable": "(c) Untrackable"}
    fig, axes = plt.subplots(1, 3, figsize=(TWO, 2.15), sharey=True)
    for ax, g in zip(axes, ("hover", "moving", "unreliable")):
        G = r["groups"][g]
        hold = sorted((v["cost"], 100 * v["stream"]["mean"]) for v in G["hold"].values())
        cx, cy = [], []
        for (c0, e0), (c1, e1) in zip(hold, hold[1:]):  # the chord: linear in cost between the strides run
            for j in range(20):
                cx.append(c0 + (c1 - c0) * j / 20)
                cy.append(e0 + (e1 - e0) * j / 20)
        ax.plot(cx, cy, color=INK_2, lw=1.0, ls=(0, (3, 2)), label="dense at a lower rate, held")
        ax.plot([c for c, _ in hold if c < 24], [e for c, e in hold if c < 24], ls="none", marker="o",
                color=INK_2, ms=3)
        for tier, c, name in (("A", SLOTS[0], "stale input (floor)"), ("B", SLOTS[1], "stale rows")):
            for s, ls, mk in ((1, "-", "o"), (4, (0, (2, 1.5)), "s")):
                cells = [G["cells"][f"{s}/{t:g}"] for t in r["thresholds_mad"]]
                ax.plot([c["cost"] for c in cells], [100 * c[f"{tier}_stream"]["mean"] for c in cells], color=c, ls=ls,
                        marker=mk, markeredgecolor="white", markeredgewidth=0.4,
                        label=f"{name}, gate at {24 // s} fps")
        for s, ls, mk in ((1, "-", "o"), (4, (0, (2, 1.5)), "s")):
            cells = [G["token_cells"][f"{s}/{e:g}"] for e in r["tier_c_thresholds"]]
            ax.plot([c["cost"] for c in cells], [100 * c["C_stream"]["mean"] for c in cells], color=SLOTS[2], ls=ls,
                    marker=mk, markeredgecolor="white", markeredgewidth=0.4,
                    label=f"token deltas, gate at {24 // s} fps")
        style(ax)
        ax.set_xscale("log")
        xt = [0.3, 1, 3, 6, 12, 24]
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{x:g}" for x in xt])
        ax.set_xlim(0.12, 30)
        ax.set_title(f"{titles[g]}, {len(G['clips'])} clips", loc="left")
        ax.set_xlabel("Cost (full frames per second)")
    ymax = max(y for ax in axes for line in ax.get_lines() for y in line.get_ydata())
    axes[0].set_ylim(0, ymax * 1.05)  # shared: set once, after every panel is drawn
    axes[0].set_ylabel("Connector error at 24 fps (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.13), handlelength=1.8,
               columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_tokens.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- caption check
def fig_captions() -> None:
    s = load("gate_captions/scores.json")
    C = s["conditions"]
    fam = lambda k: "floor" if k in ("e1_bf16", "noise_pm1") else k.split("_")[0]  # noqa: E731
    groups = [("floor", "floors (bf16, ±1 noise)", INK_2, "D"), ("hold", "dense at a lower rate, held", MUTED, "o"),
              ("A", "stale input (floor)", SLOTS[0], "o"), ("B", "stale rows", SLOTS[1], "o"),
              ("C", "token deltas", SLOTS[2], "o")]
    label = {"hold_2": "12 fps", "hold_4": "6", "hold_8": "3", "hold_24": "1 fps", "B_2": "2", "B_8": "8",
             "C_0.05": "0.05", "C_0.2": "0.2", "A_2": "2", "A_8": "8", "e1_bf16": "", "noise_pm1": ""}
    OFFSET = {"hold_2": (3, -8), "hold_4": (-3, -8), "hold_8": (-3, -8), "hold_24": (-3, -8),  # label placement,
              "A_2": (2.5, -8), "A_8": (-3, 2), "C_0.05": (-2, -10)}                        # checked by eye
    fig, ax = plt.subplots(figsize=(ONE, 2.3))
    for f, name, c, mk in groups:
        ks = sorted((k for k in C if fam(k) == f), key=lambda k: C[k]["con_err_mean"])
        xs = [100 * C[k]["con_err_mean"] for k in ks]
        ys = [C[k]["cider_minus_dense"]["mean"] for k in ks]
        ax.vlines(xs, [C[k]["cider_minus_dense"]["lo95"] for k in ks], [C[k]["cider_minus_dense"]["hi95"] for k in ks],
                  color=c, lw=0.9, alpha=0.75)
        ax.plot(xs, ys, color=c, ls=(0, (3, 2)) if f == "hold" else "none", lw=0.8, marker=mk, ms=3.5,
                markeredgecolor="white", markeredgewidth=0.4, label=name)
        for x, y, k in zip(xs, ys, ks):
            if label[k]:
                dx, dy = OFFSET.get(k, (2.5, 2.5))
                ax.annotate(label[k], (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha="right" if dx < 0 else "left", fontsize=6, color=INK_2)
    sh = s["post_hoc"]["shuffled_control"]["cider_minus_dense"]["mean"]
    ax.axhline(0, color=AXIS, lw=0.7)
    ax.axhline(sh, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.text(0.5, sh + 0.006, "another clip's references", fontsize=6, color=INK_2)
    style(ax)
    ax.set_xlim(0, None)
    ax.set_xlabel("Connector error at the four caption frames (%)")
    ax.set_ylabel("CIDEr-D minus dense")
    ax.legend(loc="lower right", frameon=False, fontsize=6, handlelength=1.4, ncol=1, bbox_to_anchor=(1.0, 0.13))
    fig.tight_layout()
    fig.savefig(OUT / "fig_captions.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig_x1()
    fig_x2()
    fig_tokens()
    fig_captions()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()

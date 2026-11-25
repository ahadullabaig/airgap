"""X2 figures from results/attention/x2_breakdown.json (bench/x2_breakdown.py).

x2_time.png: where each version's time goes at 12 heads, N = 1024 (stacked phases, each with its fixed
launch cost and its floor marked; V1 on its own axis), next to B1 and to the copies an attention call fed
from the host would add.
x2_stalls.png: what each kernel's warps wait on (Nsight Compute stall reasons, share of warp cycles per issued
instruction), grouped by the resource waited on; every reason is in the JSON.

Example:
    .venv/bin/python plots/x2_plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2"]  # categorical 1-7
PHASE_COLOUR = {"scores": SLOTS[0], "softmax": SLOTS[1], "output": SLOTS[2], "fused": SLOTS[3]}
STALL_GROUPS = [  # (legend label, reasons); every other reason is "other"
    ("global memory (long_scoreboard, lg_throttle)", ("long_scoreboard", "lg_throttle")),
    ("MIO: shared memory, warp shuffles, special math such as exp (short_scoreboard, mio_throttle)",
     ("short_scoreboard", "mio_throttle")),
    ("issue side: issuing, ready, dispatch, math pipe busy (selected, not_selected, dispatch_stall, "
     "math_pipe_throttle)", ("selected", "not_selected", "dispatch_stall", "math_pipe_throttle")),
    ("fixed-latency dependency (wait)", ("wait",)),
    ("barrier (__syncthreads)", ("barrier",)),
]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def bar(ax, y, v: dict, label: str) -> None:
    """One version as a horizontal stack of its phases; the fixed cost is the dark start of each phase and the
    floor a black tick inside it."""
    x = 0.0
    gap = v["time_ms"] * 0.004
    for r in v["phases"]:
        w = r["time_ms"]
        ax.barh(y, w - gap, left=x, height=0.55, color=PHASE_COLOUR[r["phase"]], edgecolor="none")
        ax.barh(y, max(r["fixed_ms"], 0), left=x, height=0.55, color=INK, edgecolor="none")
        f = x + r["floor_ms"]
        ax.plot([f, f], [y - 0.36, y + 0.36], color=INK, lw=1.6)
        x += w
    phases = "\n".join(f"{r['phase']} {r['time_ms']:.2f} ({r['time_over_floor']:.1f}x its floor: {r['floor_kind']})"
                        for r in v["phases"])
    ax.annotate(f"{v['time_ms']:.2f} ms at {v['median_sm_mhz'] / 1000:.2f} GHz; fixed cost "
                f"{1e3 * v['fixed_ms']:.0f} us ({100 * v['fixed_share']:.2f}%)\n{phases}", (x, y), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=6.8, color=INK_2, linespacing=1.5)
    ax.annotate(label, (0, y), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5,
                color=INK)


def time_figure(d: dict, out: Path) -> None:
    V = d["versions"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.0), dpi=200, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.3, 2.4, 2]})
    ax1, ax2, ax3 = axes
    bar(ax1, 0, V["V1"], "V1")
    ax1.set_xlim(0, V["V1"]["time_ms"] * 1.75)
    ax1.set_ylim(-1.0, 1.0)
    bar(ax2, 2.3, V["V2"], "V2")
    bar(ax2, 0.8, V["V3"], "V3")
    lim = max(V["V2"]["time_ms"], d["b1_reference"]["math"]["time_ms"]) * 1.9
    for y, (k, name) in zip((-0.3, -1.0), (("efficient_attention", "B1 memory-efficient"), ("math", "B1 math"))):
        t = d["b1_reference"][k]["time_ms"]
        ax2.barh(y, t, height=0.35, color=AXIS, edgecolor="none")
        ax2.annotate(f"{t:.2f} ms (PyTorch, separate process)", (t, y), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=7, color=INK_2)
        ax2.annotate(name, (0, y), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=8,
                     color=INK_2)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(-1.5, 3.4)
    T = d["transfers"]
    rows = [("V3 kernel (for scale)", V["V3"]["time_ms"], PHASE_COLOUR["fused"]),
            ("copy Q, K, V to the GPU (pinned)", T["h2d_qkv_pinned"]["host_ms"], AXIS),
            ("copy O back (pinned)", T["d2h_o_pinned"]["host_ms"], AXIS),
            ("copy Q, K, V to the GPU (pageable)", T["h2d_qkv_pageable"]["host_ms"], AXIS),
            ("copy O back (pageable)", T["d2h_o_pageable"]["host_ms"], AXIS)]
    for i, (name, t, c) in enumerate(rows):
        y = -i
        ax3.barh(y, t, height=0.55, color=c, edgecolor="none")
        ax3.annotate(f"{t:.2f} ms", (t, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7,
                     color=INK_2)
        ax3.annotate(name, (0, y), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=8,
                     color=INK_2 if i else INK)
    ax3.set_xlim(0, lim)
    for ax in axes:
        style(ax)
        ax.set_yticks([])
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Time per call (ms)", fontsize=8)
    fig.suptitle("Where the time goes at the encoder's shape: 12 heads, N = 1024, D = 64, FP32", x=0.01, y=0.985,
                 ha="left", fontsize=10, color=INK)
    ax3.set_title(f"Not part of the stacks: copies an attention call fed from the host would add "
                  f"({T['h2d_qkv_pinned']['mb']:.1f} MB in, {T['d2h_o_pinned']['mb']:.1f} MB out, about "
                  f"{T['h2d_qkv_pinned']['gb_per_s']:.1f} GB/s).\nIn the encoder Q, K and V stay on the GPU.",
                  loc="left", fontsize=8, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLOUR[p]) for p in ("scores", "softmax", "output", "fused")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=INK), plt.Line2D([0], [0], color=INK, lw=1.6)]
    labels = ["scores (Q K^T)", "softmax", "output (P V)", "fused (V3)", "fixed cost (empty kernels, same launches)",
              "floor: max(FLOPs at FMA peak at the version's clock, DRAM bytes at 188 GB/s)"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, frameon=False, fontsize=7)
    h1 = d["h1"]
    c2 = h1["clause2"]
    fig.text(0.01, 0.012,
             f"H1 clause 1 (fixed costs small) {'holds' if h1['clause1']['holds'] else 'fails'}: at most "
             f"{100 * max(h1['clause1']['fixed_share'].values()):.1f}% of any version. Clause 2 (fusion helps mainly "
             f"by removing N x N traffic) {'holds' if c2['holds'] else 'fails'}:\nV3 saves {c2['saving_ms']:.2f} ms over V2 "
             f"({c2['saving_at_v3_clock_ms'][0]:.2f}-{c2['saving_at_v3_clock_ms'][1]:.2f} ms with V2 at V3's clock) of the "
             f"{c2['v2_nxn_dram_floor_ms']:.2f} ms its N x N traffic costs at full bandwidth.\nFixed cost: the per-phase empty "
             f"launches, each timed alone. Floors use the bytes each kernel actually moved.", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0.13, 0.05, 1, 0.86))
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


def stall_figure(d: dict, out: Path) -> None:
    rows = []
    for ver in ("V1", "V2", "V3"):
        for r in d["versions"][ver]["phases"]:
            rows.append((f"{ver} {r['phase']}", r["stalls"], r))
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200, facecolor=SURFACE)
    colours = SLOTS[:len(STALL_GROUPS)] + [AXIS]
    for i, (name, st, r) in enumerate(rows):
        y = -i
        x = 0.0
        shares = st["shares"]
        parts = [sum(shares.get(k, 0.0) for k in reasons) for _, reasons in STALL_GROUPS]
        parts.append(max(0.0, 1.0 - sum(parts)))
        for share, c in zip(parts, colours):
            if share <= 0:
                continue
            ax.barh(y, share * 100 - 0.25, left=x * 100, height=0.6, color=c, edgecolor="none")
            if share >= 0.08:
                ax.annotate(f"{100 * share:.0f}%", (100 * (x + share / 2), y), ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if c in (SLOTS[0], SLOTS[1]) else INK)
            x += share
        ax.annotate(name, (0, y), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=8,
                    color=INK)
        ax.annotate(f"{st['cycles_per_issue']:.1f} cycles/instr, FMA {r['fma_pipe_pct']:.0f}%, warps "
                    f"{r['warps_active_pct']:.0f}%", (100, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK_2)
    style(ax)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of warp cycles per issued instruction (%)", fontsize=8)
    ax.set_title("What each kernel's warps wait on, by resource (Nsight Compute, 12 heads, N = 1024): descriptive",
                 loc="left", fontsize=10, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours]
    fig.legend(handles, [lab for lab, _ in STALL_GROUPS] + ["other reasons"], loc="lower left",
               bbox_to_anchor=(0.01, 0.0), ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0.08, 0.15, 0.86, 1))
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


def main() -> None:
    d = json.loads((ROOT / "results" / "attention" / "x2_breakdown.json").read_text())
    time_figure(d, ROOT / "plots" / "x2_time.png")
    stall_figure(d, ROOT / "plots" / "x2_stalls.png")


if __name__ == "__main__":
    main()

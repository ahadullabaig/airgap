"""Writes paper/generated/numbers.tex (one \\newcommand per number the paper quotes) and the generated tables,
from the committed result files. Never edit generated/ by hand: change the result or this script.

Every macro is listed in SOURCES with the file it comes from, so the README's "where the numbers come from"
table can be checked against it.

Example:
    .venv/bin/python -I paper/numbers.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "generated"


def load(rel: str) -> dict:
    return json.loads((RES / rel).read_text())


def med(op: dict, k: str = "event_batch_ms") -> float:
    return op["clocks"][k]["median"]


def ops(rel: str) -> dict:
    return {o["name"]: o for o in load(rel)["ops"]}


def group_mhz(rel: str, op_name: str) -> float:
    d = load(rel)
    g = {o["name"]: o["group"] for o in d["ops"]}[op_name]
    return [x["median_sm_mhz"] for x in d["groups"] if x["group"] == g][0]


MACROS: dict[str, str] = {}
SOURCES: dict[str, str] = {}


def put(name: str, value, src: str, fmt: str = "{}") -> None:
    assert name.isalpha(), name  # TeX control words: letters only
    if fmt.startswith("{:+"):  # signed: no "-0.000", and a real minus sign in text
        digits = int(fmt.split(".")[1][0])
        MACROS[name] = "\\ensuremath{" + fmt.format(round(value, digits) + 0.0) + "}"
    else:
        MACROS[name] = fmt.format(value)
    SOURCES[name] = src


def main() -> None:
    # measured limits
    pk = load("cuda/peak.json")
    put("PeakTflops", pk["fma_sustained"]["tflops_settled"], "results/cuda/peak.json", "{:.2f}")
    put("PeakGhz", sum(pk["fma_sustained"]["sm_mhz_per_second"]) / len(pk["fma_sustained"]["sm_mhz_per_second"]) / 1e3,
        "results/cuda/peak.json", "{:.1f}")
    put("DramGBs", pk["dram_read"]["gb_per_s"], "results/cuda/peak.json", "{:.0f}")
    put("LaunchUs", pk["launch_b2b"]["ms_median"] * 1e3, "results/cuda/peak.json", "{:.1f}")
    put("LaunchSyncUs", pk["launch_rt"]["ms_median"] * 1e3, "results/cuda/peak.json", "{:.1f}")

    # predictions written before any attention kernel (dense, 12 heads, N = 1024)
    pr = load("cuda/predicted_times.json")["dense_ms"]
    for v, name in (("V1", "Vone"), ("V2", "Vtwo"), ("V3", "Vthree"), ("B1", "Bone")):
        put(f"Pred{name}Lo", pr[v][0], "results/cuda/predicted_times.json", "{:.2f}")
        put(f"Pred{name}Hi", pr[v][1], "results/cuda/predicted_times.json", "{:.2f}")

    # pinned clock (the headline timing set: one clock, one driver)
    pin = load("attention/clock1395/summary.json")
    put("PinMhz", pin["pinned_mhz"], "results/attention/clock1395/summary.json", "{:.0f}")
    names = {"V1": "Vone", "V2": "Vtwo", "V3": "Vthree", "B1 math": "Bmath", "B1 memory-efficient": "Beff"}
    for k, name in names.items():
        r = pin["ops"][k]
        put(f"Pin{name}Ms", r["time_ms"], "results/attention/clock1395/summary.json", "{:.2f}")
        put(f"Pin{name}PredLo", r["predicted_ms_at_pin"][0], "results/attention/clock1395/summary.json", "{:.2f}")
        put(f"Pin{name}PredHi", r["predicted_ms_at_pin"][1], "results/attention/clock1395/summary.json", "{:.2f}")
    c = pin["v3_vs_v2_equal_clock"]
    put("PinSavingMs", c["saving_ms"], "results/attention/clock1395/summary.json", "{:.2f}")
    put("PinSavingPctOfNxN", 100 * c["share_of_nxn_floor"], "results/attention/clock1395/summary.json", "{:.0f}")
    put("PinVthreeOverVtwoPct", 100 * (1 - pin["ops"]["V3"]["time_ms"] / pin["ops"]["V2"]["time_ms"]),
        "results/attention/clock1395/summary.json", "{:.1f}")

    # as deployed (unpinned): each version at its own steady clock under the power cap
    for f, op, name in (("v1.json", "v1_h12_n1024", "Vone"), ("v2.json", "v2_h12_n1024", "Vtwo"),
                        ("v3.json", "v3_h12_n1024", "Vthree"), ("b1.json", "b1_math_h12_n1024", "Bmath"),
                        ("b1.json", "b1_efficient_attention_h12_n1024", "Beff")):
        o = ops(f"attention/{f}")[op]
        put(f"Dep{name}Ms", med(o), f"results/attention/{f}", "{:.2f}")
        put(f"Dep{name}Ghz", group_mhz(f"attention/{f}", op) / 1e3, f"results/attention/{f}", "{:.2f}")

    # X1: scaling and the memory wall (one head)
    v3 = ops("attention/v3.json")
    put("VthreeBigNMs", med(v3["v3_h1_n32768"]), "results/attention/v3.json", "{:.0f}")

    # X2: where the time goes (unpinned, as deployed)
    x2 = load("attention/x2_breakdown.json")
    for v, name in (("V1", "Vone"), ("V2", "Vtwo"), ("V3", "Vthree")):
        r = x2["versions"][v]
        put(f"Fixed{name}Us", 1e3 * r["fixed_ms"], "results/attention/x2_breakdown.json", "{:.0f}")
        put(f"Fixed{name}Pct", 100 * r["fixed_share"], "results/attention/x2_breakdown.json", "{:.2f}")
    h2 = x2["h1"]["clause2"]
    put("NxNFloorMs", h2["v2_nxn_dram_floor_ms"], "results/attention/x2_breakdown.json", "{:.2f}")
    put("NxNMB", h2["v2_nxn_mb"], "results/attention/x2_breakdown.json", "{:.0f}")
    put("DepSavingMs", h2["saving_ms"], "results/attention/x2_breakdown.json", "{:.2f}")
    t = x2["transfers"]
    put("CopyInMs", t["h2d_qkv_pinned"]["host_ms"], "results/attention/x2_breakdown.json", "{:.2f}")
    put("CopyOutMs", t["d2h_o_pinned"]["host_ms"], "results/attention/x2_breakdown.json", "{:.2f}")

    # token traces (ERA training clips; connector output, equal cost against dense at a lower rate, held)
    tk = load("tokens/era_tra/summary.json")["groups"]
    for tier, name in (("A", "TierA"), ("B", "TierB"), ("C", "TierC")):
        lower = sum(v["gate_lower"] for g in ("hover", "moving", "unreliable") for v in tk[g]["equal_cost"][tier].values() if v)
        n = sum(v["n"] for g in ("hover", "moving", "unreliable") for v in tk[g]["equal_cost"][tier].values() if v)
        put(f"{name}Lower", lower, "results/tokens/era_tra/summary.json")
        put(f"{name}Settings", n, "results/tokens/era_tra/summary.json")

    # caption-level check (CapERA test, 100 clips, E1's frozen captioner)
    cs = load("gate_captions/scores.json")
    put("CapClips", cs["clips"], "results/gate_captions/scores.json")
    put("CapDenseCider", cs["conditions"]["dense"]["corpus_vs_refs"]["CIDEr"], "results/gate_captions/scores.json", "{:.3f}")
    sh = cs["post_hoc"]["shuffled_control"]["cider_minus_dense"]
    put("CapShuffledDelta", -sh["mean"], "results/gate_captions/scores.json", "{:.3f}")
    for k, name in (("B_8", "CapBeight"), ("C_0.2", "CapCtwo"), ("hold_24", "CapHoldone")):
        ci = cs["conditions"][k]["cider_minus_dense"]
        put(f"{name}Delta", ci["mean"], "results/gate_captions/scores.json", "{:+.3f}")
        put(f"{name}Lo", ci["lo95"], "results/gate_captions/scores.json", "{:+.3f}")
        put(f"{name}Hi", ci["hi95"], "results/gate_captions/scores.json", "{:+.3f}")
        put(f"{name}Err", 100 * cs["conditions"][k]["con_err_mean"], "results/gate_captions/scores.json", "{:.0f}")
    put("CapBfloatSame", 100 * cs["conditions"]["e1_bf16"]["identical_to_dense"], "results/gate_captions/scores.json", "{:.0f}")

    OUT.mkdir(exist_ok=True)
    lines = ["% Generated by paper/numbers.py from committed result files. Do not edit.\n"]
    for k, v in MACROS.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}  % {SOURCES[k]}\n")
    (OUT / "numbers.tex").write_text("".join(lines))

    # Table: predicted against measured, at the pinned clock (headline) and as deployed
    rows = [("V1 (naive)", "Vone"), ("V2 (tiled)", "Vtwo"), ("V3 (fused)", "Vthree"),
            ("B1 math", "Bmath"), ("B1 mem.-eff.", "Beff")]
    pred_key = {"Vone": "Vone", "Vtwo": "Vtwo", "Vthree": "Vthree", "Bmath": "Bone", "Beff": "Bone"}
    tab = [r"% Generated by paper/numbers.py. Do not edit." "\n",
           r"\begin{tabular}{@{}lccccc@{}}" "\n", r"\toprule" "\n",
           r" & \multicolumn{2}{c}{Predicted (ms)} & Pinned & \multicolumn{2}{c}{As deployed} \\" "\n",
           r"\cmidrule(lr){2-3}\cmidrule(l){5-6}" "\n",
           r"Version & before & at \PinMhz\,MHz & (ms) & (ms) & (GHz) \\" "\n", r"\midrule" "\n"]
    for label, n in rows:
        p = pred_key[n]
        tab.append(f"{label} & \\Pred{p}Lo--\\Pred{p}Hi & \\Pin{n}PredLo--\\Pin{n}PredHi & \\Pin{n}Ms & "
                   f"\\Dep{n}Ms & \\Dep{n}Ghz \\\\\n")
    tab += [r"\bottomrule" "\n", r"\end{tabular}" "\n"]
    (OUT / "table_ladder.tex").write_text("".join(tab))
    print(f"wrote {OUT / 'numbers.tex'} ({len(MACROS)} macros) and {OUT / 'table_ladder.tex'}")


if __name__ == "__main__":
    main()

"""Writes paper/generated/numbers.tex (one \\newcommand per number the paper quotes) and the generated tables,
from the committed result files. Never edit generated/ by hand: change the result or this script.

Every macro is listed in SOURCES with the file it comes from, so the README's "where the numbers come from"
table can be checked against it.

Example:
    .venv/bin/python -I paper/make_numbers.py
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "generated"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from derived import pinned_breakdown  # noqa: E402


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
MACRO_VAL: dict[str, float] = {}
SOURCES: dict[str, str] = {}


def put(name: str, value, src: str, fmt: str = "{}") -> None:
    assert name.isalpha(), name  # TeX control words: letters only
    if isinstance(value, (int, float)):
        MACRO_VAL[name] = value
    if fmt.startswith("{:+"):  # signed: no "-0.000", and a real minus sign in text
        digits = int(fmt.split(".")[1][0])
        MACROS[name] = "\\ensuremath{" + fmt.format(round(value, digits) + 0.0) + "}"
    else:
        MACROS[name] = fmt.format(value).replace(",", "{,}")
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

    # X2 at the pinned clock (paper/derived.py): fixed shares and time over floor per phase
    pb = pinned_breakdown()
    src = "paper/derived.py (clock1395, x2.json, x2_breakdown.json)"
    put("PinFixedMaxPct", 100 * max(v["fixed_share"] for v in pb["versions"].values()), src, "{:.1f}")
    for v, name in (("V1", "Vone"), ("V2", "Vtwo"), ("V3", "Vthree")):
        for r in pb["versions"][v]["phases"]:
            ph = r["phase"].capitalize() if r["phase"] != "fused" else ""
            put(f"Pin{name}{ph}Over", r["over_floor"], src, "{:.1f}")

    # token traces (ERA training clips; connector output, equal cost against dense at a lower rate, held)
    tk = load("tokens/era_tra/summary.json")["groups"]
    for tier, name in (("A", "TierA"), ("B", "TierB"), ("C", "TierC")):
        lower = sum(v["gate_lower"] for g in ("hover", "moving", "unreliable") for v in tk[g]["equal_cost"][tier].values() if v)
        n = sum(v["n"] for g in ("hover", "moving", "unreliable") for v in tk[g]["equal_cost"][tier].values() if v)
        put(f"{name}Lower", lower, "results/tokens/era_tra/summary.json")
        put(f"{name}Settings", n, "results/tokens/era_tra/summary.json")

    # where a caption's time goes in E1's configuration (eval/encoder_share.py)
    es = load("e1/encoder_share.json")
    src = "results/e1/encoder_share.json"
    put("EsClips", es["clips"], src)
    put("EsTotalMs", 1e3 * es["median_total_s"], src, "{:.0f}")
    for k, name in (("decode_s", "Dec"), ("preprocess_s", "Pre"), ("encoder_s", "Enc"), ("prefill_s", "Prefill"),
                    ("decode_lm_s", "Gen")):
        put(f"Es{name}Ms", 1e3 * es["median_s"][k], src, "{:.0f}")
        put(f"Es{name}Pct", 100 * es["median_share_of_total"][k], src, "{:.0f}")
    put("EsTokens", es["median_s"]["new_tokens"], src, "{:.0f}")
    att = load("cuda/predicted_times.json")["work"]["attention_share_of_encoder_layer_flops"]
    put("AttnFlopPct", 100 * att, "results/cuda/predicted_times.json", "{:.0f}")
    put("AttnCaptionPct", 100 * att * es["median_share_of_total"]["encoder_s"],
        "results/cuda/predicted_times.json x results/e1/encoder_share.json (FLOP-share estimate)", "{:.0f}")

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

    # benchmark protocol (as recorded in every result file)
    proto = load("attention/v3.json")["protocol"]
    src = "results/attention/v3.json (protocol)"
    put("ProtoSamples", proto["samples"], src)
    put("ProtoSampleMs", proto["target_sample_ms"], src, "{:.0f}")
    put("ProtoClockTolPct", 100 * proto["clock_tol"], src, "{:.0f}")
    put("ProtoThrottlePct", 100 * proto["throttle_drop"], src, "{:.0f}")

    # correctness suite (kernels/attention/attn_test.cu output)
    tests = (RES / "attention" / "attn_test.txt").read_text().splitlines()
    rows = [ln for ln in tests if "max rel error" in ln]
    src = "results/attention/attn_test.txt"
    put("TestChecks", len(rows), src)
    put("TestCases", len({ln.split()[1] for ln in rows}), src)
    worst = max(float(ln.split("max rel error")[1].split()[0]) for ln in rows)
    put("TestWorstErr", "{:.1f}".format(worst / 10 ** math.floor(math.log10(worst))), src)
    put("TestWorstExp", math.floor(math.log10(worst)), src)
    assert tests[-1].startswith("all checks passed"), tests[-1]

    # tile shapes chosen by the sweeps
    v2op = ops("attention/v2.json")["v2_h12_n1024"]["params"]
    tile, thread = v2op["scores_config"].split("/")
    put("VtwoTile", tile.replace("x", "$\\times$"), "results/attention/v2.json")
    put("VtwoThreadTile", thread.replace("x", "$\\times$"), "results/attention/v2.json")
    v3op = ops("attention/v3.json")["v3_h12_n1024"]["params"]
    br, rest = v3op["config"].split("x")
    bc, nt = rest.split("/")
    put("VthreeRows", br, "results/attention/v3.json")
    put("VthreeKeys", bc, "results/attention/v3.json")
    put("VthreeThreads", nt, "results/attention/v3.json")
    smem = ops("attention/x2.json")["empty_v3"]["params"]["config"].split(", ")[1].split()[0]
    put("VthreeSmemKB", int(smem) / 1e3, "results/attention/x2.json (empty_v3 launch)", "{:.1f}")

    # out of memory: largest N that ran and smallest that failed, one head
    def oom(rel, prefix):
        o = sorted((x for x in load(rel)["ops"] if x["name"].startswith(prefix)), key=lambda x: int(x["params"]["n"]))
        ok = [int(x["params"]["n"]) for x in o if x["status"] == "ok"]
        bad = [int(x["params"]["n"]) for x in o if x["status"] == "failed"]
        return max(ok), min(bad)
    for rel, prefix, name in (("attention/v1.json", "v1_h1_", "Vone"), ("attention/v2.json", "v2_h1_", "Vtwo"),
                              ("attention/b1.json", "b1_math_h1_", "Bmath")):
        lo, hi = oom(rel, prefix)
        put(f"{name}OomLo", f"{lo // 1024}k", f"results/{rel}")
        put(f"{name}OomHi", f"{hi // 1024}k", f"results/{rel}")
    put("SmallMemMiB", ops("attention/v1.json")["v1_h1_n128"]["mem_mib"], "results/attention/v1.json", "{:.0f}")

    # PyTorch's FP32 memory-efficient kernel: tensor cores, FP32-level accuracy
    ncu = {ln.split()[0]: float(ln.split()[-1].replace(",", "")) for ln in
           (RES / "attention" / "b1_efficient_ncu.txt").read_text().splitlines() if ln.strip().startswith("sm__")}
    src = "results/attention/b1_efficient_ncu.txt"
    put("BeffTensorPct", ncu["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"], src, "{:.0f}")
    put("BeffFmaPct", ncu["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"], src, "{:.0f}")
    err = ops("attention/b1.json")["b1_efficient_attention_h12_n1024"]["check"]["max_rel_err"]
    put("BeffErr", "{:.1f}".format(err / 10 ** math.floor(math.log10(err))), "results/attention/b1.json")
    put("BeffErrExp", math.floor(math.log10(err)), "results/attention/b1.json")

    # half precision, as the captioner runs (unpinned)
    bf = ops("attention/b1_bf16.json")
    for be, name in (("flash_attention", "Flash"), ("cudnn_attention", "Cudnn"), ("efficient_attention", "Effbf"),
                     ("math", "Mathbf")):
        put(f"Bf{name}Ms", med(bf[f"b1_{be}_bf16_h12_n1024"]), "results/attention/b1_bf16.json", "{:.2f}")
    put("BfGhz", group_mhz("attention/b1_bf16.json", "b1_flash_attention_bf16_h12_n1024") / 1e3,
        "results/attention/b1_bf16.json", "{:.2f}")
    put("BfFlashTflops", 4 * 12 * 1024 * 1024 * 64 / (med(bf["b1_flash_attention_bf16_h12_n1024"]) * 1e-3) / 1e12,
        "results/attention/b1_bf16.json", "{:.0f}")
    put("VthreeOverFlash", MACRO_VAL["DepVthreeMs"] / med(bf["b1_flash_attention_bf16_h12_n1024"]),
        "results/attention/v3.json / b1_bf16.json (as deployed, both unpinned)", "{:.1f}")

    # the encoder's GPU time by kernel kind (bf16, four frames) and attention's share of a caption
    ep = load("e1/encoder_profile.json")
    src = "results/e1/encoder_profile.json"
    put("EncGpuMs", ep["gpu_ms_per_call"], src, "{:.0f}")
    for k, name in (("attention", "Attn"), ("matmul", "Matmul"), ("other", "Other"), ("convolution", "Conv")):
        put(f"Enc{name}Pct", 100 * ep["share_by_kind"][k], src, "{:.0f}")
    put("AttnCaptionMeasPct", 100 * ep["share_by_kind"]["attention"] * es["median_share_of_total"]["encoder_s"],
        "results/e1/encoder_profile.json x results/e1/encoder_share.json", "{:.0f}")

    # pixel-level redundancy (X0, ERA training split)
    x0 = load("x0/era_tra/summary.json")
    src = "results/x0/era_tra/summary.json"
    put("XzeroClips", x0["clips_processed"], src, "{:,}")
    put("XzeroGrouped", sum(x0["slice_sizes"][g] for g in ("hover", "moving", "unreliable")), src, "{:,}")
    put("XzeroRejected", x0["clips_processed"] - sum(x0["slice_sizes"][g] for g in ("hover", "moving", "unreliable")), src)
    for g, name in (("hover", "Still"), ("moving", "Moving"), ("unreliable", "Untrack")):
        put(f"Xzero{name}Clips", x0["slice_sizes"][g], src)
        put(f"Xzero{name}Keep", 100 * x0["slices"][g]["1/8"]["steady_mean"], src, "{:.0f}")
        put(f"Xzero{name}Cost", x0["slices"][g]["1/8"]["cost_steady"], src, "{:.1f}")

    # stall shares and repeat runs quoted in the text
    pb_st = {v: pb["versions"][v]["phases"] for v in ("V1", "V2")}
    glob = [sum(r["stalls"].get(k, 0.0) for k in ("long_scoreboard", "lg_throttle")) for r in pb_st["V1"]]
    put("VoneGlobalLo", 100 * min(glob), "results/attention/x2_ncu_v1.txt via x2_breakdown.json", "{:.0f}")
    put("VoneGlobalHi", 100 * max(glob), "results/attention/x2_ncu_v1.txt via x2_breakdown.json", "{:.0f}")
    ivs = {i["cause"]: i for i in load("attention/x2_breakdown.json")["interventions"]}
    bnd = ivs["shared-memory store conflicts (V2 scores)"]["bound"]
    put("VtwoConflictPct", 100 * bnd["conflicts_share_of_wavefronts"], "results/attention/x2_breakdown.json", "{:.0f}")
    put("VtwoMioPct", 100 * bnd["mio_share_of_warp_cycles"], "results/attention/x2_breakdown.json", "{:.1f}")
    rep = []
    for name in ("v1", "v2", "v3", "b1"):
        a = {o["name"]: o for o in load(f"attention/{name}.json")["ops"] if o["status"] == "ok"}
        b = {o["name"]: o for o in load(f"attention/{name}_repeat.json")["ops"] if o["status"] == "ok"}
        rep += [abs(med(b[k]) / med(a[k]) - 1) for k in a if k in b]
    put("RepeatMaxPct", 100 * max(rep), "results/attention/{v1,v2,v3,b1}{,_repeat}.json", "{:.0f}")

    # V1's score kernel: the L1 load path saturates (results/attention/v1_ncu.txt, first kernel)
    v1n = (RES / "attention" / "v1_ncu.txt").read_text().split("v1_softmax")[0]
    grab = lambda key: float([ln for ln in v1n.splitlines() if key in ln][0].split()[-1].replace(",", ""))  # noqa: E731
    put("VoneLonePct", grab("l1tex__throughput"), "results/attention/v1_ncu.txt", "{:.1f}")
    put("VoneLtwoPct", grab("lts__throughput"), "results/attention/v1_ncu.txt", "{:.1f}")
    put("VoneFmaPct", grab("sm__pipe_fma_cycles_active"), "results/attention/v1_ncu.txt", "{:.1f}")

    # B1's rescaled range without the math backend's N x N round trip (which the memory-efficient kernel avoids)
    put("PinBeffPredHiNoNxN", (4 * 12 * 1024 * 1024 * 64) / (0.35 * pk["fma_sustained"]["tflops_settled"] * 1e12
                                                            * pin["pinned_mhz"] / 1900.0) * 1e3
        + pk["launch_b2b"]["ms_median"], "bench/predict_times.py B1 high end without the N x N term, at the pin",
        "{:.2f}")

    # V3 against V2 at one head and small N (Fig. 1)
    v2o = ops("attention/v2.json")
    v3o = ops("attention/v3.json")
    small = [med(v3o[f"v3_h1_n{n}"]) / med(v2o[f"v2_h1_n{n}"]) for n in (128, 256, 512)]
    put("VthreeSmallLo", min(small), "results/attention/v2.json, v3.json", "{:.1f}")
    put("VthreeSmallHi", max(small), "results/attention/v2.json, v3.json", "{:.1f}")

    # stale input in the token traces: its largest median margin over the baseline, against the noise floor
    eqA = [v for g in ("hover", "moving", "unreliable")
           for v in load("tokens/era_tra/summary.json")["groups"][g]["equal_cost"]["A"].values() if v]
    put("TierAMaxMargin", -100 * min(v["median"] for v in eqA), "results/tokens/era_tra/summary.json", "{:.1f}")
    put("NoiseFloorPct", 100 * load("tokens/era_tra/summary.json")["floors_per_clip"]["noise_pm1"]["mean"]["mean"],
        "results/tokens/era_tra/summary.json", "{:.1f}")
    put("TokGrouped", sum(len(load("tokens/era_tra/summary.json")["groups"][g]["clips"])
                          for g in ("hover", "moving", "unreliable")), "results/tokens/era_tra/summary.json")

    # caption check: stale input at threshold 2, with and without CIDEr's length penalty
    ca = cs["conditions"]["A_2"]["cider_minus_dense"]
    put("CapAtwoDelta", ca["mean"], "results/gate_captions/scores.json", "{:+.3f}")
    put("CapAtwoLo", ca["lo95"], "results/gate_captions/scores.json", "{:+.3f}")
    put("CapAtwoHi", ca["hi95"], "results/gate_captions/scores.json", "{:+.3f}")
    nl = cs["post_hoc"]["no_length_penalty"]["A_2"]["cider_minus_dense"]
    put("CapAtwoNlLo", nl["lo95"], "results/gate_captions/scores.json (post hoc)", "{:+.3f}")
    put("CapAtwoNlHi", nl["hi95"], "results/gate_captions/scores.json (post hoc)", "{:+.3f}")
    put("CapGateConds", sum(1 for k in cs["conditions"] if k.split("_")[0] in ("A", "B", "C")),
        "results/gate_captions/scores.json")
    put("CapUnadjusted", len(cs["conditions"]) - 1, "results/gate_captions/scores.json (every condition but dense)")
    eff = load("cuda/predicted_times.json")["assumptions"]["efficiency_share_of_peak"]["naive_matmul"]
    put("PredVoneEffLo", 100 * eff[0], "results/cuda/predicted_times.json", "{:.0f}")
    put("PredVoneEffHi", 100 * eff[1], "results/cuda/predicted_times.json", "{:.0f}")

    # embedding similarity for the caption check (post hoc; eval/caption_embed.py)
    em = load("gate_captions/embed_scores.json")
    src = "results/gate_captions/embed_scores.json"
    put("EmbDense", em["conditions"]["dense"]["score_mean"], src, "{:.3f}")
    for k, name in (("B_8", "Beight"), ("C_0.2", "Ctwo"), ("hold_24", "Holdone"), ("A_2", "Atwo")):
        ci = em["conditions"][k]["minus_dense"]
        put(f"Emb{name}Delta", ci["mean"], src, "{:+.3f}")
        put(f"Emb{name}Lo", ci["lo95"], src, "{:+.3f}")
        put(f"Emb{name}Hi", ci["hi95"], src, "{:+.3f}")
    sh = em["shuffled_control"]["minus_dense"]
    put("EmbShuffledDelta", -sh["mean"], src, "{:.2f}")
    others = [v["minus_dense"] for k, v in em["conditions"].items() if k != "dense"]
    put("EmbExcludeZero", sum(1 for c in others if c["lo95"] > 0 or c["hi95"] < 0), src)
    half = max((c["hi95"] - c["lo95"]) / 2 for c in others)
    put("EmbHalfWidth", half, src, "{:.3f}")
    put("EmbResolutionPct", 100 * half / -sh["mean"], src, "{:.0f}")
    capres = max((v["cider_minus_dense"]["hi95"] - v["cider_minus_dense"]["lo95"]) / 2
                 for k, v in cs["conditions"].items() if k != "dense")
    put("CapResolutionPct", 100 * capres / -cs["post_hoc"]["shuffled_control"]["cider_minus_dense"]["mean"],
        "results/gate_captions/scores.json", "{:.0f}")
    eq = em["equal_cost_vs_hold"]
    put("EmbEqAtwoDelta", eq["A_2"]["mean"], src, "{:+.3f}")
    put("EmbEqAtwoLo", eq["A_2"]["lo95"], src, "{:+.3f}")
    put("EmbEqAtwoHi", eq["A_2"]["hi95"], src, "{:+.3f}")
    put("EmbEqBetter", sum(1 for v in eq.values() if v and v["lo95"] > 0), src)
    put("EmbEqWorse", sum(1 for v in eq.values() if v and v["hi95"] < 0), src)
    put("EmbBfloatCos", em["conditions"]["e1_bf16"]["cos_to_dense_caption"], src, "{:.2f}")
    put("EmbHoldoneCos", em["conditions"]["hold_24"]["cos_to_dense_caption"], src, "{:.2f}")

    # data and samples
    scan = load("era/scan.json")["clips"]
    put("EraClips", len(scan), "results/era/scan.json", "{:,}")
    put("AttnGflop", 4 * 12 * 1024 * 1024 * 64 / 1e9, "4 H N^2 d at the encoder's shape", "{:.1f}")
    tr = load("tokens/era_tra/summary.json")["groups"]
    for g, name in (("hover", "Still"), ("moving", "Moving"), ("unreliable", "Untrack")):
        L = tr[g]["trace"]["1"]["layers"]
        put(f"Tr{name}OutEmb", 100 * L["0"]["outside_mean"], "results/tokens/era_tra/summary.json", "{:.0f}")
        put(f"Tr{name}OutDeep", 100 * max(L[str(i)]["outside_mean"] for i in range(1, 13)),
            "results/tokens/era_tra/summary.json (largest over encoder layers)", "{:.0f}")
        put(f"Tr{name}In", 100 * L["0"]["inside_mean"], "results/tokens/era_tra/summary.json", "{:.0f}")
    tk = load("tokens/era_tra/summary.json")
    put("TokClips", tk["clips_processed"], "results/tokens/era_tra/summary.json")
    put("TokPerGroup", len(tk["groups"]["hover"]["clips"]), "results/tokens/era_tra/summary.json")
    put("CapConditions", len(cs["conditions"]), "results/gate_captions/scores.json")

    OUT.mkdir(exist_ok=True)
    lines = ["% Generated by paper/make_numbers.py from committed result files. Do not edit.\n"]
    for k, v in MACROS.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}  % {SOURCES[k]}\n")
    (OUT / "numbers.tex").write_text("".join(lines))

    # Table: predicted against measured, at the pinned clock (headline) and as deployed
    rows = [("V1 (naive)", "Vone"), ("V2 (tiled)", "Vtwo"), ("V3 (fused)", "Vthree"),
            ("B1 math", "Bmath"), ("B1 mem.-eff.", "Beff")]
    pred_key = {"Vone": "Vone", "Vtwo": "Vtwo", "Vthree": "Vthree", "Bmath": "Bone", "Beff": "Bone"}
    tab = [r"% Generated by paper/make_numbers.py. Do not edit." "\n",
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

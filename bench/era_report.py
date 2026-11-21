"""Summarise results/era/scan.json, check the detectors against the hand labels,
and count camera-still ("hover") clips per class.

The hover rule is fixed from the labelled training sample only
(bench/era_labels/tra_sample_seed0.json) and recorded in the output, before any
keep-rate is measured on ERA. Test-split clips are counted, never used to
choose anything.

Example:
    .venv/bin/python -I bench/era_report.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Hover rule, fixed from the labelled training sample before any ERA keep-rate was measured.
# Sorted by estimated speed, the sample's eye-labelled still clips fill the range up to 0.0096
# frame widths per second and the first reliable drifting clip is at 0.0134; 0.012 sits between.
# 0.012 fw/s is 0.38 encoder patches (1/32 of the width) per second.
# Do not tune these on keep-rates.
HOVER_MAX_FW_PER_S = 0.012
HOVER_MIN_RELIABLE = 0.5


def is_hover(r) -> bool:
    m = r["motion"]
    return (m["median_fw_per_s"] is not None and m["reliable_frac"] >= HOVER_MIN_RELIABLE
            and m["median_fw_per_s"] <= HOVER_MAX_FW_PER_S)


def clip_key(r):
    return (r["split"], r["cls"], r["id"])


def q(a, ps=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray([x for x in a if x is not None], dtype=float)
    return {f"p{p}": round(float(np.percentile(a, p)), 4) for p in ps} if len(a) else None


def main() -> None:
    scan = json.loads((ROOT / "results" / "era" / "scan.json").read_text())
    clips = [r for r in scan["clips"] if "error" not in r]
    errors = [clip_key(r) for r in scan["clips"] if "error" in r]
    labels = json.loads((ROOT / "bench" / "era_labels" / "tra_sample_seed0.json").read_text())["clips"]
    by_key = {clip_key(r): r for r in clips}
    rep = {"clips": len(scan["clips"]), "errors": errors}

    # integrity
    rep["frames"] = q([r["frames"] for r in clips])
    rep["fps"] = Counter(round(r["fps"], 3) for r in clips).most_common()
    rep["resolution"] = Counter(f'{r["width"]}x{r["height"]}' for r in clips).most_common()
    rep["pts_broken"] = [clip_key(r) for r in clips if r["pts_broken"]]
    miss = [r["missing_slots"] for r in clips]
    rep["missing_slots"] = {"clips_with_any": sum(m > 0 for m in miss), "dist": Counter(miss).most_common(8)}

    # repeated frames: pairs far below the clip's own typical change
    sig = np.load(ROOT / "results" / "era" / "raw" / "signals.npz")
    dup_clips, dup_pairs, all_pairs = 0, 0, 0
    for r in clips:
        m = sig[f'{r["split"]}/{r["cls"]}/{r["id"]}/mad1']
        med = float(np.median(m))
        all_pairs += len(m)
        if med < 2.0:  # too still to tell a repeat from a quiet scene
            continue
        d = int(((m < 0.5) & (m < 0.1 * med)).sum())
        dup_pairs += d
        dup_clips += d > 0
    rep["repeated_frames"] = {"rule": "consecutive grey MAD < 0.5 and < 0.1 x clip median (clips with median >= 2)",
                              "clips_with_any": dup_clips, "pairs": dup_pairs, "of_pairs": all_pairs}

    rep["key_pulse"] = q([r["key_pulse"] for r in clips])
    rep["keyframes_per_clip"] = Counter(r["keyframes"] for r in clips).most_common(6)
    rep["static_block_frac"] = q([r["static_block_frac"] for r in clips])
    rep["clips_static_ge_5pct"] = sum(r["static_block_frac"] >= 0.05 for r in clips)
    bars = [r for r in clips if any(v > 0 for v in r["bars_px"].values())]
    rep["clips_with_dark_bars"] = len(bars)

    # shot detector vs labels
    flagged = lambda r: len(r["shot"]["events"]) > 0  # noqa: E731
    lab = [(l, by_key[clip_key(l)]) for l in labels]
    rep["shot_vs_labels"] = {
        "multi": [(l["id"], flagged(r), r["shot"]["events"]) for l, r in lab if l["shot"] == "multi"],
        "single_flagged": [(l["id"], r["shot"]["events"]) for l, r in lab if l["shot"] == "single" and flagged(r)],
        "single_total": sum(l["shot"] == "single" for l, _ in lab),
    }
    rep["shot_flagged"] = {s: sum(flagged(r) for r in clips if r["split"] == s) for s in ("Tra", "Test")}

    # motion vs labels
    by_lab = defaultdict(list)
    for l, r in lab:
        m = r["motion"]
        by_lab[l["camera"]].append((m["median_fw_per_s"], m["reliable_frac"], l["id"]))
    rep["motion_vs_labels"] = {c: {"n": len(v), "reliable_frac": q([x[1] for x in v]),
                                   "median_fw_per_s": q([x[0] for x in v])} for c, v in sorted(by_lab.items())}
    rep["motion_sorted"] = sorted(((round(x[0], 4) if x[0] is not None else None, round(x[1], 2), c, x[2])
                                   for c, v in by_lab.items() for x in v),
                                  key=lambda t: (t[0] is None, t[0] or 0))
    # hover rule against the eye labels, then counts
    rep["hover_rule"] = {"max_fw_per_s": HOVER_MAX_FW_PER_S, "min_reliable_frac": HOVER_MIN_RELIABLE}
    rep["hover_vs_labels"] = {c: Counter(is_hover(r) for l, r in lab if l["camera"] == c) for c in "SDMT"}
    hov = [r for r in clips if is_hover(r)]
    rep["hover_count"] = {s_: sum(r["split"] == s_ for r in hov) for s_ in ("Tra", "Test")}
    per_cls = defaultdict(lambda: [0, 0, 0, 0])
    for r in clips:
        i = 0 if r["split"] == "Tra" else 2
        per_cls[r["cls"]][i + 1] += 1
        per_cls[r["cls"]][i] += is_hover(r)
    rep["hover_per_class"] = {c: {"Tra": f"{v[0]}/{v[1]}", "Test": f"{v[2]}/{v[3]}"} for c, v in sorted(per_cls.items())}
    rep["hover_shot_flagged"] = [clip_key(r) for r in hov if flagged(r)]
    # Clean hover set: hover and no shot event. In clips the hover rule picks, a shot event usually marks a
    # fast pan, tilt or zoom in part of the clip, not a cut (bench/era_labels/tra_hover_check_seed1.json:
    # 12 of 13 at every frame, 1 unclear), so these clips are not still throughout and are dropped.
    clean = [r for r in hov if not flagged(r)]
    rep["hover_clean_count"] = {s_: sum(r["split"] == s_ for r in clean) for s_ in ("Tra", "Test")}
    hc = json.loads((ROOT / "bench" / "era_labels" / "tra_hover_check_seed1.json").read_text())["clips"]
    rep["hover_eye_check"] = {
        "flagged": Counter(f'{r["camera"]}/{r["shot"]}' for r in hc if r["shot_flagged"]),
        "unflagged": Counter(f'{r["camera"]}/{r["shot"]}' for r in hc if not r["shot_flagged"]),
    }
    (ROOT / "results" / "era" / "hover_tra.json").write_text(json.dumps(
        {"rule": rep["hover_rule"] | {"no_shot_events": True}, "clips": [clip_key(r) for r in clean if r["split"] == "Tra"]},
        indent=0) + "\n")
    rep["unreliable_motion"] = {s_: sum(r["split"] == s_ and r["motion"]["reliable_frac"] < HOVER_MIN_RELIABLE for r in clips)
                                for s_ in ("Tra", "Test")}

    # Three buckets. The estimator finds hover clips; it is not a hover/moving classifier: in the labelled
    # sample the unreliable bucket is almost all eye-labelled moving or tracking footage (fast, water).
    def bucket(r, cut=HOVER_MAX_FW_PER_S):
        m = r["motion"]
        if m["median_fw_per_s"] is None or m["reliable_frac"] < HOVER_MIN_RELIABLE:
            return "unreliable"
        return "hover" if m["median_fw_per_s"] <= cut else "moving"
    rep["buckets"] = {s_: Counter(bucket(r) for r in clips if r["split"] == s_) for s_ in ("Tra", "Test")}
    rep["buckets_vs_labels"] = Counter(f'{l["camera"]}->{bucket(r)}' for l, r in lab)
    rep["hover_cutoff_sensitivity"] = {
        f"{c:g}": {s_: [sum(r["split"] == s_ and bucket(r, c) == "hover" for r in clips),
                        sum(r["split"] == s_ and bucket(r, c) == "hover" and not flagged(r) for r in clips)]
                   for s_ in ("Tra", "Test")}
        for c in (0.010, HOVER_MAX_FW_PER_S, 0.015)}  # [hover, hover without shot events]
    rep["clean_hover_tra_repeats_unmeasurable"] = [sum(r["mad1_median"] < 2.0 for r in clean if r["split"] == "Tra"),
                                                   sum(r["split"] == "Tra" for r in clean)]

    # Static blocks. Flat sky or water stays under the threshold even under camera motion, so only static
    # blocks with texture (static_textured_frac) are overlay suspects.
    moving = [r for r in clips if r["motion"]["reliable_frac"] >= HOVER_MIN_RELIABLE and not is_hover(r)]
    rep["static_in_hover"] = q([r["static_block_frac"] for r in hov])
    rep["static_in_moving"] = q([r["static_block_frac"] for r in moving])
    rep["static_textured"] = q([r.get("static_textured_frac") for r in clips])
    rep["overlay_suspects"] = sorted(((round(r["static_textured_frac"], 3), *clip_key(r)) for r in moving
                                      if r.get("static_textured_frac", 0) >= 0.01), reverse=True)
    rep["dark_bars"] = Counter(tuple(k for k, v in r["bars_px"].items() if v > 0) for r in bars).most_common()

    # repeated frames: how many per clip, and their spacing (frame-rate conversion leaves a fixed period)
    per_clip, gaps = [], Counter()
    for r in clips:
        m = sig[f'{r["split"]}/{r["cls"]}/{r["id"]}/mad1']
        med = float(np.median(m))
        if med < 2.0:
            continue
        idx = np.flatnonzero((m < 0.5) & (m < 0.1 * med))
        per_clip.append(len(idx))
        gaps.update(np.diff(idx).tolist())
    rep["repeated_per_clip"] = Counter(min(n, 30) for n in per_clip).most_common(12)
    rep["repeated_spacing"] = gaps.most_common(8)

    out = ROOT / "results" / "era" / "report.json"
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({k: rep[k] for k in ("hover_rule", "hover_vs_labels", "hover_count", "hover_per_class",
                                          "hover_shot_flagged", "hover_clean_count", "hover_eye_check", "buckets",
                                          "buckets_vs_labels", "hover_cutoff_sensitivity",
                                          "clean_hover_tra_repeats_unmeasurable",
                                          "unreliable_motion", "static_in_hover", "static_textured",
                                          "static_in_moving", "overlay_suspects", "dark_bars", "repeated_per_clip",
                                          "repeated_spacing")}, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

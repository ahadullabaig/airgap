"""E3: score the demo's reports against the event lists (PLAN.md 2.5). Fixed 2026-11-27, before any demo run.

Inputs per video: the event list (eval/e3_events/<video>.json, committed before the run), the run record
(results/demo/<video>/run.json) and a judgment file written by the scorer after reading the report and looking
at the frames the model saw (results/e3/<video>_judgments.json). The scorer is Claude, at the authors' request;
Claude also wrote the event lists, so the write-up must say that the same AI wrote the reference and judged.

Rule:
  Events found. An event is found when at least one timeline caption describes it: names its key object or
    activity ("found_if" in the event list for main events; the event's description otherwise). Synonyms count;
    naming only the setting does not. Notable-events lines repeat captions, so they add nothing to this count.
  In the right window. A found event is in the right window when one of the captions that describes it belongs
    to a window overlapping [start - slack, end + slack]: 1 s at a camera cut, 5 s in a continuous flight
    (the event's "boundary").
  Summary. Whether the summary paragraph mentions each main event (it has no times, so only presence counts).
  Invented content. Every timeline caption, notable-events line and summary sentence gets a verdict:
    supported  everything it says can be seen in the video at that time (for the summary, anywhere)
    partly     its main content is right but a detail is not in the video
    invented   its main content is not in the video
    A notable-events line is judged on its event label: is that event in its windows?
  Reported per video and over the three: main events found and in the right window, all events found and in the
  right window, main events in the summary, and the verdict counts per section.

Judgment file:
  {"video": ..., "scorer": ..., "events": {"<id>": {"described_by": [window indices], "summary": bool, "note": ""}},
   "timeline": {"<window index>": {"verdict": ..., "note": ""}},
   "notable": [{"verdict": ..., "note": ""}]   # in report order
   "summary": [{"sentence": ..., "verdict": ..., "note": ""}]}

Example:
    .venv/bin/python -I eval/e3.py score
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "eval" / "e3_events"
OUT = ROOT / "results" / "e3"
VIDEOS = ["demo_or58_truck_recovery", "demo_minnesota_flood", "demo_surfing_dana_point"]
SLACK = {"cut": 1.0, "gradual": 5.0}
VERDICTS = ("supported", "partly", "invented")


def score_video(name: str) -> dict:
    ev = json.loads((EVENTS / f"{name}.json").read_text())
    run = json.loads((ROOT / "results" / "demo" / name / "run.json").read_text())
    j = json.loads((OUT / f"{name}_judgments.json").read_text())
    wins = {w["index"]: w for w in run["windows"]}
    assert set(j["timeline"]) == {str(k) for k, w in wins.items() if "caption" in w}, "judge every caption"
    assert len(j["notable"]) == len(run["notable"]), "judge every notable-events line"
    rows = []
    for e in ev["events"]:
        jd = j["events"][str(e["id"])]
        slack = SLACK[e["boundary"]]
        found = bool(jd["described_by"])
        right = any(wins[k]["start_s"] < e["end_s"] + slack and wins[k]["end_s"] > e["start_s"] - slack
                    for k in jd["described_by"])
        rows.append({"id": e["id"], "main": e["main"], "found": found, "right_window": right,
                     "summary": bool(jd.get("summary")) if e["main"] else None, "described_by": jd["described_by"]})
    for section in ("timeline", "notable", "summary"):
        items = j[section].values() if section == "timeline" else j[section]
        assert all(i["verdict"] in VERDICTS for i in items), f"bad verdict in {section}"

    def tally(rs):
        return {"events": len(rs), "found": sum(r["found"] for r in rs), "right_window": sum(r["right_window"] for r in rs)}
    main = [r for r in rows if r["main"]]
    return {"video": name, "scorer": j["scorer"], "main": tally(main) | {"in_summary": sum(r["summary"] for r in main)},
            "all": tally(rows), "events": rows,
            "verdicts": {s: dict(Counter(i["verdict"] for i in (j[s].values() if s == "timeline" else j[s])))
                         for s in ("timeline", "notable", "summary")}}


def score() -> None:
    per = [score_video(v) for v in VIDEOS]
    total = {"main": {k: sum(p["main"][k] for p in per) for k in per[0]["main"]},
             "all": {k: sum(p["all"][k] for p in per) for k in per[0]["all"]},
             "verdicts": {s: dict(sum((Counter(p["verdicts"][s]) for p in per), Counter()))
                          for s in ("timeline", "notable", "summary")}}
    res = {"experiment": "E3: the demo's reports against event lists written before the runs",
           "reference_and_scorer": "both Claude, at the authors' request; not checked by a person",
           "videos": per, "total": total}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(res, indent=1) + "\n")
    for p in per + [{"video": "all three", **total}]:
        m, a = p["main"], p["all"]
        print(f"{p['video']:28s} main {m['found']}/{m['events']} found, {m['right_window']} in the right window, "
              f"{m['in_summary']} in the summary; all {a['found']}/{a['events']} found, {a['right_window']} right window")
        print(f"{'':28s} {p['verdicts']}")


def main() -> None:
    if sys.argv[1:] != ["score"]:
        sys.exit(__doc__)
    score()


if __name__ == "__main__":
    main()

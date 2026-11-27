"""E4: error analysis (PLAN.md section 2.5). Selects about 50 CapERA test clips for the team to tag by failure
type, prints what a tagger needs, and counts the tags. Uses only E1's and E2's committed outputs, so it does not
wait for the demo; if the demo changes E1's settings and E1/E2 are rerun, redo the selection.

Everything below was fixed before any clip was tagged.

Selection (seed 0), two clips per class, 50 in all:
  slot 1  a random clip that E2 got wrong (strict rule: wrong class, unmatched or ambiguous). Boating and
          CarRacing have no E2 errors; their slot 1 is filled as slot 2 is.
  slot 2  the class's lowest-scoring E1 caption (CIDEr-D, then METEOR, then id, ascending), not already chosen.
  Each row records why it was picked, and whether it is also an E2 error or among its class's five lowest E1
  captions. The sample is deliberately made of failures, so its counts describe failures, not the test split.
Good examples: for each class, the highest-CIDEr E1 caption among clips E2 got right (any clip if E2 got none
right). The team picks five of these 25 for the slides, and five bad examples from the tagged 50.

Tagging (results/e4/tags.csv; one row per clip). Watch the clip, read the references (`show`), and read both of
our outputs: the E1 caption and the E2 answer. Put 1 in every column that applies, to either output:
  small_objects     the key objects or people are small in the frame, and an output misses or misreads them
  counting          an output gets a number or amount wrong (one car for many, a few people for a crowd)
  viewpoint         the aerial angle or height causes a misreading (shapes or scale from above)
  invented_objects  an output names something that is not in the video
  wrong_event       an output names the wrong event or activity. Every E2 error is one by definition, so `select`
                    fills it in on every row with e2_wrong = 1 (40 of the 50); the other columns say why
  acceptable        the caption is a fair description of the clip. On E1-picked rows this means a metric or
                    reference problem (fair despite its low score); on E2-picked rows, the caption was right where
                    the answer was not
  other             anything else; say what in notes
Fill tagger with your initials on each row you have tagged; `count` refuses a sheet with any row untagged.

Counts (results/e4/counts.json): how many rows carry each tag, overall, by why the row was picked, and by
whether E2 got it wrong; the classes behind each tag; the notes. If two people tag, each fills their own copy (tags_<initials>.csv); the
counts then come from the first sheet given, and the agreement per tag between the two is added (share of rows
where both agree, and Cohen's kappa).

Committed files hold ids, class names and our outputs, never CapERA's reference captions: `show` prints the
references to the terminal only.

Example:
    .venv/bin/python -I eval/e4.py select
    .venv/bin/python -I eval/e4.py show [Fire_012.mp4 ...]     # all 50 when no id is given
    .venv/bin/python -I eval/e4.py count [results/e4/tags_ab.csv results/e4/tags_cd.csv]
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import capera  # noqa: E402

OUT = ROOT / "results" / "e4"
TAGS = ["small_objects", "counting", "viewpoint", "invented_objects", "wrong_event", "acceptable", "other"]
FIELDS = ["id", "cls", "picked_for", "e2_wrong", "e1_low5", "e2_answer", "e1_caption", "e1_cider", *TAGS, "tagger",
          "notes"]
SEED = 0
LOW = 5


def outputs() -> tuple[dict, dict]:
    e1 = {r["id"]: r for r in json.loads((ROOT / "results" / "e1" / "per_clip_test.json").read_text())}
    e2 = {r["id"]: r for r in json.loads((ROOT / "results" / "e2" / "per_clip_test.json").read_text())}
    assert set(e1) == set(e2), "E1 and E2 cover different clips"
    return e1, e2


def select() -> None:
    sheet = OUT / "tags.csv"
    if sheet.exists():
        sys.exit(f"{sheet} exists: the selection is fixed")
    e1, e2 = outputs()
    rng = random.Random(SEED)
    by_cls: dict[str, list[str]] = {}
    for i in sorted(e1):
        by_cls.setdefault(e1[i]["cls"], []).append(i)
    rows, good = [], []
    for cls in sorted(by_cls):
        ids = by_cls[cls]
        wrong = [i for i in ids if e2[i]["strict"] != cls]
        low = sorted(ids, key=lambda i: (e1[i]["CIDEr"], e1[i]["METEOR"], i))
        low5 = set(low[:LOW])
        picks = []
        if wrong:
            picks.append((rng.choice(wrong), "e2_wrong"))
        for i in low:
            if len(picks) == 2:
                break
            if i not in {p for p, _ in picks}:
                picks.append((i, "e1_low"))
        for i, why in picks:
            rows.append({"id": i, "cls": cls, "picked_for": why, "e2_wrong": int(i in wrong),
                         "e1_low5": int(i in low5), "e2_answer": e2[i]["answer"], "e1_caption": e1[i]["caption"],
                         "e1_cider": f"{e1[i]['CIDEr']:.3f}", **{t: "" for t in TAGS}, "tagger": "", "notes": ""} | {"wrong_event": "1" if i in wrong else ""})
        right = [i for i in ids if e2[i]["strict"] == cls] or ids
        best = max(right, key=lambda i: (e1[i]["CIDEr"], i))
        good.append({"id": best, "cls": cls, "e2_answer": e2[best]["answer"], "e2_right": int(best not in wrong),
                     "e1_caption": e1[best]["caption"], "e1_cider": round(e1[best]["CIDEr"], 3)})
    OUT.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    (OUT / "good_candidates.json").write_text(json.dumps(good, indent=1) + "\n")
    print(f"{len(rows)} clips: {Counter(r['picked_for'] for r in rows)}; also E2 errors: "
          f"{sum(r['e2_wrong'] for r in rows)}, also among the {LOW} lowest E1 captions: "
          f"{sum(r['e1_low5'] for r in rows)}")
    print(f"wrote {sheet} and {OUT / 'good_candidates.json'}")


def show(ids: list[str]) -> None:
    clips = {c.id: c for c in capera.load("test")}
    rows = list(csv.DictReader((OUT / "tags.csv").open()))
    picked = {r["id"]: r for r in rows}
    for n, i in enumerate(ids or list(picked)):
        c, r = clips[i], picked.get(i)
        print(f"\n[{n + 1}] {i}  class {c.cls}  {c.path.relative_to(ROOT)}")
        if r:
            print(f"    picked for {r['picked_for']}   E2 answer: {r['e2_answer']}")
            print(f"    E1 caption (CIDEr {r['e1_cider']}): {r['e1_caption']}")
        for k, ref in enumerate(c.refs):
            print(f"    reference {k + 1}{' (human)' if k == 0 else ' (paraphrase)'}: {ref}")


def read_sheet(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        if not r["tagger"].strip():
            sys.exit(f"{path.name}: row {r['id']} has no tagger: fill in your initials once the row is tagged")
        bad = [t for t in TAGS if r[t].strip() not in ("", "1")]
        if bad or not any(r[t].strip() == "1" for t in TAGS):
            sys.exit(f"{path.name}: row {r['id']} needs at least one 1 and nothing else in {', '.join(TAGS)}")
        if r["e2_wrong"] == "1" and r["wrong_event"].strip() != "1":
            sys.exit(f"{path.name}: row {r['id']} is an E2 error, so wrong_event must be 1")
    return rows


def kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def count(paths: list[Path]) -> None:
    paths = [p.resolve() for p in paths] or [OUT / "tags.csv"]
    sheets = [read_sheet(p) for p in paths]
    rows = sheets[0]
    res = {"sheet": str(paths[0].relative_to(ROOT)), "rows": len(rows),
           "note": "a sample of failures (E2 errors and each class's lowest E1 captions), not the test split",
           "tags": {}}
    for t in TAGS:
        hit = [r for r in rows if r[t].strip() == "1"]
        res["tags"][t] = {"rows": len(hit), "by_pick": dict(Counter(r["picked_for"] for r in hit)),
                          "e2_wrong": sum(r["e2_wrong"] == "1" for r in hit),
                          "classes": dict(Counter(r["cls"] for r in hit).most_common())}
    res["notes"] = [{"id": r["id"], "note": r["notes"]} for r in rows if r["notes"].strip()]
    if len(sheets) == 2:
        a = {r["id"]: r for r in sheets[0]}
        b = {r["id"]: r for r in sheets[1]}
        assert set(a) == set(b), "the two sheets cover different clips"
        ids = sorted(a)
        res["agreement"] = {}
        for t in TAGS:
            x = [int(a[i][t].strip() == "1") for i in ids]
            y = [int(b[i][t].strip() == "1") for i in ids]
            res["agreement"][t] = {"agree": sum(p == q for p, q in zip(x, y)) / len(ids), "kappa": kappa(x, y)}
    (OUT / "counts.json").write_text(json.dumps(res, indent=1) + "\n")
    for t, v in res["tags"].items():
        print(f"{t:17s} {v['rows']:3d}  {v['by_pick']}  E2 errors: {v['e2_wrong']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["select", "show", "count"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    {"select": select, "show": lambda: show(a.args), "count": lambda: count([Path(p) for p in a.args])}[a.command]()


if __name__ == "__main__":
    main()

"""CapERA captions joined to the ERA videos.

CapERA (data/capera/CapERA_DATASET_{train,test}.json) gives 5 captions per clip; ERA
(data/era/Videos/{Tra,Test}/<Class>/<Class>_NNN .mp4) gives the videos. Ids repeat across splits, so
everything is keyed on (split, id). Each clip's window is its own length (from results/era/scan.json:
the timestamp span plus one frame), not the demo's 8 s default.

CapERA states no licence: reference captions are read from data/ and never written to committed files.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERA_SPLIT = {"train": "Tra", "test": "Test"}


@dataclass
class Clip:
    split: str
    id: str
    cls: str
    path: Path
    duration_s: float
    refs: list[str]


def load(split: str) -> list[Clip]:
    caps = json.loads((ROOT / "data" / "capera" / f"CapERA_DATASET_{split}.json").read_text())["ERA_caption"]
    era = ERA_SPLIT[split]
    scan = {(r["split"], r["id"]): r for r in json.loads((ROOT / "results" / "era" / "scan.json").read_text())["clips"]}
    out = []
    for c in caps:
        vid = c["video_id"]
        cls = vid.rsplit("_", 1)[0]
        s = scan[(era, vid)]
        if s.get("pts_broken"):
            duration = s["frames"] / s["fps"]
        else:
            duration = s["pts_span_s"] + 1.0 / s["fps"]
        out.append(Clip(split, vid, cls, ROOT / "data" / "era" / "Videos" / era / cls / s["file"], duration,
                        list(c["annotation"]["English_caption"])))
    out.sort(key=lambda c: (c.cls, c.id))
    return out


def stratified_order(clips: list[Clip], seed: int) -> list[Clip]:
    """Round robin over classes after a seeded shuffle within each class: any prefix is class-stratified."""
    rng = random.Random(seed)
    by_cls: dict[str, list[Clip]] = {}
    for c in clips:
        by_cls.setdefault(c.cls, []).append(c)
    for v in by_cls.values():
        rng.shuffle(v)
    order, k = [], 0
    while any(k < len(v) for v in by_cls.values()):
        for cls in sorted(by_cls):
            if k < len(by_cls[cls]):
                order.append(by_cls[cls][k])
        k += 1
    return order

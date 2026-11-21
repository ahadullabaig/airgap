"""Contact sheets for hand-labelling ERA clips (shot changes, camera motion, overlays).

Each row is one clip: --frames evenly spaced frames (default 4: 0, 1/3, 2/3, end), then |first - last|
(grey, x3) which lights up edges everywhere when the camera has moved and only
moving objects when it has not.

ERA is academic-use only with no redistribution: write sheets to the scratchpad
or under data/, never into the repo or anywhere public.

Examples:
    # class-stratified random sample from the training split
    .venv/bin/python -I bench/era_sheets.py --sample 4 --split Tra --seed 0 --out <scratch>/sheets
    # specific clips (split/class/id)
    .venv/bin/python -I bench/era_sheets.py --clips Tra/Fire/Fire_003.mp4 --out <scratch>/flagged
"""

import argparse
import json
import random
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
THUMB = 192
ROWS = 6


def clip_path(era: Path, split: str, cls: str, vid: str) -> Path:
    return era / "Videos" / split / cls / vid.replace(".mp4", " .mp4")  # ERA file names end in " .mp4"


def frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        return [f.to_ndarray(format="rgb24") for f in c.decode(st)]


def row(path: Path, label: str, k: int = 4, thumb: int = THUMB) -> Image.Image:
    fr = frames(path)
    n = len(fr)
    picks = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    tiles = [Image.fromarray(fr[i]).resize((thumb, thumb)) for i in picks]
    g0 = np.asarray(Image.fromarray(fr[0]).convert("L"), dtype=np.int16)
    g1 = np.asarray(Image.fromarray(fr[-1]).convert("L"), dtype=np.int16)
    diff = np.clip(np.abs(g1 - g0) * 3, 0, 255).astype(np.uint8)
    tiles.append(Image.fromarray(diff).resize((thumb, thumb)).convert("RGB"))
    im = Image.new("RGB", (len(tiles) * (thumb + 4) - 4, thumb + 18), (255, 255, 255))
    for j, t in enumerate(tiles):
        im.paste(t, (j * (thumb + 4), 18))
    ImageDraw.Draw(im).text((2, 2), f"{label}   n={n}", fill=(0, 0, 0))
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", default=str(ROOT / "data" / "era"))
    ap.add_argument("--split", default="Tra")
    ap.add_argument("--sample", type=int, help="clips per class, drawn at random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", nargs="+", help="split/class/id entries")
    ap.add_argument("--frames", type=int, default=4, help="frames per row")
    ap.add_argument("--thumb", type=int, default=THUMB, help="thumbnail side in pixels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    era, out = Path(args.era), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.clips:
        items = [tuple(c.split("/")) for c in args.clips]
    else:
        rng = random.Random(args.seed)
        items = []
        for cdir in sorted((era / "Videos" / args.split).iterdir()):
            ids = sorted(p.name.replace(" .mp4", ".mp4") for p in cdir.glob("*.mp4"))
            items += [(args.split, cdir.name, v) for v in sorted(rng.sample(ids, args.sample))]
    (out / "index.json").write_text(json.dumps([{"row": i, "split": s, "cls": c, "id": v}
                                                for i, (s, c, v) in enumerate(items)], indent=0))
    for start in range(0, len(items), ROWS):
        chunk = items[start:start + ROWS]
        rows = [row(clip_path(era, *it), f"#{start + i}  {it[0]}/{it[1]}/{it[2]}", args.frames, args.thumb)
                for i, it in enumerate(chunk)]
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 6 * len(rows)), (200, 200, 200))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + 6
        sheet.save(out / f"sheet_{start // ROWS:02d}.jpg", quality=88)
    print(f"{len(items)} clips, {(len(items) + ROWS - 1) // ROWS} sheets in {out}")


if __name__ == "__main__":
    main()

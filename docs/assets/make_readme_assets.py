"""The README's images, drawn from committed results and the demo videos (DATA.md).

  banner-light.svg, banner-dark.svg   wordmark, text converted to paths
  hero.webp                           three chosen windows: one of the four frames the model saw, and its caption
  terminal.webp                       rendered from the run record of the evaluated surfing run (run.json)
  works-vs-misses.webp                one window described correctly and one that misses the event

Needs the cut demo videos (bash demo/prepare_videos.sh) and the Inter and JetBrains Mono fonts (Fedora packages
rsms-inter-vf-fonts and jetbrains-mono-fonts).

Example:
    .venv/bin/python -I docs/assets/make_readme_assets.py
"""
import json, sys, tempfile, textwrap
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets"
FONTS = Path(tempfile.mkdtemp())
from fontTools.varLib import instancer
for _w, _o, _n in ((700, 32, "DisplayBold"), (600, 14, "SemiBold"), (400, 14, "Regular"), (500, 14, "Medium")):
    instancer.instantiateVariableFont(TTFont("/usr/share/fonts/rsms-inter-vf-fonts/InterVariable.ttf"),
                                      {"wght": _w, "opsz": _o}).save(FONTS / f"Inter-{_n}.ttf")
sys.path.insert(0, str(ROOT / "demo"))
from sampling import sample_window

INTER = {k: str(FONTS / f"Inter-{k}.ttf") for k in ("DisplayBold", "SemiBold", "Regular", "Medium")}
MONO = "/usr/share/fonts/jetbrains-mono-fonts/JetBrainsMono-Regular.otf"
MONO_B = "/usr/share/fonts/jetbrains-mono-fonts/JetBrainsMono-Bold.otf"
F = lambda k, s: ImageFont.truetype(INTER[k], s)

# ---------------------------------------------------------------- banner (SVG, text converted to paths)
def word_path(text, size, x0, baseline):
    font = TTFont(INTER["DisplayBold"])
    gs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    s = size / font["head"].unitsPerEm
    pen = SVGPathPen(gs)
    x = 0
    for ch in text:
        g = cmap[ord(ch)]
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, x0 + x * s, baseline)))
        x += hmtx[g][0]
    return pen.getCommands(), x * s, font["OS/2"].sxHeight * s

def banner(fg, accent, bg, path):
    size, mark, gap, pad = 112, 112, 34, 8
    d, width, xh = word_path("airgap", size, 0, 0)
    W = int(pad * 2 + mark + gap + width)
    H = 150
    baseline = 112
    mx, my = pad, baseline - xh / 2 - mark / 2          # mark centred on the x-height band
    d, _, _ = word_path("airgap", size, pad + mark + gap, baseline)
    c = lambda v: round(v, 2)
    rot = [(20, 20), (92, 20), (20, 92), (92, 92)]
    k = mark / 112
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="airgap">',
           f'<g transform="translate({c(mx)} {c(my)}) scale({c(k)})">']
    for x, y in rot:
        svg.append(f'<line x1="56" y1="56" x2="{x}" y2="{y}" stroke="{fg}" stroke-width="8" stroke-linecap="round"/>')
    for x, y in rot:
        svg.append(f'<circle cx="{x}" cy="{y}" r="17" fill="{accent}" fill-opacity="0.16" stroke="{accent}" stroke-width="6"/>')
        svg.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{fg}"/>')
    svg.append(f'<rect x="36" y="36" width="40" height="40" rx="12" fill="{accent}"/>')
    svg.append(f'<circle cx="56" cy="56" r="7.5" fill="{bg}"/>')
    svg.append('</g>')
    svg.append(f'<path d="{d}" fill="{fg}"/>')
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n")

banner("#1f2328", "#0969da", "#ffffff", OUT / "banner-light.svg")
banner("#e6edf3", "#4493f8", "#0d1117", OUT / "banner-dark.svg")

# ---------------------------------------------------------------- shared helpers
def rounded(im, r, corners=(True, True, True, True)):
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.size[0] - 1, im.size[1] - 1), r, fill=255, corners=corners)
    out = Image.new("RGBA", im.size, (0, 0, 0, 0)); out.paste(im.convert("RGBA"), (0, 0), mask); return out

def frame(video, t, w, h):
    (_, img), = sample_window(str(ROOT / video), t - 0.02, 0.04, 1)
    iw, ih = img.size
    s = max(w / iw, h / ih)
    img = img.resize((round(iw * s), round(ih * s)), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))

def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur); cur = wd
    return lines + [cur]

INK, MUTED, BORDER, CARD = (31, 35, 40), (89, 99, 110), (208, 215, 222), (255, 255, 255)
ACCENT, AMBER, GREEN, RED = (9, 105, 218), (154, 103, 0), (26, 127, 55), (207, 34, 46)

def chip(d, x, y, text, fg, font, pad=(14, 7)):
    w = d.textlength(text, font=font)
    bbox = font.getbbox("Ag")
    h = bbox[3] - bbox[1] + 2 * pad[1]
    d.rounded_rectangle((x, y, x + w + 2 * pad[0], y + h), h // 2, fill=fg + (28,), outline=fg + (90,), width=2)
    d.text((x + pad[0], y + pad[1] - bbox[1]), text, font=font, fill=fg)
    return x + w + 2 * pad[0]

# ---------------------------------------------------------------- hero: three windows, the frame and the caption the model wrote
run = lambda v: json.loads((ROOT / "results" / "demo" / v / "run.json").read_text())
TILES = [("demo_surfing_dana_point", 2, 2, "Surfing, Dana Point"),
         ("demo_surfing_dana_point", 10, 0, "Surfing, Dana Point"),
         ("demo_or58_truck_recovery", 14, 2, "Truck recovery, OR 58")]
TW, TH, GAP, PADX = 600, 338, 36, 26
cap_font, meta_font, chip_font = F("Regular", 27), F("Medium", 23), F("SemiBold", 21)
probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
tiles = []
for v, k, fi, label in TILES:
    w = run(v)["windows"][k]
    lines = wrap(probe, "“" + w["caption"] + "”", cap_font, TW - 2 * PADX)
    tiles.append((v, w, fi, label, lines))
text_h = max(len(t[4]) for t in tiles) * 38 + 138
CH = TH + text_h
canvas = Image.new("RGBA", (3 * TW + 2 * GAP + 8, CH + 8), (0, 0, 0, 0))
for i, (v, w, fi, label, lines) in enumerate(tiles):
    x0, y0 = 4 + i * (TW + GAP), 4
    card = Image.new("RGB", (TW, CH), CARD)
    card.paste(frame(f"data/clips/{v}.mp4", w["frame_times_s"][fi], TW, TH), (0, 0))
    d = ImageDraw.Draw(card, "RGBA")
    m = lambda s: f"{round(s) // 60}:{round(s) % 60:02d}"
    x = chip(d, PADX, TH + 22, f"{m(w['start_s'])}–{m(w['end_s'])}", ACCENT, chip_font)
    if w["uncertain"]:
        chip(d, x + 10, TH + 22, "low confidence", AMBER, chip_font)
    for j, ln in enumerate(lines):
        d.text((PADX, TH + 78 + j * 38), ln, font=cap_font, fill=INK)
    d.text((PADX, CH - 26), label, font=meta_font, fill=MUTED, anchor="ls")
    card = rounded(card, 18)
    ImageDraw.Draw(card).rounded_rectangle((0, 0, TW - 1, CH - 1), 18, outline=BORDER, width=2)
    canvas.alpha_composite(card, (x0, y0))
canvas.save(OUT / "hero.webp", "WEBP", quality=90, method=6)

# ---------------------------------------------------------------- terminal capture (real output of one run, middle windows elided)
rec = run("demo_surfing_dana_point")
clk = lambda t: f"{round(t) // 60}:{round(t) % 60:02d}"
n = len(rec["windows"])
wins = [f"[{w['index'] + 1}/{n}] {clk(w['start_s'])}-{clk(w['end_s'])}  {w['decode_s'] + w['preprocess_s'] + w['generate_s']:.2f} s  "
        + ("uncertain  " if w["uncertain"] else "") + w["caption"] for w in rec["windows"]]
out_dir = "results/demo/demo_surfing_dana_point"
done = [f"wrote {out_dir}/report.md and", f"      {out_dir}/run.json in {rec['timing_s']['total']:.1f} s"]
mono, monob = ImageFont.truetype(MONO, 25), ImageFont.truetype(MONO_B, 25)
TW2, LH, LP = 1760, 38, 34
cmd = ["$ unshare -rn .venv/bin/python -I demo/run_mission.py \\", "      --video data/clips/demo_surfing_dana_point.mp4 --offline"]
body = ([("cmd", c) for c in cmd] + [("gap", "")] + [("win", l) for l in wins[:8]] + [("dim", "   \u22ee   windows 9\u201327")]
        + [("win", l) for l in wins[-3:]] + [("gap", "")] + [("done", l) for l in done])
rows = []
for kind, l in body:
    if kind == "win":
        head, rest = l.split("  ", 1)
        idx, span = head.split(" ", 1)
        t, cap = rest.split(" s  ", 1)
        unc = cap.startswith("uncertain  ")
        cap = cap.replace("uncertain  ", "", 1)
        pre = f"{idx:<8}{span:<11}{t.strip():>5}s  "
        parts = textwrap.wrap(cap, 72)
        rows.append(("win", pre, parts[0], unc))
        for p in parts[1:]:
            rows.append(("cont", " " * len(pre), p, False))
    else:
        rows.append((kind, l, "", False))
H2 = 70 + LP * 2 + LH * len(rows)
term = Image.new("RGBA", (TW2, H2), (13, 17, 23, 255))
d = ImageDraw.Draw(term)
d.rectangle((0, 0, TW2, 62), fill=(22, 27, 34))
for j, col in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
    d.ellipse((28 + j * 34, 21, 48 + j * 34, 41), fill=col)
d.text((TW2 // 2, 31), "airgap — offline run", font=F("Medium", 23), fill=(139, 148, 158), anchor="mm")
y = 62 + LP
for kind, a, b, unc in rows:
    if kind == "cmd":
        d.text((LP, y), a[:2], font=monob, fill=(63, 185, 80)); d.text((LP + d.textlength(a[:2], font=mono), y), a[2:], font=mono, fill=(230, 237, 243))
    elif kind == "dim":
        d.text((LP, y), a, font=mono, fill=(110, 118, 129))
    elif kind == "done":
        d.text((LP, y), a, font=mono, fill=(63, 185, 80))
    elif kind in ("win", "cont"):
        if kind == "win":
            idx, rest = a[:8], a[8:]
            d.text((LP, y), idx, font=mono, fill=(110, 118, 129))
            x = LP + d.textlength(idx, font=mono)
            d.text((x, y), rest[:11], font=mono, fill=(121, 192, 255))
            x += d.textlength(rest[:11], font=mono)
            d.text((x, y), rest[11:], font=mono, fill=(110, 118, 129))
        x = LP + d.textlength(a, font=mono)
        d.text((x, y), b, font=mono, fill=(230, 237, 243))
        if unc:
            chip_x = x + d.textlength(b, font=mono) + 12
            d.text((chip_x, y), "uncertain", font=monob, fill=(210, 153, 34))
    y += LH
term = rounded(term, 20)
ImageDraw.Draw(term).rounded_rectangle((0, 0, TW2 - 1, H2 - 1), 20, outline=(48, 54, 61), width=2)
term.save(OUT / "terminal.webp", "WEBP", quality=90, method=6)

# ---------------------------------------------------------------- where it works, and where it misses the event
ROWS = [("demo_surfing_dana_point", 16, "Gets it right", GREEN, "Surfing, Dana Point",
         "A surfer rides a wave while others wait in the water."),
        ("demo_minnesota_flood", 17, "Misses the event", RED, "River flooding, Highway 41",
         "Floodwater spreads through the woods, with trees standing in it. The report never mentions a flood.")]
FW, FH, FG, PX = 452, 254, 12, 28
W3 = 4 * FW + 3 * FG + 2 * PX
cap_font2, truth_font = F("Regular", 28), F("Regular", 26)
blocks = []
for v, k, tag, col, label, truth in ROWS:
    w = run(v)["windows"][k]
    lines = wrap(probe, "Model: “" + w["caption"] + "”", cap_font2, W3 - 2 * PX)
    blocks.append((v, w, tag, col, label, truth, lines))
heights = [86 + FH + 26 + len(b[6]) * 40 + 46 + 24 for b in blocks]
canvas = Image.new("RGBA", (W3 + 8, sum(heights) + 30 + 8), (0, 0, 0, 0))
y0 = 4
for (v, w, tag, col, label, truth, lines), h in zip(blocks, heights):
    card = Image.new("RGB", (W3, h), CARD)
    d = ImageDraw.Draw(card, "RGBA")
    x = chip(d, PX, 24, tag, col, F("SemiBold", 24))
    m = lambda s: f"{round(s) // 60}:{round(s) % 60:02d}"
    d.text((x + 16, 31), f"{label}  ·  {m(w['start_s'])}–{m(w['end_s'])}  ·  the four frames the model saw", font=F("Medium", 24), fill=MUTED)
    for i, t in enumerate(w["frame_times_s"]):
        fr = rounded(frame(f"data/clips/{v}.mp4", t, FW, FH), 10)
        card.paste(fr, (PX + i * (FW + FG), 86), fr)
    yy = 86 + FH + 26
    for j, ln in enumerate(lines):
        d.text((PX, yy + j * 40), ln, font=cap_font2, fill=INK)
    d.text((PX, yy + len(lines) * 40 + 8), "Actually: " + truth, font=truth_font, fill=MUTED)
    card = rounded(card, 18)
    ImageDraw.Draw(card).rounded_rectangle((0, 0, W3 - 1, h - 1), 18, outline=BORDER, width=2)
    canvas.alpha_composite(card, (4, y0))
    y0 += h + 30
canvas.save(OUT / "works-vs-misses.webp", "WEBP", quality=90, method=6)
print("written:", sorted(p.name for p in OUT.iterdir()))

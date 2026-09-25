"""Generate examples/datasets/shapes_dataset.zip, a deliberately messy image dataset.

Three classes of drawn shapes with planted problems (the evaluation set checks that
MOSAIC finds them):
- class imbalance: circles 60, squares 36, triangles 14
- 4 exact duplicate copies and 4 near-duplicates (resized and re-encoded)
- 2 cross-class duplicates (the same square saved under circles/ and squares/)
- 3 mislabeled images: unique squares saved under circles/
- 5 blurry, 3 very dark, 1 blank image
- 3 corrupt (truncated) files and 1 tiny 8x8 image
- 2 PNGs with transparency, 1 grayscale image, 2 JPEGs that need EXIF rotation
- a README.txt, which should be treated as a note rather than data
"""

from __future__ import annotations

import io
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).parent / "datasets" / "shapes_dataset.zip"
SIZE = 128
PALETTE = [(181, 71, 27), (47, 111, 115), (165, 122, 0), (122, 62, 101), (94, 127, 51)]


def shape(kind: str, rng: random.Random) -> Image.Image:
    # a random two-color gradient background, so images don't look alike
    top = [rng.randint(170, 250) for _ in range(3)]
    bottom = [rng.randint(170, 250) for _ in range(3)]
    im = Image.new("RGB", (SIZE, SIZE))
    d = ImageDraw.Draw(im)
    for y in range(SIZE):
        t = y / (SIZE - 1)
        d.line(
            [(0, y), (SIZE, y)],
            fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom, strict=True)),
        )
    for _ in range(rng.randint(2, 5)):  # small distractor marks
        x, y, r = rng.randrange(SIZE), rng.randrange(SIZE), rng.randint(2, 7)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(rng.randint(90, 200) for _ in range(3)))
    color = rng.choice(PALETTE)
    s = rng.randint(18, 50)
    cx, cy = rng.randint(s + 2, SIZE - s - 2), rng.randint(s + 2, SIZE - s - 2)
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    if kind == "circles":
        ld.ellipse([cx - s, cy - s, cx + s, cy + s], fill=color)
    elif kind == "squares":
        ld.rectangle([cx - s, cy - s, cx + s, cy + s], fill=color)
    else:
        ld.polygon([(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)], fill=color)
    if kind != "circles":
        layer = layer.rotate(rng.uniform(-35, 35), center=(cx, cy))
    im.paste(layer, mask=layer)
    return im


def jpeg(im: Image.Image, quality: int = 90, exif: Image.Exif | None = None) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, **({"exif": exif} if exif else {}))
    return buf.getvalue()


def png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def main(seed: int = 7) -> Path:
    rng = random.Random(seed)
    files: dict[str, bytes] = {}
    counts = {"circles": 60, "squares": 36, "triangles": 14}
    images: dict[str, list[tuple[str, Image.Image]]] = {k: [] for k in counts}
    for kind, n in counts.items():
        for i in range(n):
            im = shape(kind, rng)
            name = f"shapes/{kind}/{kind[:-1]}_{i:03d}.jpg"
            images[kind].append((name, im))
            files[name] = jpeg(im)

    circles, squares = images["circles"], images["squares"]
    # planted quality problems (modify existing files so class counts stay the same)
    for name, im in circles[:3] + squares[:2]:
        files[name] = jpeg(im.filter(ImageFilter.GaussianBlur(6)))
    for name, im in circles[3:5] + images["triangles"][:1]:
        files[name] = jpeg(Image.eval(im, lambda v: v // 12))
    files[circles[5][0]] = jpeg(Image.new("RGB", (SIZE, SIZE), (240, 240, 240)))
    for name, _ in circles[6:8] + squares[2:3]:
        files[name] = files[name][: len(files[name]) // 3]  # truncated = corrupt
    files[circles[8][0]] = jpeg(Image.new("RGB", (8, 8), (200, 60, 30)))  # tiny

    # formats and metadata
    for name, im in squares[3:5]:
        rgba = im.convert("RGBA")
        rgba.putalpha(200)
        del files[name]
        files[name.replace(".jpg", ".png")] = png(rgba)
    name, im = images["triangles"][1]
    files[name] = jpeg(im.convert("L"))
    for name, im in circles[9:11]:
        exif = Image.Exif()
        exif[0x0112] = 6  # "rotate 90 degrees" in EXIF orientation
        files[name] = jpeg(im.rotate(90, expand=True), exif=exif)

    # duplicates
    for name, _ in circles[12:14] + squares[6:8]:
        files[name.replace(".jpg", "_copy.jpg")] = files[name]
    for name, im in circles[14:16] + squares[8:10]:
        files[name.replace(".jpg", "_resized.jpg")] = jpeg(im.resize((100, 100)), quality=70)
    for name, _ in squares[10:12]:
        files["shapes/circles/" + name.split("/")[-1]] = files[name]  # cross-class duplicates

    # mislabeled: unique squares saved under circles/
    for i in range(3):
        files[f"shapes/circles/circle_extra_{i}.jpg"] = jpeg(shape("squares", rng))

    files["shapes/README.txt"] = b"Toy shapes dataset for MOSAIC EDA. Classes are folder names.\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.writestr(name, files[name])
    return OUT


if __name__ == "__main__":
    print(main())

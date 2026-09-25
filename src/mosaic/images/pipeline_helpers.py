"""Helper functions for image cleaning.

This module's source is copied verbatim into every exported cleaning_pipeline.py for
images, so it must only import the standard library, numpy, pandas, and Pillow.
One row per image: metrics are computed once, cleaning operations filter rows or set
transform flags, and export_images() writes the cleaned copies.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 60_000_000  # larger images are treated as unsafe (decompression bombs)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
TRANSFORM_DEFAULTS = {
    "fix_orientation": False,
    "to_rgb": False,
    "max_side": 0,
    "strip_exif": False,
    "suspected_mislabel": False,
}

try:  # HEIC support is optional
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


def _dct_matrix(n=32):
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


_DCT = _dct_matrix()


def phash(im):
    """64-bit perceptual hash (hex): signs of the low-frequency DCT coefficients."""
    a = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), float)
    low = (_DCT @ a @ _DCT.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    # a hex string, not an int: pandas would turn 64-bit ints into floats and lose bits
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def thumb_signature(im):
    """8x8 grayscale thumbnail as hex: a direct pixel check to confirm hash matches."""
    a = np.asarray(im.convert("L").resize((8, 8), Image.Resampling.LANCZOS), np.uint8)
    return a.tobytes().hex()


def thumb_distance(a, b):
    """Mean absolute pixel difference (0-255) between two thumbnail signatures."""
    x = np.frombuffer(bytes.fromhex(a), np.uint8).astype(int)
    y = np.frombuffer(bytes.fromhex(b), np.uint8).astype(int)
    return float(np.abs(x - y).mean())


def blur_score(gray):
    """Variance of the Laplacian: low values mean few sharp edges (a blurry image)."""
    lap = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var()) if lap.size else 0.0


def image_record(path, rel, label):
    path = Path(path)
    data = path.read_bytes()
    rec = {
        "path": rel,
        "class": label,
        "file_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "corrupt": False,
        "error": "",
    }
    try:
        with Image.open(path) as im:
            im.verify()  # structural check; the file must be reopened afterwards
        with Image.open(path) as im:
            im.load()
            rec.update(
                format=im.format or "",
                mode=im.mode,
                width=im.width,
                height=im.height,
                frames=getattr(im, "n_frames", 1),
                exif_orientation=int(im.getexif().get(0x0112, 1) or 1),
                has_alpha=im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info,
            )
            rgb = im.convert("RGB")
            small = rgb.copy()
            small.thumbnail((256, 256))
            arr = np.asarray(small, float)
            gray = arr.mean(axis=2)
            rec.update(
                brightness=round(float(gray.mean()), 2),
                contrast=round(float(gray.std()), 2),
                blur=round(blur_score(gray), 2),
                grayscale=bool(np.abs(arr - arr.mean(axis=2, keepdims=True)).mean() < 2),
                phash=phash(rgb),
                thumb=thumb_signature(rgb),
            )
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        rec.update(corrupt=True, error=f"{type(exc).__name__}: {str(exc)[:120]}")
    return rec


def list_images(folder):
    """(relative path, class) for every image; the class is the first folder level."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    rels = [p.relative_to(folder).parts for p in files]
    tops = {r[0] for r in rels if len(r) > 1}
    skip = 1 if len(tops) == 1 and all(len(r) > 2 for r in rels) else 0  # one wrapper folder
    return [("/".join(r), r[skip] if len(r) > skip + 1 else "") for r in rels]


def build_table(folder, files=None):
    """One row per image with its metrics. `files` is a list of (relative path, class)."""
    folder = Path(folder)
    rows = [image_record(folder / rel, rel, label) for rel, label in (files or list_images(folder))]
    df = pd.DataFrame(rows)
    for name, default in TRANSFORM_DEFAULTS.items():
        df[name] = default
    return df


def near_groups(df, max_distance=6, max_pixel_diff=8.0):
    """Group near-duplicate images.

    Two images match when their perceptual hashes differ in at most max_distance bits
    AND their 8x8 thumbnails differ by at most max_pixel_diff on average (0-255).
    The hash finds candidates quickly; the pixel check confirms them.
    """
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ok = df["phash"].notna().to_numpy() if "phash" in df else np.zeros(len(df), bool)
    idx = np.flatnonzero(ok)
    hashes = np.array([int(h, 16) for h in df["phash"].to_numpy()[idx]], dtype=np.uint64)
    if len(hashes) > 1:
        xor = np.bitwise_xor.outer(hashes, hashes)
        dist = np.unpackbits(xor.view(np.uint8), axis=-1).reshape(len(hashes), len(hashes), -1)
        dist = dist.sum(axis=-1)
        thumbs = df["thumb"].to_numpy()[idx]
        for a, b in zip(*np.nonzero(np.triu(dist <= max_distance, k=1)), strict=True):
            if thumb_distance(thumbs[a], thumbs[b]) <= max_pixel_diff:
                parent[find(idx[a])] = find(idx[b])
    return pd.Series([find(i) for i in range(len(df))], index=df.index)


def drop_near_duplicates(df, max_distance=6):
    groups = near_groups(df, max_distance)
    return df[~groups.duplicated()].reset_index(drop=True)


def cross_class_mask(df, max_distance=6):
    """Images whose near-duplicates appear under more than one class (ambiguous labels)."""
    groups = near_groups(df, max_distance)
    classes_per_group = df.groupby(groups)["class"].nunique()
    return groups.map(classes_per_group).gt(1).to_numpy()


def export_images(df, source_folder, out_folder):
    """Write the kept images, applying the transform flags, plus a manifest CSV."""
    source_folder, out_folder = Path(source_folder), Path(out_folder)
    for row in df.itertuples(index=False):
        if row.corrupt:
            continue
        target = out_folder / row.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_folder / row.path) as im:
            fmt = im.format
            if row.fix_orientation:
                im = ImageOps.exif_transpose(im)
            if row.to_rgb and im.mode != "RGB":
                background = Image.new("RGB", im.size, "white")
                rgba = im.convert("RGBA")
                background.paste(rgba, mask=rgba.split()[-1])
                im = background
            if row.max_side:
                im.thumbnail((int(row.max_side), int(row.max_side)))
            params = {} if row.strip_exif else {"exif": im.getexif()}
            if fmt == "JPEG" and im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.save(target, format=fmt, **params)
    keep = [c for c in df.columns if c not in ("phash", "thumb")]
    df[keep].to_csv(out_folder / "image_manifest.csv", index=False)
    return out_folder

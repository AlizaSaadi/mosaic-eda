"""Profile an image dataset in code and save every result to the evidence store."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageOps

from mosaic.evidence.store import EvidenceStore
from mosaic.images.pipeline_helpers import cross_class_mask, near_groups
from mosaic.tables.profile import r4
from mosaic.ui.palette import HARVEST

DARK = 25.0
BRIGHT = 235.0
BLANK_CONTRAST = 4.0
TINY_SIDE = 32
BLUR_FRACTION = 0.25  # blurry = below 25% of the dataset's median blur score
SHEET_COLS = 8
SHEET_ROWS = 5
THUMB = 96


@dataclass
class ImageProfile:
    artifact_ids: list[str] = field(default_factory=list)
    chart_ids: list[str] = field(default_factory=list)
    sheet_ids: list[str] = field(default_factory=list)
    quality: float = 0.0
    blur_threshold: float = 0.0
    categories: dict[str, list[str]] = field(default_factory=dict)


def _stats(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {}
    return {"min": r4(values.min()), "median": r4(values.median()), "max": r4(values.max())}


def categorize(df: pd.DataFrame) -> tuple[dict[str, list[str]], float]:
    """Give each problem image one main category, so a dark image isn't also 'blurry'."""
    ok = df[~df["corrupt"]]
    blur_threshold = round(float(ok["blur"].median()) * BLUR_FRACTION, 2) if len(ok) else 0.0
    cats: dict[str, list[str]] = {"corrupt": df.loc[df["corrupt"], "path"].tolist()}
    taken = set(cats["corrupt"])
    rules = [
        ("tiny", ok[["width", "height"]].min(axis=1) < TINY_SIDE),
        ("too_dark", ok["brightness"] < DARK),  # before near_blank: dark images lack contrast too
        ("too_bright", ok["brightness"] > BRIGHT),
        ("near_blank", ok["contrast"] < BLANK_CONTRAST),
        ("blurry", ok["blur"] < blur_threshold),
    ]
    for name, mask in rules:
        files = [p for p in ok.loc[mask, "path"] if p not in taken]
        cats[name] = files
        taken.update(files)
    return cats, blur_threshold


def _fig(fig: go.Figure, title: str) -> dict:
    fig.update_layout(
        title=title,
        template="plotly_white",
        colorway=HARVEST["chart"],
        height=340,
        font={"family": "Inter, system-ui, sans-serif", "color": HARVEST["text"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    return json.loads(fig.to_json())


def contact_sheets(df: pd.DataFrame, root: Path, out: Path, per_class_sheets: int = 2):
    """Numbered grids of thumbnails, one or more per class, for the vision model and the report."""
    out.mkdir(parents=True, exist_ok=True)
    per_sheet = SHEET_COLS * SHEET_ROWS
    sheets = []
    ok = df[~df["corrupt"]]
    for label, group in ok.groupby("class", sort=True):
        paths = group["path"].tolist()
        for s in range(min(per_class_sheets, -(-len(paths) // per_sheet))):
            chunk = paths[s * per_sheet : (s + 1) * per_sheet]
            rows = -(-len(chunk) // SHEET_COLS)
            sheet = Image.new("RGB", (SHEET_COLS * THUMB, rows * THUMB), "white")
            draw = ImageDraw.Draw(sheet)
            index = {}
            for i, rel in enumerate(chunk, 1):
                with Image.open(root / rel) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((THUMB - 4, THUMB - 4))
                x, y = ((i - 1) % SHEET_COLS) * THUMB, ((i - 1) // SHEET_COLS) * THUMB
                sheet.paste(im, (x + 2, y + 2))
                draw.rectangle([x + 2, y + 2, x + 24, y + 16], fill="black")
                draw.text((x + 4, y + 3), str(i), fill="white")
                index[i] = rel
            name = f"sheet_{label or 'all'}_{s + 1}.png".replace("/", "_")
            sheet.save(out / name)
            sheets.append({"class": label or "(no class)", "path": str(out / name), "index": index})
    return sheets


def profile_images(
    df: pd.DataFrame,
    store: EvidenceStore,
    *,
    root: Path,
    sheets_dir: Path,
    class_counts: dict[str, int],
    total_images: int,
    stage: str = "raw",
) -> ImageProfile:
    result = ImageProfile()
    ok = df[~df["corrupt"]]
    cats, result.blur_threshold = categorize(df)
    result.categories = cats
    min_side = ok[["width", "height"]].min(axis=1) if len(ok) else pd.Series(dtype=float)

    overview = {
        "images": total_images,
        "analyzed": len(df),
        "classes": len([c for c in class_counts if c]),
        "formats": dict(Counter(ok["format"])),
        "modes": dict(Counter(ok["mode"])),
        "corrupt": len(cats["corrupt"]),
        "width": _stats(ok["width"]),
        "height": _stats(ok["height"]),
        "exif_rotated": int((ok["exif_orientation"] != 1).sum()),
        "has_alpha": int(ok["has_alpha"].sum()),
        "grayscale": int(ok["grayscale"].sum()),
        "animated": int((ok["frames"] > 1).sum()),
        "tiny": int((min_side < TINY_SIDE).sum()),
    }
    sample_note = "all analyzed" if len(df) >= total_images else f"{len(df)} sampled"
    a = store.add(
        "img_overview",
        "profile",
        "image_inventory",
        f"Images ({stage}): {total_images} ({sample_note}); {overview['classes']} classes; "
        f"formats {overview['formats']}; modes {overview['modes']}; {overview['corrupt']} corrupt "
        f"(corrupt); widths {overview['width']}; {overview['exif_rotated']} need EXIF rotation "
        f"(exif_rotated); {overview['has_alpha']} with transparency (has_alpha); "
        f"{overview['grayscale']} grayscale; {overview['tiny']} under {TINY_SIDE}px (tiny).",
        overview,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    counts = {k: len(v) for k, v in cats.items()}
    quality = {
        "counts": counts,
        "files": {k: v[:20] for k, v in cats.items()},
        "blur_threshold": result.blur_threshold,
        "brightness": _stats(ok["brightness"]),
        "contrast": _stats(ok["contrast"]),
        "blur": _stats(ok["blur"]),
        "per_class": {
            str(label): {
                "count": len(g),
                "brightness_mean": r4(g["brightness"].mean()),
                "blur_median": r4(g["blur"].median()),
            }
            for label, g in ok.groupby("class")
        },
    }
    lines = "; ".join(f"{k} {n}" for k, n in counts.items() if k != "corrupt")
    listed = "; ".join(f"{k}: {', '.join(v[:6])}" for k, v in cats.items() if v and k != "corrupt")
    a = store.add(
        "img_quality",
        "profile",
        "image_quality",
        f"Image quality ({stage}), one main problem per image: {lines} (keys counts.<category>). "
        f"Blur threshold {result.blur_threshold} (blur_threshold). Files: {listed or 'none'}.",
        quality,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    exact_groups = df[df.duplicated("sha256", keep=False)].groupby("sha256")["path"].apply(list)
    groups = near_groups(df)
    near = df.groupby(groups)["path"].apply(list)
    near = [g for g in near if len(g) > 1]
    cross = df.loc[cross_class_mask(df), "path"].tolist()
    dupes = {
        "exact_groups": len(exact_groups),
        "exact_extra_copies": int(sum(len(g) - 1 for g in exact_groups)),
        "near_groups": len(near),
        "near_extra_copies": int(sum(len(g) - 1 for g in near)),
        "cross_class_files": len(cross),
        "cross_class": cross[:20],
        "examples": near[:10],
    }
    a = store.add(
        "img_dupes",
        "profile",
        "duplicates",
        f"Duplicates ({stage}): {dupes['exact_groups']} exact groups with "
        f"{dupes['exact_extra_copies']} extra copies (exact_extra_copies); {dupes['near_groups']} "
        f"near-duplicate groups including exact ones, {dupes['near_extra_copies']} extra copies "
        f"(near_extra_copies); {len(cross)} files duplicated across classes (cross_class_files): "
        f"{', '.join(cross[:8]) or 'none'}. Groups: {near[:6]}",
        dupes,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    labelled = {k: v for k, v in class_counts.items() if k}
    if labelled:
        total = sum(labelled.values())
        balance = {
            "classes": {k: {"count": v, "share": r4(100 * v / total)} for k, v in labelled.items()},
            "imbalance_ratio": r4(max(labelled.values()) / max(min(labelled.values()), 1)),
        }
        text = ", ".join(f"{k} {v['count']} ({v['share']}%)" for k, v in balance["classes"].items())
        a = store.add(
            "img_balance",
            "profile",
            "class_balance",
            f"Class balance from folder names (all {total} files): {text}; largest/smallest ratio "
            f"{balance['imbalance_ratio']} (imbalance_ratio). Keys: classes.<name>.count/.share.",
            balance,
            {"stage": stage},
        )
        result.artifact_ids.append(a.id)
        fig = go.Figure(
            go.Bar(x=list(labelled), y=list(labelled.values()), marker_color=HARVEST["accent"])
        )
        result.chart_ids.append(
            store.add(
                "chart",
                "chart",
                "class_balance",
                f"Chart: images per class ({stage})",
                {"figure": _fig(fig, "Images per class")},
            ).id
        )

    n = max(len(df), 1)
    bad = sum(counts.values())
    imbalance = labelled and max(labelled.values()) / max(min(labelled.values()), 1)
    parts = {
        "corrupt": -min(20.0, 100 * counts["corrupt"] / n * 2),
        "duplicates": -min(20.0, 100 * dupes["near_extra_copies"] / n * 1.5),
        "low_quality": -min(25.0, 100 * (bad - counts["corrupt"]) / n * 1.5),
        "class_imbalance": -min(15.0, max(0.0, (imbalance or 1) - 1.5) * 3),
        "cross_class": -min(10.0, len(cross) * 2.0),
    }
    parts = {k: round(v, 1) for k, v in parts.items()}
    score = round(max(0.0, 100 + sum(parts.values())), 1)
    result.quality = score
    a = store.add(
        "img_quality_score",
        "profile",
        "quality_score",
        f"Image data quality score ({stage}): {score}/100 (score); deductions {parts}.",
        {"score": score, "parts": parts},
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    for column, title, color in (("brightness", "Brightness", 1), ("blur", "Blur score (log)", 2)):
        values = ok[column].dropna()
        if column == "blur":
            values = np.log10(values.clip(lower=1))
        counts_, edges = np.histogram(values, bins=20)
        fig = go.Figure(
            go.Bar(
                x=(edges[:-1] + edges[1:]) / 2,
                y=counts_,
                width=np.diff(edges),
                marker_color=HARVEST["chart"][color],
            )
        )
        result.chart_ids.append(
            store.add(
                "chart",
                "chart",
                f"{column}_hist",
                f"Chart: {title} ({stage})",
                {"figure": _fig(fig, title)},
            ).id
        )
    fig = go.Figure(
        go.Scatter(
            x=ok["width"],
            y=ok["height"],
            mode="markers",
            marker={"color": HARVEST["chart"][3], "size": 7, "opacity": 0.7},
            text=ok["path"],
        )
    )
    fig.update_xaxes(title="width (px)")
    fig.update_yaxes(title="height (px)")
    result.chart_ids.append(
        store.add(
            "chart",
            "chart",
            "sizes",
            f"Chart: image sizes ({stage})",
            {"figure": _fig(fig, "Image sizes")},
        ).id
    )

    if stage == "raw":
        for sheet in contact_sheets(df, root, sheets_dir):
            a = store.add(
                "img_sheet",
                "image",
                "contact_sheet",
                f"Contact sheet: class {sheet['class']}, {len(sheet['index'])} images",
                sheet,
            )
            result.sheet_ids.append(a.id)
    return result

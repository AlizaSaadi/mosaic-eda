"""The allowed cleaning operations for images.

Images are cleaned through a table of files (one row per image, with its metrics).
Operations either remove rows (dropping images) or set transform flags that
export_images() applies when writing the cleaned copies. Like table operations,
each one is a code template, so the exported pipeline is exactly what ran.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from mosaic.images import pipeline_helpers
from mosaic.tables.ops import NoParams, OpSpec, Risk


class DistanceParams(NoParams):
    max_distance: int = Field(6, ge=0, le=16)


class MinSideParams(NoParams):
    min_side: int = Field(32, ge=1, le=4096)


class BlurParams(NoParams):
    min_blur: float = Field(ge=0)


class DarkParams(NoParams):
    max_brightness: float = Field(20, ge=0, le=128)


class BrightParams(NoParams):
    min_brightness: float = Field(240, ge=128, le=255)


class ContrastParams(NoParams):
    min_contrast: float = Field(4, ge=0, le=64)


class FilesParams(NoParams):
    files: list[str] = Field(min_length=1, max_length=500)


class MaxSideParams(NoParams):
    max_side: int = Field(512, ge=32, le=4096)


# Paths must exist in the original dataset (a file removed by an earlier step is fine)
_CHECK_FILES = (
    "_unknown = sorted(set({files!r}) - ORIGINAL_PATHS)\n"
    "assert not _unknown, f'unknown files (use exact paths from the evidence): {{_unknown[:5]}}'\n"
)

IMAGE_OPS: dict[str, OpSpec] = {}


def _op(name, risk, description, params, template):
    IMAGE_OPS[name] = OpSpec(name, risk, description, params, template, needs_columns=False)


_op(
    "remove_corrupt",
    Risk.DESTRUCTIVE,
    "Remove files that can't be opened as images.",
    NoParams,
    lambda c, p: "df = df[~df['corrupt']].reset_index(drop=True)",
)
_op(
    "drop_exact_duplicates",
    Risk.DESTRUCTIVE,
    "Keep one copy of byte-identical files.",
    NoParams,
    lambda c, p: "df = df.drop_duplicates(subset='sha256').reset_index(drop=True)",
)
_op(
    "drop_near_duplicates",
    Risk.DESTRUCTIVE,
    "Keep one image per group of near-identical images (resized or re-encoded copies).",
    DistanceParams,
    lambda c, p: f"df = drop_near_duplicates(df, max_distance={p.max_distance})",
)
_op(
    "drop_cross_class_duplicates",
    Risk.DESTRUCTIVE,
    "Remove images whose duplicates appear under more than one class (ambiguous labels).",
    DistanceParams,
    lambda c, p: (
        f"df = df[~cross_class_mask(df, max_distance={p.max_distance})].reset_index(drop=True)"
    ),
)
_op(
    "drop_too_small",
    Risk.DESTRUCTIVE,
    "Remove images whose shorter side is below min_side.",
    MinSideParams,
    lambda c, p: (
        f"df = df[df[['width', 'height']].min(axis=1).fillna(0) >= {p.min_side}]"
        ".reset_index(drop=True)"
    ),
)
_op(
    "drop_blurry",
    Risk.DESTRUCTIVE,
    "Remove images whose blur score (Laplacian variance) is below min_blur.",
    BlurParams,
    lambda c, p: f"df = df[df['blur'].fillna(0) >= {p.min_blur!r}].reset_index(drop=True)",
)
_op(
    "drop_too_dark",
    Risk.DESTRUCTIVE,
    "Remove images with mean brightness at or below the value.",
    DarkParams,
    lambda c, p: (
        f"df = df[df['brightness'].fillna(0) > {p.max_brightness!r}].reset_index(drop=True)"
    ),
)
_op(
    "drop_too_bright",
    Risk.DESTRUCTIVE,
    "Remove images with mean brightness at or above the value.",
    BrightParams,
    lambda c, p: (
        f"df = df[df['brightness'].fillna(255) < {p.min_brightness!r}].reset_index(drop=True)"
    ),
)
_op(
    "drop_near_blank",
    Risk.DESTRUCTIVE,
    "Remove nearly uniform images (contrast below the value).",
    ContrastParams,
    lambda c, p: f"df = df[df['contrast'].fillna(0) >= {p.min_contrast!r}].reset_index(drop=True)",
)
_op(
    "drop_files",
    Risk.DESTRUCTIVE,
    "Remove specific files, for example confirmed mislabels.",
    FilesParams,
    lambda c, p: (
        _CHECK_FILES.format(files=p.files)
        + f"df = df[~df['path'].isin({p.files!r})].reset_index(drop=True)"
    ),
)
_op(
    "flag_suspected_mislabels",
    Risk.SAFE,
    "Mark files for human review in the manifest (nothing is removed).",
    FilesParams,
    lambda c, p: (
        _CHECK_FILES.format(files=p.files)
        + f"df['suspected_mislabel'] = df['suspected_mislabel'] | df['path'].isin({p.files!r})"
    ),
)
_op(
    "fix_exif_orientation",
    Risk.SAFE,
    "Rotate images upright using their EXIF orientation tag.",
    NoParams,
    lambda c, p: "df['fix_orientation'] = True",
)
_op(
    "convert_to_rgb",
    Risk.SAFE,
    "Convert grayscale, palette, and transparent images to RGB (transparency on white).",
    NoParams,
    lambda c, p: "df['to_rgb'] = True",
)
_op(
    "resize_max_side",
    Risk.LOSSY,
    "Shrink images so the longer side is at most max_side.",
    MaxSideParams,
    lambda c, p: f"df['max_side'] = {p.max_side}",
)
_op(
    "strip_exif",
    Risk.SAFE,
    "Remove EXIF metadata (camera details, GPS) when exporting.",
    NoParams,
    lambda c, p: "df['strip_exif'] = True",
)


def image_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"pd": pd}
    ns.update({k: v for k, v in vars(pipeline_helpers).items() if not k.startswith("_")})
    return ns

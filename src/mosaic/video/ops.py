"""The allowed cleaning operations for video.

Like images and audio, video is cleaned through a table of files (one row per video).
Operations remove rows or set transform flags that export_video() applies with ffmpeg,
and each one is a code template, so the exported pipeline is exactly what ran.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from mosaic.images.ops import _CHECK_FILES, FilesParams
from mosaic.tables.ops import NoParams, OpSpec, Risk
from mosaic.video import pipeline_helpers


class NearParams(NoParams):
    max_distance: float = Field(6.0, ge=0, le=30)


class ShortParams(NoParams):
    min_seconds: float = Field(1.0, ge=0.1, le=600)


class MotionParams(NoParams):
    min_motion: float = Field(1.0, ge=0, le=50)


class DarkParams(NoParams):
    min_brightness: float = Field(16.0, ge=0, le=128)


class HeightParams(NoParams):
    max_height: int = Field(720, ge=90, le=4320)


class TrimParams(NoParams):
    max_seconds: float = Field(60.0, ge=1, le=3600)


class FpsParams(NoParams):
    fps: float = Field(1.0, gt=0, le=30)


VIDEO_OPS: dict[str, OpSpec] = {}


def _op(name, risk, description, params, template):
    VIDEO_OPS[name] = OpSpec(name, risk, description, params, template, needs_columns=False)


_drop = ".reset_index(drop=True)"
_op(
    "remove_corrupt",
    Risk.DESTRUCTIVE,
    "Remove files that can't be decoded as video.",
    NoParams,
    lambda c, p: f"df = df[~df['corrupt']]{_drop}",
)
_op(
    "drop_exact_duplicates",
    Risk.DESTRUCTIVE,
    "Keep one copy of byte-identical files.",
    NoParams,
    lambda c, p: f"df = df.drop_duplicates(subset='sha256'){_drop}",
)
_op(
    "drop_near_duplicates",
    Risk.DESTRUCTIVE,
    "Keep one video per group of re-encoded, resized, or rotated copies of the same footage.",
    NearParams,
    lambda c, p: f"df = drop_near_duplicates(df, max_distance={p.max_distance!r})",
)
_op(
    "drop_too_short",
    Risk.DESTRUCTIVE,
    "Remove videos shorter than min_seconds.",
    ShortParams,
    lambda c, p: f"df = df[df['duration_s'].fillna(0) >= {p.min_seconds!r}]{_drop}",
)
_op(
    "drop_black",
    Risk.DESTRUCTIVE,
    "Remove videos whose average brightness is below min_brightness (black footage).",
    DarkParams,
    lambda c, p: f"df = df[df['brightness'].fillna(0) >= {p.min_brightness!r}]{_drop}",
)
_op(
    "drop_frozen",
    Risk.DESTRUCTIVE,
    "Remove videos whose picture barely changes (motion below min_motion, 0-255 scale).",
    MotionParams,
    lambda c, p: f"df = df[df['motion'].fillna(0) >= {p.min_motion!r}]{_drop}",
)
_op(
    "drop_without_audio",
    Risk.DESTRUCTIVE,
    "Remove videos that have no audio track (only when the goal needs sound).",
    NoParams,
    lambda c, p: f"df = df[df['has_audio'].fillna(False).astype(bool)]{_drop}",
)
_op(
    "drop_files",
    Risk.DESTRUCTIVE,
    "Remove specific videos, for example confirmed mislabels.",
    FilesParams,
    lambda c, p: (
        _CHECK_FILES.format(files=p.files) + f"df = df[~df['path'].isin({p.files!r})]{_drop}"
    ),
)
_op(
    "flag_suspected_mislabels",
    Risk.SAFE,
    "Mark videos for human review in the manifest (nothing is removed).",
    FilesParams,
    lambda c, p: (
        _CHECK_FILES.format(files=p.files)
        + f"df['suspected_mislabel'] = df['suspected_mislabel'] | df['path'].isin({p.files!r})"
    ),
)
_op(
    "fix_rotation",
    Risk.SAFE,
    "Re-encode videos stored with rotation metadata so they play upright everywhere.",
    NoParams,
    lambda c, p: "df['fix_rotation'] = df['rotation'].fillna(0) != 0",
)
_op(
    "downscale",
    Risk.LOSSY,
    "Shrink videos taller than max_height pixels (keeps the aspect ratio).",
    HeightParams,
    lambda c, p: f"df['max_height'] = {p.max_height}",
)
_op(
    "trim",
    Risk.LOSSY,
    "Keep only the first max_seconds of each video.",
    TrimParams,
    lambda c, p: f"df['trim_to'] = {p.max_seconds!r}",
)
_op(
    "strip_audio",
    Risk.LOSSY,
    "Remove the audio track when exporting (for vision-only datasets).",
    NoParams,
    lambda c, p: "df['strip_audio'] = True",
)
_op(
    "extract_audio",
    Risk.SAFE,
    "Also export each video's audio track as a 16 kHz WAV file.",
    NoParams,
    lambda c, p: "df['extract_audio'] = True",
)
_op(
    "sample_frames",
    Risk.SAFE,
    "Also export frames as JPEG files at the given rate (frames per second).",
    FpsParams,
    lambda c, p: f"df['frames_fps'] = {p.fps!r}",
)


def video_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"pd": pd}
    ns.update({k: v for k, v in vars(pipeline_helpers).items() if not k.startswith("_")})
    return ns

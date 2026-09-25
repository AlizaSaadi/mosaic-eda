"""The allowed cleaning operations for audio.

Like images, audio is cleaned through a table of files (one row per clip). Operations
remove rows or set transform flags that export_audio() applies with ffmpeg, and each one
is a code template, so the exported pipeline is exactly what ran.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from mosaic.audio import pipeline_helpers
from mosaic.images.ops import _CHECK_FILES, FilesParams
from mosaic.tables.ops import NoParams, OpSpec, Risk


class SilenceParams(NoParams):
    max_silence_ratio: float = Field(0.97, ge=0.5, le=1.0)


class ShortParams(NoParams):
    min_seconds: float = Field(0.3, ge=0.05, le=30)


class ClipParams(NoParams):
    max_clip_ratio: float = Field(0.001, ge=0, le=0.5)


class NoiseParams(NoParams):
    min_snr_db: float = Field(10, ge=0, le=40)


class NearParams(NoParams):
    max_distance: float = Field(3.0, ge=0, le=20)


class RateParams(NoParams):
    sample_rate: int = Field(16000, ge=8000, le=48000)


class LoudnessParams(NoParams):
    target_dbfs: float = Field(-20.0, ge=-40, le=-6)


AUDIO_OPS: dict[str, OpSpec] = {}


def _op(name, risk, description, params, template):
    AUDIO_OPS[name] = OpSpec(name, risk, description, params, template, needs_columns=False)


_drop = ".reset_index(drop=True)"
_op(
    "remove_corrupt",
    Risk.DESTRUCTIVE,
    "Remove files that can't be decoded as audio.",
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
    "Keep one clip per group of re-encoded or re-saved copies of the same recording.",
    NearParams,
    lambda c, p: f"df = drop_near_duplicates(df, max_distance={p.max_distance!r})",
)
_op(
    "drop_silent",
    Risk.DESTRUCTIVE,
    "Remove clips that are mostly silence (silence_ratio at or above the value).",
    SilenceParams,
    lambda c, p: f"df = df[df['silence_ratio'].fillna(1) < {p.max_silence_ratio!r}]{_drop}",
)
_op(
    "drop_too_short",
    Risk.DESTRUCTIVE,
    "Remove clips shorter than min_seconds.",
    ShortParams,
    lambda c, p: f"df = df[df['duration_s'].fillna(0) >= {p.min_seconds!r}]{_drop}",
)
_op(
    "drop_clipped",
    Risk.DESTRUCTIVE,
    "Remove clips where more than max_clip_ratio of samples are clipped.",
    ClipParams,
    lambda c, p: f"df = df[df['clip_ratio'].fillna(0) <= {p.max_clip_ratio!r}]{_drop}",
)
_op(
    "drop_noisy",
    Risk.DESTRUCTIVE,
    "Remove clips with an SNR estimate below min_snr_db.",
    NoiseParams,
    lambda c, p: f"df = df[df['snr_db'].fillna(60) >= {p.min_snr_db!r}]{_drop}",
)
_op(
    "drop_files",
    Risk.DESTRUCTIVE,
    "Remove specific files, for example clips with no speech or confirmed mislabels.",
    FilesParams,
    lambda c, p: (
        _CHECK_FILES.format(files=p.files) + f"df = df[~df['path'].isin({p.files!r})]{_drop}"
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
    "to_mono",
    Risk.SAFE,
    "Mix stereo clips down to one channel when exporting.",
    NoParams,
    lambda c, p: "df['to_mono'] = True",
)
_op(
    "resample",
    Risk.LOSSY,
    "Convert every clip to one sample rate when exporting.",
    RateParams,
    lambda c, p: f"df['resample_to'] = {p.sample_rate}",
)
_op(
    "normalize_loudness",
    Risk.LOSSY,
    "Adjust each clip's volume to a target average level (dBFS), with a limiter.",
    LoudnessParams,
    lambda c, p: f"df['target_dbfs'] = {p.target_dbfs!r}",
)
_op(
    "trim_silence",
    Risk.LOSSY,
    "Cut silence at the start and end of clips.",
    NoParams,
    lambda c, p: "df['trim_silence'] = True",
)
_op(
    "convert_to_wav",
    Risk.SAFE,
    "Export every clip as 16-bit WAV.",
    NoParams,
    lambda c, p: "df['to_wav'] = True",
)


def audio_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"pd": pd}
    ns.update({k: v for k, v in vars(pipeline_helpers).items() if not k.startswith("_")})
    return ns

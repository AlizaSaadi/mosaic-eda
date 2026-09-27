"""Profile a video dataset in code and save every result to the evidence store.

One row per video, measured on its first two minutes: metadata, frame quality (dark,
frozen, blurry), duplicates, the audio track, and a timeline that lines scene cuts up
with what is said in each scene (each scene's audio is transcribed separately).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from mosaic.audio.transcribe import TranscriptionResult
from mosaic.evidence.store import EvidenceStore
from mosaic.reporting import charts
from mosaic.tables.profile import r4
from mosaic.text.profile import snippet
from mosaic.video.pipeline_helpers import near_groups

TOO_SHORT_S = 1.0
BLACK_RATIO = 0.9
FROZEN_MOTION = 1.0
DARK = 40.0
SILENT_RATIO = 0.97
SHEET_COLS, SHEET_ROWS, THUMB = 6, 4, 120
TIMELINE_VIDEOS = 8


@dataclass
class VideoProfile:
    artifact_ids: list[str] = field(default_factory=list)
    chart_ids: list[str] = field(default_factory=list)
    sheet_ids: list[str] = field(default_factory=list)
    quality: float = 0.0
    categories: dict[str, list[str]] = field(default_factory=dict)


def _stats(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {}
    return {"min": r4(values.min()), "median": r4(values.median()), "max": r4(values.max())}


def categorize(df: pd.DataFrame) -> dict[str, list[str]]:
    """One main problem per video, in a fixed order (a black video isn't also 'frozen')."""
    ok = df[~df["corrupt"]]
    cats: dict[str, list[str]] = {"corrupt": df.loc[df["corrupt"], "path"].tolist()}
    taken = set(cats["corrupt"])
    rules = [
        ("too_short", ok["duration_s"] < TOO_SHORT_S),
        ("black", ok["black_ratio"].fillna(0) >= BLACK_RATIO),
        ("frozen", ok["motion"].fillna(0) < FROZEN_MOTION),
        ("very_dark", ok["brightness"].fillna(255) < DARK),
    ]
    for name, mask in rules:
        files = [p for p in ok.loc[mask, "path"] if p not in taken]
        cats[name] = files
        taken.update(files)
    return cats


def scene_spans(row: dict) -> list[tuple[float, float]]:
    """(start, end) of each scene in the analyzed part, from the cut times."""
    end = min(float(row["duration_s"] or 0), 120.0)
    cuts = [t for t in json.loads(row["scenes"] or "[]") if 0 < t < end]
    edges = [0.0, *cuts, end]
    return [(round(a, 2), round(b, 2)) for a, b in pairwise(edges) if b > a]


def _chart(store: EvidenceStore, name: str, title: str, figure: dict, stage: str) -> str:
    return store.add("chart", "chart", name, f"Chart: {title} ({stage})", {"figure": figure}).id


def keyframe_sheets(df: pd.DataFrame, out: Path) -> list[dict]:
    """Numbered grids of keyframes (one frame from the middle of each video), per class."""
    out.mkdir(parents=True, exist_ok=True)
    sheets = []
    if "keyframe" not in df:
        return sheets
    usable = df[~df["corrupt"] & df["keyframe"].notna()]
    for label, group in usable.groupby("class", sort=True):
        rows = group.head(SHEET_COLS * SHEET_ROWS)
        n_rows = -(-len(rows) // SHEET_COLS)
        sheet = Image.new("RGB", (SHEET_COLS * THUMB, n_rows * THUMB), "white")
        draw = ImageDraw.Draw(sheet)
        index = {}
        for i, row in enumerate(rows.to_dict("records"), 1):
            with Image.open(row["keyframe"]) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB - 4, THUMB - 4))
            x, y = ((i - 1) % SHEET_COLS) * THUMB, ((i - 1) // SHEET_COLS) * THUMB
            sheet.paste(im, (x + 2, y + 2))
            draw.rectangle([x + 2, y + 2, x + 24, y + 16], fill="black")
            draw.text((x + 4, y + 3), str(i), fill="white")
            index[i] = row["path"]
        name = f"keyframes_{label or 'all'}.png".replace("/", "_")
        sheet.save(out / name)
        sheets.append({"class": label or "(no class)", "path": str(out / name), "index": index})
    return sheets


def timeline(
    df: pd.DataFrame, transcripts: TranscriptionResult | None
) -> tuple[dict, dict[str, str]]:
    """Scenes per video with what was said in each (scene transcripts are keyed path#s<n>)."""
    said = {t.path: t for t in (transcripts.transcripts if transcripts else [])}
    videos, per_video = {}, {}
    scene_lengths, scenes_total, scenes_with_speech, with_speech = [], 0, 0, 0
    for row in df[~df["corrupt"]].to_dict("records"):
        spans = scene_spans(row)
        scenes, words, texts = [], 0, []
        for n, (start, end) in enumerate(spans, 1):
            t = said.get(f"{row['path']}#s{n}")
            scene_words = t.words if t else 0
            words += scene_words
            scenes_with_speech += scene_words > 0
            scene_lengths.append(end - start)
            if t and t.text:
                texts.append(t.text)
            scenes.append(
                {
                    "scene": n,
                    "start": start,
                    "end": end,
                    "words": scene_words,
                    "said": snippet(t.text, 60) if t and t.text else "",
                }
            )
        scenes_total += len(spans)
        with_speech += words > 0
        per_video[row["path"]] = " ".join(texts)
        if len(videos) < TIMELINE_VIDEOS:
            videos[row["path"]] = scenes
    summary = {
        "scenes": scenes_total,
        "median_scene_seconds": r4(float(np.median(scene_lengths))) if scene_lengths else 0.0,
        "videos_with_speech": with_speech,
        "scenes_with_speech": scenes_with_speech,
        "transcribed_seconds": transcripts.seconds if transcripts else 0.0,
        "engine": transcripts.engine if transcripts else "",
        "videos": videos,
    }
    return summary, per_video


def profile_video(
    df: pd.DataFrame,
    store: EvidenceStore,
    *,
    work: Path,
    class_counts: dict[str, int],
    total_videos: int,
    transcripts: TranscriptionResult | None = None,
    stage: str = "raw",
) -> VideoProfile:
    result = VideoProfile()
    ok = df[~df["corrupt"]]
    cats = categorize(df)
    result.categories = cats
    n = max(len(df), 1)
    labelled = {k: v for k, v in class_counts.items() if k}

    sizes = Counter(f"{int(w)}x{int(h)}" for w, h in zip(ok["width"], ok["height"], strict=False))
    overview = {
        "videos": total_videos,
        "analyzed": len(df),
        "classes": len(labelled),
        "corrupt": len(cats["corrupt"]),
        "duration_s": _stats(ok["duration_s"]),
        "total_seconds": r4(ok["duration_s"].sum()),
        "resolutions": dict(sizes),
        "fps": {str(r4(k)): v for k, v in Counter(ok["fps"]).items()},
        "codecs": dict(Counter(ok["codec"])),
        "rotated": int((ok["rotation"].fillna(0) != 0).sum()),
        "with_audio": int(ok["has_audio"].fillna(False).astype(bool).sum()),
        "without_audio": int((~ok["has_audio"].fillna(False).astype(bool)).sum()),
    }
    sample = "all analyzed" if len(df) >= total_videos else f"{len(df)} sampled"
    a = store.add(
        "vid_overview",
        "profile",
        "video_inventory",
        f"Video ({stage}): {total_videos} videos ({sample}; 'videos' is the count); "
        f"{len(labelled)} classes; {overview['corrupt']} corrupt (corrupt); lengths in seconds "
        f"{overview['duration_s']} (duration_s.median etc.), total_seconds "
        f"{overview['total_seconds']}; resolutions {dict(sizes)} (resolutions.<WxH>); frame "
        f"rates {overview['fps']}; codecs {overview['codecs']}; {overview['rotated']} stored "
        f"with rotation metadata (rotated); {overview['with_audio']} with an audio track "
        f"(with_audio), {overview['without_audio']} without (without_audio). Only the first "
        "120 seconds of each video are analyzed.",
        overview,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    counts = {k: len(v) for k, v in cats.items()}
    silent = ok[ok["has_audio"].fillna(False).astype(bool) & (ok["silence_ratio"] >= SILENT_RATIO)]
    quality = {
        "counts": counts,
        "files": {k: v[:20] for k, v in cats.items()},
        "silent_audio": len(silent),
        "silent_audio_files": silent["path"].tolist()[:20],
        "brightness": _stats(ok["brightness"]),
        "motion": _stats(ok["motion"]),
        "thresholds": {
            "too_short_s": TOO_SHORT_S,
            "black_ratio": BLACK_RATIO,
            "frozen_motion": FROZEN_MOTION,
            "very_dark_brightness": DARK,
        },
    }
    listed = "; ".join(f"{k}: {', '.join(v[:6])}" for k, v in cats.items() if v)
    a = store.add(
        "vid_quality",
        "profile",
        "video_quality",
        f"Video quality ({stage}), one main problem per video: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
        + " (keys counts.<problem>; black = almost every sampled frame is black, frozen = the "
        f"picture barely changes, motion under {FROZEN_MOTION} on a 0-255 scale). Audio tracks "
        f"that are silent: {len(silent)} (silent_audio): "
        f"{', '.join(silent['path'][:5]) or 'none'}. "
        f"Motion {quality['motion']}. Videos: {listed or 'none'}.",
        quality,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    exact = df[df.duplicated("sha256", keep=False)].groupby("sha256")["path"].apply(list)
    groups = near_groups(df)
    near = [g for g in df.groupby(groups)["path"].apply(list) if len(g) > 1]
    dupes = {
        "exact_groups": len(exact),
        "exact_extra_copies": int(sum(len(g) - 1 for g in exact)),
        "near_groups": len(near),
        "near_extra_copies": int(sum(len(g) - 1 for g in near)),
        "examples": near[:10],
    }
    a = store.add(
        "vid_dupes",
        "profile",
        "duplicates",
        f"Duplicates ({stage}): {dupes['exact_groups']} groups of byte-identical files "
        f"({dupes['exact_extra_copies']} extra copies, exact_extra_copies); "
        f"{dupes['near_groups']} groups of the same footage re-encoded, resized, or rotated, "
        f"including exact ones ({dupes['near_extra_copies']} extra copies, near_extra_copies): "
        f"{near[:6]}",
        dupes,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    if labelled:
        total = sum(labelled.values())
        balance = {
            "classes": {k: {"count": v, "share": r4(100 * v / total)} for k, v in labelled.items()},
            "imbalance_ratio": r4(max(labelled.values()) / max(min(labelled.values()), 1)),
        }
        text = ", ".join(f"{k} {v['count']} ({v['share']}%)" for k, v in balance["classes"].items())
        a = store.add(
            "vid_balance",
            "profile",
            "class_balance",
            f"Class balance from folder names ({total} videos): {text}; largest/smallest ratio "
            f"{balance['imbalance_ratio']} (imbalance_ratio). Keys: classes.<name>.count/.share.",
            balance,
            {"stage": stage},
        )
        result.artifact_ids.append(a.id)
        fig = charts.class_balance(labelled, noun="videos")
        result.chart_ids.append(_chart(store, "class_balance", "Videos per class", fig, stage))

    if stage == "raw":
        summary, _ = timeline(df, transcripts)
        shown = "; ".join(
            f"{path}: "
            + ", ".join(
                f"scene {s['scene']} {s['start']}-{s['end']}s"
                + (f" says '{s['said']}'" if s["said"] else " (no speech)")
                for s in scenes
            )
            for path, scenes in list(summary["videos"].items())[:5]
        )
        a = store.add(
            "vid_timeline",
            "profile",
            "timeline",
            f"Timeline (code: ffmpeg scene cuts, and each scene's audio transcribed on its own "
            f"with {summary['engine'] or 'no transcription'}): {summary['scenes']} scenes in "
            f"total (scenes), median scene length {summary['median_scene_seconds']} s "
            f"(median_scene_seconds); {summary['videos_with_speech']} videos contain speech "
            f"(videos_with_speech); {summary['scenes_with_speech']} scenes contain speech "
            f"(scenes_with_speech). Examples: {shown or 'none'}.",
            summary,
            {"stage": stage},
        )
        result.artifact_ids.append(a.id)
        for sheet in keyframe_sheets(df, work / "sheets"):
            a = store.add(
                "vid_sheet",
                "image",
                "keyframe_sheet",
                f"Keyframe sheet: class {sheet['class']}, {len(sheet['index'])} videos",
                sheet,
            )
            result.sheet_ids.append(a.id)

    bad = sum(v for k, v in counts.items() if k != "corrupt")
    imbalance = labelled and max(labelled.values()) / max(min(labelled.values()), 1)
    parts = {
        "corrupt": -min(20.0, 100 * counts["corrupt"] / n * 2),
        "duplicates": -min(15.0, 100 * dupes["near_extra_copies"] / n * 1.5),
        "unusable": -min(25.0, 100 * bad / n * 1.5),
        "silent_audio": -min(5.0, 100 * len(silent) / n * 0.5),
        "class_imbalance": -min(15.0, max(0.0, (imbalance or 1) - 1.5) * 3),
    }
    parts = {k: round(v, 1) for k, v in parts.items()}
    score = round(max(0.0, 100 + sum(parts.values())), 1)
    result.quality = score
    a = store.add(
        "vid_quality_score",
        "profile",
        "quality_score",
        f"Video data quality score ({stage}): {score}/100 (score); deductions {parts}.",
        {"score": score, "parts": parts},
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    values = ok["duration_s"].dropna()
    if len(values):
        fig = charts.histogram(values.tolist(), title="Video length", x="Length (seconds)",
                               noun="videos")  # fmt: skip
        result.chart_ids.append(_chart(store, "duration_hist", "Video length", fig, stage))
        motion = ok["motion"].dropna().tolist()
        fig = charts.histogram(
            motion,
            title="Motion",
            x="Average change between frames (0 = static, 255 = maximum)",
            noun="videos",
            color=2,
        )
        result.chart_ids.append(_chart(store, "motion", "Motion per video", fig, stage))
    return result

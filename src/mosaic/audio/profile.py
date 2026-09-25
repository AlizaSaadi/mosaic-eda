"""Profile an audio dataset in code and save every result to the evidence store."""

from __future__ import annotations

import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from mosaic.audio.pipeline_helpers import decode, near_groups
from mosaic.audio.transcribe import TranscriptionResult, label_matches
from mosaic.evidence.store import EvidenceStore
from mosaic.tables.profile import r4
from mosaic.ui.palette import HARVEST

TOO_SHORT_S = 0.3
SILENT_RATIO = 0.97
SILENT_DBFS = -60.0
CLIP_RATIO = 0.001
NOISY_SNR = 10.0
QUIET_DBFS = -40.0
SPECTROGRAMS = 6


@dataclass
class AudioProfile:
    artifact_ids: list[str] = field(default_factory=list)
    chart_ids: list[str] = field(default_factory=list)
    spectrograms: list[tuple[str, str]] = field(default_factory=list)  # (png path, caption)
    quality: float = 0.0
    categories: dict[str, list[str]] = field(default_factory=dict)


def _stats(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {}
    return {
        "min": r4(values.min()),
        "median": r4(values.median()),
        "max": r4(values.max()),
        "total": r4(values.sum()),
    }


def categorize(df: pd.DataFrame) -> dict[str, list[str]]:
    """One main problem per clip, in a fixed order (a silent clip isn't also 'quiet')."""
    ok = df[~df["corrupt"]]
    cats: dict[str, list[str]] = {"corrupt": df.loc[df["corrupt"], "path"].tolist()}
    taken = set(cats["corrupt"])
    rules = [
        ("silent", (ok["silence_ratio"] >= SILENT_RATIO) | (ok["rms_dbfs"] <= SILENT_DBFS)),
        ("too_short", ok["duration_s"] < TOO_SHORT_S),
        ("clipped", ok["clip_ratio"] > CLIP_RATIO),
        ("noisy", ok["snr_db"].fillna(60) < NOISY_SNR),
        ("very_quiet", ok["rms_dbfs"] < QUIET_DBFS),
    ]
    for name, mask in rules:
        files = [p for p in ok.loc[mask, "path"] if p not in taken]
        cats[name] = files
        taken.update(files)
    return cats


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


def _colormap() -> np.ndarray:
    """256-entry color table from the Harvest sequential scale (parchment to espresso)."""
    stops = [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in HARVEST["sequential"]]
    xs = np.linspace(0, 255, len(stops))
    return np.stack(
        [np.interp(np.arange(256), xs, [s[ch] for s in stops]) for ch in range(3)], axis=1
    ).astype(np.uint8)


def spectrogram_png(x: np.ndarray, out: Path, width: int = 640, height: int = 170) -> Path:
    n_fft, hop = 512, 160
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    power = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)).T + 1e-6)
    power = np.clip((power - power.max() + 80) / 80, 0, 1)  # an 80 dB range
    image = Image.fromarray(_colormap()[(power[::-1] * 255).astype(np.uint8)])
    image.resize((width, height)).save(out)
    return out


def profile_audio(
    df: pd.DataFrame,
    store: EvidenceStore,
    *,
    root: Path,
    work: Path,
    class_counts: dict[str, int],
    total_clips: int,
    transcripts: TranscriptionResult | None = None,
    stage: str = "raw",
) -> AudioProfile:
    result = AudioProfile()
    ok = df[~df["corrupt"]]
    cats = categorize(df)
    result.categories = cats

    overview = {
        "clips": total_clips,
        "analyzed": len(df),
        "classes": len([c for c in class_counts if c]),
        "corrupt": len(cats["corrupt"]),
        "duration_s": _stats(ok["duration_s"]),
        "total_seconds": r4(ok["duration_s"].sum()),
        "codecs": dict(Counter(ok["codec"])),
        "sample_rates": {str(k): v for k, v in Counter(ok["sample_rate"]).items()},
        "sample_rates_khz": sorted({r4(k / 1000) for k in ok["sample_rate"].dropna()}),
        "channels": {str(k): v for k, v in Counter(ok["channels"]).items()},
    }
    sample = "all analyzed" if len(df) >= total_clips else f"{len(df)} sampled"
    a = store.add(
        "aud_overview",
        "profile",
        "audio_inventory",
        f"Audio ({stage}): {total_clips} clips ({sample}); {overview['classes']} classes; "
        f"{overview['corrupt']} corrupt (corrupt); clip count is 'clips', not a duration; "
        f"durations in seconds {overview['duration_s']}; total_seconds "
        f"{overview['total_seconds']} (seconds, not clips); codecs {overview['codecs']}; "
        f"sample rates in Hz {overview['sample_rates']} (clips per rate), in kHz "
        f"{overview['sample_rates_khz']} (sample_rates_khz); channels {overview['channels']}.",
        overview,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    counts = {k: len(v) for k, v in cats.items()}
    quality = {
        "counts": counts,
        "files": {k: v[:20] for k, v in cats.items()},
        "rms_dbfs": _stats(ok["rms_dbfs"]),
        "snr_db": _stats(ok["snr_db"]),
        "silence_ratio": _stats(ok["silence_ratio"]),
        "thresholds": {
            "too_short_s": TOO_SHORT_S,
            "silent_ratio": SILENT_RATIO,
            "clip_ratio": CLIP_RATIO,
            "noisy_snr_db": NOISY_SNR,
            "quiet_dbfs": QUIET_DBFS,
        },
    }
    lines = "; ".join(f"{k} {n}" for k, n in counts.items() if k != "corrupt")
    listed = "; ".join(f"{k}: {', '.join(v[:6])}" for k, v in cats.items() if v and k != "corrupt")
    a = store.add(
        "aud_quality",
        "profile",
        "audio_quality",
        f"Audio quality ({stage}), one main problem per clip: {lines} (keys counts.<category>). "
        f"Loudness dBFS {quality['rms_dbfs']}; SNR dB {quality['snr_db']}. "
        f"Files: {listed or 'none'}.",
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
        "aud_dupes",
        "profile",
        "duplicates",
        f"Duplicates ({stage}): {dupes['exact_groups']} exact groups ({dupes['exact_extra_copies']}"
        f" extra copies, exact_extra_copies); {dupes['near_groups']} groups of the same recording "
        f"re-encoded or re-saved, including exact ones ({dupes['near_extra_copies']} extra copies, "
        f"near_extra_copies): {near[:6]}",
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
            "aud_balance",
            "profile",
            "class_balance",
            f"Class balance from folder names ({total} clips): {text}; largest/smallest ratio "
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
                f"Chart: clips per class ({stage})",
                {"figure": _fig(fig, "Clips per class")},
            ).id
        )

    if transcripts is not None:
        labels = dict(zip(df["path"], df["class"], strict=False))
        rows = {t.path: t for t in transcripts.transcripts}
        spoken = [t for t in transcripts.transcripts if t.words]
        non_speech = [p for p, t in rows.items() if not t.words and p not in cats["silent"]]
        wordlike = labelled and all(k.isalpha() for k in labelled)
        mismatches = [
            {"path": t.path, "label": labels.get(t.path, ""), "heard": t.text[:60]}
            for t in spoken
            if wordlike and labels.get(t.path) and not label_matches(labels[t.path], t.text)
        ]
        languages = Counter(t.language for t in spoken if t.language)
        data = {
            "engine": transcripts.engine,
            "transcribed_seconds": transcripts.seconds,
            "clips_transcribed": len(transcripts.transcripts),
            "clips_with_speech": len(spoken),
            "non_speech_clips": len(non_speech),
            "non_speech": non_speech[:20],
            "languages": dict(languages),
            "label_mismatches": len(mismatches),
            "mismatches": mismatches[:20],
            "skipped_for_time": len(transcripts.skipped_budget),
            "examples": {t.path: t.text[:80] for t in spoken[:12]},
        }
        mis = "; ".join(
            f"{m['path']} (label '{m['label']}', heard '{m['heard']}')" for m in mismatches[:8]
        )
        a = store.add(
            "aud_transcripts",
            "profile",
            "transcription",
            f"Transcription ({transcripts.engine}, {transcripts.seconds} s of audio): "
            f"{len(spoken)} clips with speech (clips_with_speech); {len(non_speech)} non-silent "
            f"clips with no words, likely music or noise (non_speech_clips): "
            f"{', '.join(non_speech[:6]) or 'none'}; languages {dict(languages)}. "
            + (
                f"Transcript does not match the folder label in {len(mismatches)} clips "
                f"(label_mismatches, checked by code): {mis}."
                if wordlike
                else "Labels aren't single words, so transcripts weren't compared with labels."
            )
            + (
                f" {len(transcripts.skipped_budget)} clips skipped for time (skipped_for_time)."
                if transcripts.skipped_budget
                else ""
            ),
            data,
            {"stage": stage},
        )
        result.artifact_ids.append(a.id)

    n = max(len(df), 1)
    bad = sum(v for k, v in counts.items() if k != "corrupt")
    imbalance = labelled and max(labelled.values()) / max(min(labelled.values()), 1)
    parts = {
        "corrupt": -min(20.0, 100 * counts["corrupt"] / n * 2),
        "duplicates": -min(15.0, 100 * dupes["near_extra_copies"] / n * 1.5),
        "low_quality": -min(25.0, 100 * bad / n * 1.5),
        "class_imbalance": -min(15.0, max(0.0, (imbalance or 1) - 1.5) * 3),
    }
    parts = {k: round(v, 1) for k, v in parts.items()}
    score = round(max(0.0, 100 + sum(parts.values())), 1)
    result.quality = score
    a = store.add(
        "aud_quality_score",
        "profile",
        "quality_score",
        f"Audio data quality score ({stage}): {score}/100 (score); deductions {parts}.",
        {"score": score, "parts": parts},
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    for column, title, color in (
        ("duration_s", "Clip length (seconds)", 1),
        ("rms_dbfs", "Loudness (dBFS)", 2),
        ("snr_db", "SNR (dB)", 3),
    ):
        values = ok[column].dropna()
        if values.empty:
            continue
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

    if stage == "raw":
        work.mkdir(parents=True, exist_ok=True)
        picks: list[tuple[str, str]] = []
        for category in ("clipped", "noisy", "silent", "too_short", "very_quiet"):
            picks += [(p, category.replace("_", " ")) for p in cats.get(category, [])[:1]]
        if transcripts is not None:
            picks += [(p, "no speech") for p in data["non_speech"][:1]]
            picks += [(m["path"], "label mismatch") for m in data["mismatches"][:1]]
        normal = [p for p in ok["path"] if p not in {q for q, _ in picks}]
        picks += [(p, "typical clip") for p in normal[:1]]
        for i, (rel, why) in enumerate(picks[:SPECTROGRAMS]):
            png = spectrogram_png(decode(root / rel), work / f"spectrogram_{i}.png")
            result.spectrograms.append((str(png), f"{why}: {rel}"))
    return result


def wav_bytes(x: np.ndarray, rate: int = 16000) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()

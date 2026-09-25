"""Helper functions for audio cleaning.

This module's source is copied verbatim into every exported cleaning_pipeline.py for
audio, so it only imports the standard library, numpy, and pandas, plus two optional
packages (imageio-ffmpeg for an ffmpeg binary, webrtcvad for speech detection).
One row per clip: metrics are computed once, cleaning operations filter rows or set
transform flags, and export_audio() writes the cleaned copies with ffmpeg.
"""

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma"}
RATE = 16000
FRAME_MS = 30
SILENCE_DB = -45.0
FLOOR_DB = -90.0  # digital silence would otherwise read as -200 dB
MAX_SECONDS = 600
METRIC_COLUMNS = (
    "codec",
    "sample_rate",
    "channels",
    "bitrate_kbps",
    "duration_s",
    "rms_dbfs",
    "peak_dbfs",
    "silence_ratio",
    "clip_ratio",
    "dc_offset",
    "speech_ratio",
    "snr_db",
    "signature",
)
TRANSFORM_DEFAULTS = {
    "to_mono": False,
    "resample_to": 0,
    "target_dbfs": 0.0,
    "trim_silence": False,
    "to_wav": False,
    "suspected_mislabel": False,
}

try:  # optional: speech detection
    import webrtcvad
except ImportError:
    webrtcvad = None


def ffmpeg_exe():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or "ffmpeg"


def probe(path):
    """Duration, codec, sample rate, channels, and bitrate from ffmpeg's stream info."""
    result = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    info = result.stderr
    out = {"codec": "", "sample_rate": 0, "channels": 0, "bitrate_kbps": 0, "duration_s": 0.0}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    if m:
        h, mi, s = m.groups()
        out["duration_s"] = round(int(h) * 3600 + int(mi) * 60 + float(s), 3)
    m = re.search(r"Audio: (\w+)[^,]*, (\d+) Hz, ([^,]+)", info)
    if m:
        out["codec"], out["sample_rate"] = m.group(1), int(m.group(2))
        layout = m.group(3).strip()
        out["channels"] = {"mono": 1, "stereo": 2}.get(layout, int(re.sub(r"\D", "", layout) or 0))
    m = re.search(r"bitrate: (\d+) kb/s", info)
    if m:
        out["bitrate_kbps"] = int(m.group(1))
    return out


def decode(path, rate=RATE, mono=True, max_seconds=MAX_SECONDS):
    """Decode any format to float32 samples in [-1, 1] with ffmpeg."""
    args = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-t", str(max_seconds), "-f", "f32le"]
    if mono:
        args += ["-ac", "1"]
    if rate:
        args += ["-ar", str(rate)]
    result = subprocess.run([*args, "-"], capture_output=True, timeout=120)
    if result.returncode != 0 or not result.stdout:
        raise ValueError(result.stderr.decode(errors="replace").strip()[:160] or "no audio")
    return np.frombuffer(result.stdout, np.float32)


def frame_db(x, rate=RATE, frame_ms=FRAME_MS):
    n = int(rate * frame_ms / 1000)
    frames = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-10
    return np.maximum(20 * np.log10(rms), FLOOR_DB)


def speech_frames(x, rate=RATE, aggressiveness=2):
    """Boolean per 30 ms frame: does it contain speech? None if webrtcvad isn't installed."""
    if webrtcvad is None or len(x) < rate * FRAME_MS // 1000:
        return None
    vad = webrtcvad.Vad(aggressiveness)
    n = rate * FRAME_MS // 1000
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    return np.array(
        [vad.is_speech(pcm[i : i + n].tobytes(), rate) for i in range(0, len(pcm) - n + 1, n)]
    )


def envelope_signature(x, rate=RATE, points=64):
    """Loudness shape over time as hex bytes: a fingerprint that survives re-encoding."""
    db = frame_db(x, rate)
    if len(db) < 2:
        return ""
    curve = np.interp(np.linspace(0, len(db) - 1, points), np.arange(len(db)), db)
    curve = np.clip((curve + 80) / 80 * 255, 0, 255).astype(np.uint8)
    return curve.tobytes().hex()


def signature_distance(a, b):
    x = np.frombuffer(bytes.fromhex(a), np.uint8).astype(int)
    y = np.frombuffer(bytes.fromhex(b), np.uint8).astype(int)
    return float(np.abs(x - y).mean())


def audio_record(path, rel, label):
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
        rec.update(probe(path))
        native = decode(path, rate=0, mono=False)
        x = decode(path)
        if len(x) < RATE // 100:
            raise ValueError("no decodable audio")
        db = frame_db(x)
        speech = speech_frames(x)
        rec.update(
            duration_s=round(len(x) / RATE, 3),
            rms_dbfs=round(
                max(
                    float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-10)),
                    FLOOR_DB,
                ),
                2,
            ),
            peak_dbfs=round(float(20 * np.log10(np.abs(x).max() + 1e-10)), 2),
            silence_ratio=round(float((db < SILENCE_DB).mean()), 4),
            clip_ratio=round(float((np.abs(native) >= 0.999).mean()), 5),
            dc_offset=round(float(x.mean()), 5),
            speech_ratio=None if speech is None else round(float(speech.mean()), 4),
            signature=envelope_signature(x),
        )
        # loud frames vs quiet frames: a robust signal-to-noise estimate (60 = very clean)
        rec["snr_db"] = round(float(min(np.percentile(db, 90) - np.percentile(db, 10), 60.0)), 2)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        rec.update(corrupt=True, error=f"{type(exc).__name__}: {str(exc)[:120]}")
    return rec


def list_audio(folder):
    """(relative path, class) for every clip; the class is the first folder level."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    rels = [p.relative_to(folder).parts for p in files]
    tops = {r[0] for r in rels if len(r) > 1}
    skip = 1 if len(tops) == 1 and all(len(r) > 2 for r in rels) else 0  # one wrapper folder
    return [("/".join(r), r[skip] if len(r) > skip + 1 else "") for r in rels]


def build_table(folder, files=None):
    folder = Path(folder)
    rows = [audio_record(folder / rel, rel, label) for rel, label in (files or list_audio(folder))]
    df = pd.DataFrame(rows)
    for name in METRIC_COLUMNS:  # present even when no file could be decoded
        if name not in df:
            df[name] = np.nan
    for name, default in TRANSFORM_DEFAULTS.items():
        df[name] = default
    return df


def near_groups(df, max_distance=3.0, duration_tolerance=0.02):
    """Group clips with the same loudness shape and almost the same length.

    This catches re-encoded or re-saved copies of one recording. Separate recordings of
    the same word differ in timing, so they rarely match on both. (A production system
    would use a spectral fingerprint such as Chromaprint.)
    """
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sigs = df["signature"].fillna("").to_numpy() if "signature" in df else [""] * len(df)
    durations = df["duration_s"].fillna(0).to_numpy() if "duration_s" in df else [0] * len(df)
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if not sigs[i] or not sigs[j]:
                continue
            longer = max(durations[i], durations[j]) or 1
            if abs(durations[i] - durations[j]) / longer > duration_tolerance:
                continue
            if signature_distance(sigs[i], sigs[j]) <= max_distance:
                parent[find(i)] = find(j)
    return pd.Series([find(i) for i in range(len(df))], index=df.index)


def drop_near_duplicates(df, max_distance=3.0):
    groups = near_groups(df, max_distance)
    return df[~groups.duplicated()].reset_index(drop=True)


def export_audio(df, source_folder, out_folder):
    """Write the kept clips, applying the transform flags with ffmpeg, plus a manifest CSV."""
    source_folder, out_folder = Path(source_folder), Path(out_folder)
    for row in df.itertuples(index=False):
        if row.corrupt:
            continue
        src = source_folder / row.path
        changes = row.to_mono or row.resample_to or row.target_dbfs or row.trim_silence
        target = out_folder / row.path
        if row.to_wav or changes:
            target = target.with_suffix(".wav")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (row.to_wav or changes):
            shutil.copyfile(src, target)
            continue
        args = [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src)]
        filters = []
        if row.trim_silence:
            filters.append(
                "silenceremove=start_periods=1:start_threshold=-45dB:"
                "stop_periods=1:stop_threshold=-45dB"
            )
        if row.target_dbfs and pd.notna(row.rms_dbfs):
            filters.append(f"volume={row.target_dbfs - row.rms_dbfs:.2f}dB,alimiter=limit=0.97")
        if filters:
            args += ["-af", ",".join(filters)]
        if row.to_mono:
            args += ["-ac", "1"]
        if row.resample_to:
            args += ["-ar", str(int(row.resample_to))]
        subprocess.run([*args, "-c:a", "pcm_s16le", str(target)], check=True, timeout=120)
    keep = [c for c in df.columns if c not in ("signature",)]
    df[keep].to_csv(out_folder / "audio_manifest.csv", index=False)
    return out_folder

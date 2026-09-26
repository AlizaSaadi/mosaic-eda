"""Helper functions for video cleaning.

This module's source is copied verbatim into every exported cleaning_pipeline.py for
video, so it only imports the standard library, numpy, pandas, and Pillow, plus two
optional packages (imageio-ffmpeg for an ffmpeg binary, webrtcvad for speech detection).
One row per video: metrics are computed once from the first MAX_SECONDS, cleaning
operations filter rows or set transform flags, and export_video() applies them with ffmpeg.
"""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
MAX_SECONDS = 120  # only the start of long videos is analyzed
FRAMES = 8  # frames sampled evenly across the analyzed part
SCENE_THRESHOLD = 0.3  # ffmpeg scene score for a cut
RATE = 16000
FRAME_MS = 30
SILENCE_DB = -45.0
METRIC_COLUMNS = (
    "duration_s",
    "width",
    "height",
    "fps",
    "codec",
    "rotation",
    "has_audio",
    "bitrate_kbps",
    "brightness",
    "contrast",
    "blur",
    "motion",
    "black_ratio",
    "scene_cuts",
    "scenes",
    "audio_rms_dbfs",
    "silence_ratio",
    "speech_ratio",
    "signature",
)
TRANSFORM_DEFAULTS = {
    "max_height": 0,
    "trim_to": 0.0,
    "strip_audio": False,
    "extract_audio": False,
    "frames_fps": 0.0,
    "fix_rotation": False,
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


def _run(args, timeout=180):
    return subprocess.run(
        [ffmpeg_exe(), "-hide_banner", *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def probe(path):
    """Duration, size, frame rate, codec, rotation, and audio from ffmpeg's stream info."""
    info = _run(["-i", str(path)], timeout=60).stderr
    out = {
        "duration_s": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "codec": "",
        "rotation": 0,
        "has_audio": False,
        "bitrate_kbps": 0,
    }
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    if m:
        h, mi, s = m.groups()
        out["duration_s"] = round(int(h) * 3600 + int(mi) * 60 + float(s), 3)
    m = re.search(r"Stream #[^\n]*Video: (\w+)[^\n]*?, (\d{2,5})x(\d{2,5})", info)
    if m:
        out["codec"], out["width"], out["height"] = m.group(1), int(m.group(2)), int(m.group(3))
    m = re.search(r"Video:[^\n]*?([\d.]+) fps", info)
    if m:
        out["fps"] = float(m.group(1))
    m = re.search(r"rotate\s*:\s*(-?\d+)|rotation of (-?[\d.]+) degrees", info)
    if m:
        out["rotation"] = int(float(m.group(1) or m.group(2))) % 360
    out["has_audio"] = bool(re.search(r"Stream #[^\n]*Audio:", info))
    m = re.search(r"bitrate: (\d+) kb/s", info)
    if m:
        out["bitrate_kbps"] = int(m.group(1))
    return out


def sample_frames(path, duration, n=FRAMES, width=160):
    """n frames spread evenly over the analyzed part, as small RGB images."""
    seconds = min(duration or MAX_SECONDS, MAX_SECONDS)
    rate = n / max(seconds, 0.1)
    with tempfile.TemporaryDirectory() as tmp:
        _run(
            [
                "-t", str(seconds), "-i", str(path),
                "-vf", f"fps={rate:.5f},scale={width}:-2",
                "-frames:v", str(n), str(Path(tmp) / "f%03d.png"),
            ]
        )  # fmt: skip
        frames = []
        for png in sorted(Path(tmp).glob("f*.png")):
            with Image.open(png) as im:
                frames.append(im.convert("RGB"))
    return frames


def scene_cuts(path, threshold=SCENE_THRESHOLD):
    """Times (seconds) where ffmpeg's scene-change score passes the threshold."""
    result = _run(
        [
            "-t", str(MAX_SECONDS), "-i", str(path), "-an",
            "-vf", f"scale=160:-2,select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-",
        ]
    )  # fmt: skip
    times = [round(float(t), 2) for t in re.findall(r"pts_time:([\d.]+)", result.stderr)]
    return [t for t in times if t >= 0.2]  # the first frames often score as a "cut"


def decode_audio(path, max_seconds=MAX_SECONDS):
    """The audio track as 16 kHz mono float32 samples (None when there is no audio)."""
    args = [ffmpeg_exe(), "-v", "error", "-t", str(max_seconds), "-i", str(path), "-vn"]
    args += ["-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]
    result = subprocess.run(args, capture_output=True, timeout=120)
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, np.float32)


def frame_db(x, frame_ms=FRAME_MS):
    n = int(RATE * frame_ms / 1000)
    frames = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-10
    return np.maximum(20 * np.log10(rms), -90.0)


def speech_ratio(x):
    if webrtcvad is None or len(x) < RATE * FRAME_MS // 1000:
        return None
    vad = webrtcvad.Vad(2)
    n = RATE * FRAME_MS // 1000
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    flags = [vad.is_speech(pcm[i : i + n].tobytes(), RATE) for i in range(0, len(pcm) - n + 1, n)]
    return round(float(np.mean(flags)), 4) if flags else 0.0


def blur_score(gray):
    """Variance of the Laplacian: low values mean few sharp edges."""
    lap = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var()) if lap.size else 0.0


def thumb(im):
    """8x8 grayscale thumbnail: a tiny fingerprint of one frame."""
    return np.asarray(im.convert("L").resize((8, 8), Image.Resampling.BILINEAR), np.uint8)


def signature_distance(a, b):
    """Mean absolute difference (0-255) between two videos' frame fingerprints."""
    x = np.frombuffer(bytes.fromhex(a), np.uint8).astype(int)
    y = np.frombuffer(bytes.fromhex(b), np.uint8).astype(int)
    if len(x) != len(y) or not len(x):
        return 255.0
    return float(np.abs(x - y).mean())


def video_record(path, rel, label, frames_dir=None):
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
        if not rec["width"] or rec["duration_s"] <= 0:
            raise ValueError("no video stream")
        frames = sample_frames(path, rec["duration_s"])
        if not frames:
            raise ValueError("no decodable frames")
        grays = [np.asarray(f.convert("L"), float) for f in frames]
        thumbs = [thumb(f) for f in frames]
        rec["brightness"] = round(float(np.mean([g.mean() for g in grays])), 2)
        rec["contrast"] = round(float(np.mean([g.std() for g in grays])), 2)
        rec["blur"] = round(float(np.median([blur_score(g) for g in grays])), 2)
        diffs = [np.abs(a.astype(int) - b.astype(int)).mean() for a, b in pairwise(thumbs)]
        rec["motion"] = round(float(np.mean(diffs)) if diffs else 0.0, 2)
        rec["black_ratio"] = round(float(np.mean([g.mean() < 16 for g in grays])), 3)
        # fingerprint: FRAMES thumbnails, padded so every video's signature has the same length
        padded = (thumbs + [thumbs[-1]] * FRAMES)[:FRAMES]
        rec["signature"] = np.concatenate([t.ravel() for t in padded]).tobytes().hex()
        cuts = scene_cuts(path)
        rec["scene_cuts"] = len(cuts)
        rec["scenes"] = json.dumps(cuts)
        audio = decode_audio(path) if rec["has_audio"] else None
        if audio is not None and len(audio):
            db = frame_db(audio)
            rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-10
            rec["audio_rms_dbfs"] = round(max(float(20 * np.log10(rms)), -90.0), 2)
            rec["silence_ratio"] = round(float((db < SILENCE_DB).mean()), 4)
            rec["speech_ratio"] = speech_ratio(audio)
        if frames_dir is not None:
            keyframe = Path(frames_dir) / (hashlib.md5(rel.encode()).hexdigest()[:12] + ".png")
            keyframe.parent.mkdir(parents=True, exist_ok=True)
            frames[len(frames) // 2].save(keyframe)
            rec["keyframe"] = str(keyframe)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        rec.update(corrupt=True, error=f"{type(exc).__name__}: {str(exc)[:120]}")
    return rec


def list_videos(folder):
    """(relative path, class) for every video; the class is the first folder level."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    rels = [p.relative_to(folder).parts for p in files]
    tops = {r[0] for r in rels if len(r) > 1}
    skip = 1 if len(tops) == 1 and all(len(r) > 2 for r in rels) else 0  # one wrapper folder
    return [("/".join(r), r[skip] if len(r) > skip + 1 else "") for r in rels]


def build_table(source, files=None, frames_dir=None):
    """One row per video. `source` is a folder (class = first folder level) or one video."""
    source = Path(source)
    if source.is_file():
        pairs, base = [(source.name, "")], source.parent
    else:
        pairs, base = files or list_videos(source), source
    df = pd.DataFrame([video_record(base / rel, rel, label, frames_dir) for rel, label in pairs])
    for name in METRIC_COLUMNS:  # present even when no file could be decoded
        if name not in df:
            df[name] = np.nan
    for name, default in TRANSFORM_DEFAULTS.items():
        df[name] = default
    return df


def near_groups(df, max_distance=6.0, duration_tolerance=0.05):
    """Group videos whose frame fingerprints match and whose lengths are close: re-encoded,
    resized, or re-saved copies of the same footage."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sigs = df["signature"].fillna("").tolist()
    durations = df["duration_s"].fillna(0).tolist()
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


def drop_near_duplicates(df, max_distance=6.0):
    return df[~near_groups(df, max_distance).duplicated()].reset_index(drop=True)


def export_video(df, source, out_folder):
    """Write the kept videos (transcoded only when a transform asks for it), optional audio
    tracks and frames, and a manifest CSV."""
    source, out_folder = Path(source), Path(out_folder)
    base = source.parent if source.is_file() else source
    for row in df.to_dict("records"):
        if row["corrupt"]:
            continue
        src = base / row["path"]
        target = out_folder / "videos" / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        transcode = row["max_height"] or row["trim_to"] or row["strip_audio"] or row["fix_rotation"]
        if not transcode:
            shutil.copyfile(src, target)
        else:
            target = target.with_suffix(".mp4")
            args = ["-y", "-v", "error"]
            if row["trim_to"]:
                args += ["-t", str(row["trim_to"])]
            args += ["-i", str(src)]
            if row["max_height"]:
                args += ["-vf", f"scale=-2:'min({int(row['max_height'])},ih)'"]
            args += ["-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
            args += ["-an"] if row["strip_audio"] else ["-c:a", "aac"]
            args += ["-metadata:s:v", "rotate=0", str(target)]
            subprocess.run([ffmpeg_exe(), "-hide_banner", *args], check=True, timeout=600)
        if row["extract_audio"] and row["has_audio"]:
            wav = out_folder / "audio" / Path(row["path"]).with_suffix(".wav")
            wav.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
                 "-ar", str(RATE), str(wav)],
                check=True,
                timeout=600,
            )  # fmt: skip
        if row["frames_fps"]:
            frames = out_folder / "frames" / Path(row["path"]).with_suffix("")
            frames.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src),
                 "-vf", f"fps={row['frames_fps']}", str(frames / "frame_%04d.jpg")],
                check=True,
                timeout=600,
            )  # fmt: skip
    keep = [c for c in df.columns if c not in ("signature", "keyframe")]
    df[keep].to_csv(out_folder / "video_manifest.csv", index=False)
    return out_folder

"""Generate examples/datasets/pattern_clips.zip, a deliberately messy video dataset.

Short clips of animated patterns in three folders (the labels): fractals (Mandelbrot
zooms), patterns (test cards and color bars), and cells (cellular automata). Each clip has
two scenes and a spoken narration in the first scene, from the Windows speech synthesizer
(so this script runs on Windows; the zip it produces is committed and works everywhere).

Planted problems:
- 1 corrupt (truncated) file and 1 clip only 0.6 seconds long
- 1 exact duplicate and 1 re-encoded copy (smaller and lower quality) of the same footage
- 1 black clip, 1 frozen clip (a single still color), 1 clip with no audio track,
  1 clip whose audio track is silent
- 1 clip stored with rotation metadata (filmed sideways)
- 1 mislabel: a Mandelbrot zoom saved under patterns/
- a README.txt, which should be treated as a note rather than data
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_audio_zip import synthesize

OUT = Path(__file__).parent / "datasets" / "pattern_clips.zip"
W, H, FPS, SCENE = 256, 144, 15, 2.5  # each clip: two scenes of 2.5 seconds
VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]

SIZE = f"size={W}x{H}:rate={FPS}"
# Mandelbrot regions worth zooming into: each fractal scene uses a different one
REGIONS = [
    (-0.743643887, 0.131825904, 0.05),
    (-0.75, 0.1, 0.3),
    (0.285, 0.01, 0.2),
    (-1.25066, 0.02012, 0.1),
    (-0.1011, 0.9563, 0.15),
    (-0.7453, 0.1127, 0.02),
    (-0.16, 1.0405, 0.05),
    (-1.7497, 0.0, 0.05),
    (0.3245, 0.04855, 0.03),
    (-0.235125, 0.827215, 0.04),
    (-0.722, 0.246, 0.08),
    (-0.7, 0.35, 0.25),
]
TEST_CARDS = ["testsrc", "testsrc2", "rgbtestsrc", "yuvtestsrc", "smptebars", "smptehdbars",
              "pal75bars", "pal100bars"]  # fmt: skip


def fractal(k: int) -> str:
    x, y, scale = REGIONS[k % len(REGIONS)]
    scale *= 1 + 0.6 * (k // len(REGIONS))  # repeated regions get a different zoom
    return f"mandelbrot={SIZE}:start_x={x}:start_y={y}:start_scale={scale}:end_scale={scale / 8}"


def sources(label: str, i: int) -> list[str]:
    """Two different scenes for clip i of a class."""
    if label == "fractals":
        return [fractal(2 * i), fractal(2 * i + 1)]
    if label == "patterns":  # a moving test card cut to a still one, varied per clip
        card = TEST_CARDS[(i + 4) % len(TEST_CARDS)]
        return [f"{TEST_CARDS[i % 2]}={SIZE},hue=h={i * 60}", f"{card}={SIZE}"]
    rule = [30, 90, 110, 150, 105][i % 5]
    return [
        f"cellauto={SIZE}:rule={rule}:random_seed={i}",
        f"life={SIZE}:mold=10:ratio=0.{i + 2}:seed={i}:death_color=#2F6F73:life_color=#F3E9DC",
    ]


def ffmpeg(*args: str) -> None:
    import imageio_ffmpeg

    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-v", "error", "-y", *args],
        check=True,
    )


def clip(out: Path, scenes: list[str], narration: Path | None, *, audio: str = "voice") -> None:
    """Two scenes joined with a hard cut, plus an audio track (voice, silence, or none)."""
    args: list[str] = []
    for source in scenes:
        args += ["-f", "lavfi", "-t", str(SCENE), "-i", source]
    total = SCENE * len(scenes)
    video = "".join(f"[{i}:v]" for i in range(len(scenes))) + f"concat=n={len(scenes)}:v=1[v]"
    maps = ["-map", "[v]"]
    if audio == "voice" and narration:
        args += ["-i", str(narration)]
        video += f";[{len(scenes)}:a]apad,atrim=0:{total}[a]"
        maps += ["-map", "[a]"]
    elif audio == "silent":
        args += ["-f", "lavfi", "-t", str(total), "-i", "anullsrc=r=16000:cl=mono"]
        maps += ["-map", f"{len(scenes)}:a"]
    ffmpeg(
        *args,
        "-filter_complex", video, *maps,
        "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "48k", "-t", str(total), str(out),
    )  # fmt: skip


def main() -> Path:
    work = Path(tempfile.mkdtemp())
    speech, narrations = [], {}
    counts = {"fractals": 8, "patterns": 8, "cells": 6}
    for label, n in counts.items():
        for i in range(1, n + 1):
            text = f"Clip {i}. This clip shows {label}."
            path = work / f"{label}_{i}.wav"
            speech.append((text, VOICES[i % 2], i % 3 - 1, path))
            narrations[(label, i)] = path
    synthesize(speech)

    files: dict[str, Path] = {}
    for label, n in counts.items():
        for i in range(1, n + 1):
            scenes = sources(label, i)
            out = work / f"{label}_{i:02d}.mp4"
            audio = "voice"
            if (label, i) == ("cells", 2):
                audio = "none"
            if (label, i) == ("fractals", 4):
                audio = "silent"
            clip(out, scenes, narrations[(label, i)], audio=audio)
            files[f"{label}/{label[:-1]}_{i:02d}.mp4"] = out

    # a Mandelbrot zoom saved under patterns/ (a mislabel)
    wrong = work / "wrong.mp4"
    clip(wrong, [fractal(40), fractal(41)], narrations[("patterns", 1)])  # a zoom no clip uses
    files["patterns/pattern_09.mp4"] = wrong

    # a black clip and a frozen clip
    black = work / "black.mp4"
    clip(black, [f"color=c=black:size={W}x{H}:rate={FPS}"] * 2, narrations[("patterns", 2)])
    files["patterns/pattern_10.mp4"] = black
    frozen = work / "frozen.mp4"
    clip(frozen, [f"color=c=0x7A3E65:size={W}x{H}:rate={FPS}"] * 2, narrations[("fractals", 1)])
    files["fractals/fractal_09.mp4"] = frozen

    # exact duplicate and a re-encoded (smaller, lower quality) copy
    files["fractals/fractal_10.mp4"] = files["fractals/fractal_02.mp4"]
    reencoded = work / "reencoded.mp4"
    ffmpeg(
        "-i", str(files["patterns/pattern_03.mp4"]), "-vf", "scale=192:108",
        "-c:v", "libx264", "-crf", "36", "-c:a", "aac", "-b:a", "32k", str(reencoded),
    )  # fmt: skip
    files["patterns/pattern_11.mp4"] = reencoded

    # a clip only 0.6 seconds long, one stored sideways, and a truncated file
    short = work / "short.mp4"
    ffmpeg("-i", str(files["cells/cell_01.mp4"]), "-t", "0.6", "-c:v", "libx264", str(short))
    files["cells/cell_07.mp4"] = short
    rotated = work / "rotated.mp4"
    ffmpeg("-display_rotation:v:0", "90", "-i", str(files["cells/cell_03.mp4"]), "-c", "copy",
           str(rotated))  # fmt: skip
    files["cells/cell_08.mp4"] = rotated
    broken = work / "broken.mp4"
    broken.write_bytes(files["patterns/pattern_04.mp4"].read_bytes()[:3000])
    files["patterns/pattern_12.mp4"] = broken

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
        z.writestr(
            "pattern_clips/README.txt",
            "Synthetic pattern clips for MOSAIC EDA. Folders are labels.\n",
        )
        for name in sorted(files):
            z.write(files[name], f"pattern_clips/{name}")
    print(f"Wrote {len(files)} clips ({OUT.stat().st_size // 1024} KB) to {OUT}")
    return OUT


if __name__ == "__main__":
    main()

"""Generate examples/datasets/speech_commands.zip, a deliberately messy audio dataset.

Spoken words come from the Windows speech synthesizer (voices David and Zira), so this
script runs on Windows; the zip it produces is committed and works everywhere.

Folders are labels: yes (12 clips), no (10), stop (5). Planted problems:
- 1 silent clip (no/), 1 heavily clipped clip (yes/), 1 noisy clip (no/)
- 1 music-only clip with no speech (stop/), 1 clip only 0.15 seconds long (yes/)
- 1 exact duplicate and 1 re-encoded MP3 copy of an existing clip
- 2 mislabels: "stop" saved under yes/, "no" saved under stop/
- 1 corrupt file (truncated), 1 stereo 44.1 kHz file, some MP3s
- a README.txt, which should be treated as a note rather than data
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
import zipfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "datasets" / "speech_commands.zip"
RATE = 16000


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def synthesize(items: list[tuple[str, str, int, Path]]) -> None:
    """Speak each (text, voice, rate, path) with Windows SAPI into 16 kHz mono WAV files."""
    lines = [
        "Add-Type -AssemblyName System.Speech",
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono)",
    ]
    for text, voice, rate, path in items:
        lines += [
            f"$s.SelectVoice('{voice}')",
            f"$s.Rate = {rate}",
            f"$s.SetOutputToWaveFile('{path}', $f)",
            f"$s.Speak('{text}')",
            "$s.SetOutputToNull()",
        ]
    script = Path(tempfile.mkdtemp()) / "speak.ps1"
    script.write_text("\n".join(lines), encoding="utf-8")
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        check=True,
        capture_output=True,
    )


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def wav_bytes(x: np.ndarray, rate: int = RATE, channels: int = 1) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def encode(data: bytes, fmt: str, *args: str) -> bytes:
    src = Path(tempfile.mkdtemp()) / "in.wav"
    dst = src.with_name(f"out.{fmt}")
    src.write_bytes(data)
    subprocess.run(
        [ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), *args, str(dst)], check=True
    )
    return dst.read_bytes()


_JITTER = np.random.default_rng(17)


def pad(x: np.ndarray) -> np.ndarray:
    """Random silence, volume, and a faint noise floor, so no two clips are identical
    (the speech synthesizer is deterministic)."""
    before, after = _JITTER.uniform(0.1, 0.45), _JITTER.uniform(0.2, 0.6)
    y = np.concatenate(
        [np.zeros(int(before * RATE)), x * _JITTER.uniform(0.6, 1.3), np.zeros(int(after * RATE))]
    )
    return y + _JITTER.normal(0, 0.0015, len(y))


def main(seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp())
    plan = {"yes": 12, "no": 10, "stop": 5}
    voices = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
    items, names = [], []
    for word, n in plan.items():
        for i in range(n):
            path = tmp / f"{word}_{i:03d}.wav"
            items.append((word, voices[i % 2], int(rng.integers(-2, 3)), path))
            names.append((word, i, path))
    extra = {
        "stop_under_yes": ("stop", voices[0]),
        "no_under_stop": ("no", voices[1]),
        "noisy": ("no", voices[0]),
    }
    for key, (word, voice) in extra.items():
        items.append((word, voice, 0, tmp / f"{key}.wav"))
    synthesize(items)

    files: dict[str, bytes] = {}
    clips: dict[str, np.ndarray] = {}
    for word, i, path in names:
        x = pad(read_wav(path))
        clips[f"{word}/{word}_{i:03d}"] = x
        files[f"speech_commands/{word}/{word}_{i:03d}.wav"] = wav_bytes(x)

    base = "speech_commands"
    files[f"{base}/no/no_009.wav"] = wav_bytes(np.zeros(RATE))  # silent
    loud = clips["yes/yes_003"] * 12  # clipped
    files[f"{base}/yes/yes_003.wav"] = wav_bytes(loud)
    speech = pad(read_wav(tmp / "noisy.wav"))
    noisy = speech + rng.normal(0, 0.12, len(speech))
    files[f"{base}/no/no_008.wav"] = wav_bytes(noisy)
    t = np.arange(int(2.5 * RATE)) / RATE  # music only: a chord progression, no speech
    chords = [(261.6, 329.6, 392.0), (220.0, 261.6, 329.6), (174.6, 220.0, 261.6)]
    music = np.concatenate(
        [sum(0.12 * np.sin(2 * np.pi * f * t[: len(t) // 3]) for f in chord) for chord in chords]
    )
    files[f"{base}/stop/stop_004.wav"] = wav_bytes(music)
    word = clips["yes/yes_011"]
    start = int(np.argmax(np.abs(word) > 0.05))  # cut from where the speech starts
    files[f"{base}/yes/yes_011.wav"] = wav_bytes(word[start : start + int(0.15 * RATE)])
    files[f"{base}/yes/yes_001_copy.wav"] = files[f"{base}/yes/yes_001.wav"]  # exact duplicate
    files[f"{base}/no/no_002_reencoded.mp3"] = encode(
        files[f"{base}/no/no_002.wav"], "mp3", "-b:a", "64k"
    )
    files[f"{base}/yes/yes_012.wav"] = wav_bytes(pad(read_wav(tmp / "stop_under_yes.wav")))
    files[f"{base}/stop/stop_005.wav"] = wav_bytes(pad(read_wav(tmp / "no_under_stop.wav")))
    stereo = np.repeat(clips["no/no_004"][:, None], 2, axis=1).flatten()
    files[f"{base}/no/no_004.wav"] = encode(wav_bytes(stereo, channels=2), "wav", "-ar", "44100")
    for name in ("yes/yes_006", "no/no_006"):
        del files[f"{base}/{name}.wav"]
        files[f"{base}/{name}.mp3"] = encode(wav_bytes(clips[name]), "mp3", "-b:a", "96k")
    good = files[f"{base}/yes/yes_007.wav"]
    files[f"{base}/yes/yes_007.wav"] = good[: len(good) // 3][:60]  # truncated: corrupt
    files[f"{base}/README.txt"] = (
        b"Toy speech commands dataset for MOSAIC EDA. Labels are folders.\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.writestr(name, files[name])
    return OUT


if __name__ == "__main__":
    print(main())

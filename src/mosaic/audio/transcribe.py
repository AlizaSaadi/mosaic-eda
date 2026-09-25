"""Speech-to-text for audio datasets: GPU Whisper on ZeroGPU, faster-whisper on CPU otherwise.

Whisper invents text for silence and music ("Thank you.", "♪"), so clips are only
treated as speech when they produce real words, and known hallucinations are dropped.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

CPU_MODEL = "base"
HALLUCINATIONS = {
    "you",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "bye",
    "subtitles by the amara.org community",
    "music",
    "applause",
    "silence",
}
_WORD = re.compile(r"[A-Za-zÀ-ɏЀ-ӿ؀-ۿ一-鿿']+")


@dataclass
class Transcript:
    path: str
    text: str
    words: int
    language: str = ""
    engine: str = ""


@dataclass
class TranscriptionResult:
    transcripts: list[Transcript] = field(default_factory=list)
    skipped_budget: list[str] = field(default_factory=list)
    engine: str = ""
    seconds: float = 0.0


def clean_text(text: str) -> str:
    text = re.sub(r"[♪♫*\[\]()]+", " ", text or "").strip()
    normalized = re.sub(r"[^\w\s']", "", text.lower()).strip()
    return "" if not normalized or normalized in HALLUCINATIONS else text


class CpuWhisper:
    """faster-whisper with its built-in Silero voice detection (loaded once, lazily)."""

    _model = None
    _lock = threading.Lock()

    @classmethod
    def model(cls):
        with cls._lock:
            if cls._model is None:
                from faster_whisper import WhisperModel

                cls._model = WhisperModel(CPU_MODEL, device="cpu", compute_type="int8")
            return cls._model

    def __call__(self, clips: list[np.ndarray]) -> list[tuple[str, str]]:
        """Whisper pads every input to 30 seconds, so short clips are packed into ~28-second
        windows (separated by silence) and segments are mapped back by timestamp."""
        model = self.model()
        texts: list[list[str]] = [[] for _ in clips]
        languages = [""] * len(clips)
        for window in pack_windows([len(c) for c in clips]):
            parts, spans, cursor = [], [], 0
            for index in window:
                parts += [clips[index], np.zeros(GAP, np.float32)]
                spans.append((index, cursor / RATE, (cursor + len(clips[index])) / RATE))
                cursor += len(clips[index]) + GAP
            segments, info = model.transcribe(
                np.concatenate(parts),
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=True,  # one segment can span several clips; words can't
            )
            for seg in segments:
                if seg.no_speech_prob >= 0.6:
                    continue
                for word in seg.words or []:
                    middle = (word.start + word.end) / 2
                    for index, start, end in spans:
                        if start - 0.2 <= middle <= end + 0.2:
                            texts[index].append(word.word.strip())
                            languages[index] = info.language or ""
                            break
        # Packing can drop repeated short words; retry empty clips that aren't silent one by one,
        # so no spoken clip is wrongly reported as "no speech"
        for index, clip in enumerate(clips):
            if texts[index] or not len(clip) or float(np.abs(clip).max()) < 0.02:
                continue
            segments, info = model.transcribe(
                clip, beam_size=1, vad_filter=True, condition_on_previous_text=False
            )
            texts[index] = [s.text for s in segments if s.no_speech_prob < 0.6]
            languages[index] = info.language or ""
        return [(" ".join(t).strip(), lang) for t, lang in zip(texts, languages, strict=True)]


RATE = 16000
GAP = RATE  # one second of silence between packed clips
WINDOW = 28 * RATE


def pack_windows(lengths: list[int]) -> list[list[int]]:
    """Group clip indexes into windows of at most ~28 seconds (long clips go alone)."""
    windows, current, used = [], [], 0
    for index, n in enumerate(lengths):
        if current and used + n + GAP > WINDOW:
            windows.append(current)
            current, used = [], 0
        current.append(index)
        used += n + GAP
    if current:
        windows.append(current)
    return windows


Engine = Callable[[list[np.ndarray]], list[tuple[str, str]]]


def default_engine() -> tuple[Engine, str]:
    from mosaic import gpu

    if gpu.gpu_available():
        return (lambda clips: [(t, "") for t in gpu.transcribe_gpu(clips)]), "whisper-gpu"
    return CpuWhisper(), f"faster-whisper-{CPU_MODEL}-cpu"


def transcribe_clips(
    clips: list[tuple[str, np.ndarray]],
    budget_seconds: float,
    engine: Engine | None = None,
    engine_name: str = "",
) -> TranscriptionResult:
    """Transcribe clips (path, 16 kHz audio) until the time budget is used up."""
    if engine is None:
        engine, engine_name = default_engine()
    result = TranscriptionResult(engine=engine_name)
    chosen, used = [], 0.0
    for path, audio in clips:
        seconds = len(audio) / 16000
        if used + seconds > budget_seconds:
            result.skipped_budget.append(path)
            continue
        chosen.append((path, audio))
        used += seconds
    result.seconds = round(used, 2)
    if not chosen:
        return result
    texts = engine([audio for _, audio in chosen])
    for (path, _), (text, language) in zip(chosen, texts, strict=True):
        text = clean_text(text)
        result.transcripts.append(
            Transcript(
                path=path,
                text=text,
                words=len(_WORD.findall(text)),
                language=language,
                engine=engine_name,
            )
        )
    return result


def label_matches(label: str, text: str) -> bool:
    """Does the transcript contain the folder label as a word (for spoken-word datasets)?"""
    words = {w.lower() for w in _WORD.findall(text)}
    return label.lower() in words

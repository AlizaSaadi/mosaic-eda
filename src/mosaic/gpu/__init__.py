"""GPU work that runs on Hugging Face ZeroGPU.

ZeroGPU Spaces must define at least one @spaces.GPU function at startup. On ZeroGPU
(SPACES_ZERO_GPU=true) a Whisper pipeline is loaded onto CUDA when this module is
imported, as ZeroGPU recommends; everywhere else the GPU path is off and transcription
uses faster-whisper on the CPU.
"""

from __future__ import annotations

import logging
import os

import spaces

log = logging.getLogger(__name__)

ON_ZERO_GPU = os.getenv("SPACES_ZERO_GPU", "").lower() in {"1", "true", "yes"}
WHISPER_GPU_MODEL = os.getenv("WHISPER_GPU_MODEL", "openai/whisper-base")
_ASR = None

if ON_ZERO_GPU:  # pragma: no cover - only runs on the Space
    try:
        import torch
        from transformers import pipeline

        _ASR = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_GPU_MODEL,
            device="cuda",
            dtype=torch.float16,  # transformers 5 renamed torch_dtype to dtype
        )
        log.info("Loaded %s for GPU transcription", WHISPER_GPU_MODEL)
    except Exception:  # the CPU path still works
        log.exception("GPU Whisper couldn't load; transcription will use the CPU")
        _ASR = None


@spaces.GPU(duration=10)
def gpu_ready() -> str:
    """Tiny GPU task used as a health check."""
    return "ready"


def gpu_available() -> bool:
    return _ASR is not None


def _gpu_seconds(clips: list) -> int:
    # short, predictable GPU slots get better queue priority on ZeroGPU
    return min(20 + 2 * len(clips), 120)


@spaces.GPU(duration=_gpu_seconds)
def transcribe_gpu(clips: list) -> list[str]:  # pragma: no cover - only runs on the Space
    """Transcribe 16 kHz mono float arrays on the GPU."""
    inputs = [{"raw": clip, "sampling_rate": 16000} for clip in clips]
    outputs = _ASR(inputs, batch_size=8, generate_kwargs={"task": "transcribe"})
    return [o["text"].strip() for o in outputs]

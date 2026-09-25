"""GPU work that runs on Hugging Face ZeroGPU.

ZeroGPU Spaces must define at least one @spaces.GPU function at startup. Outside
ZeroGPU the decorator does nothing and these functions run on CPU. Whisper
transcription moves here in Phase 5.
"""

from __future__ import annotations

import spaces


@spaces.GPU(duration=10)
def gpu_ready() -> str:
    """Tiny GPU task used as a health check."""
    return "ready"

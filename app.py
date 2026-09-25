"""MOSAIC EDA: Hugging Face Space entry point."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

for _stream in (sys.stdout, sys.stderr):  # CrewAI prints emoji; keep Windows consoles happy
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from mosaic.config import get_settings
from mosaic.gpu import gpu_ready  # noqa: F401  (ZeroGPU needs a @spaces.GPU function at startup)
from mosaic.logging_setup import setup_logging
from mosaic.ui.app import launch

if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    launch()

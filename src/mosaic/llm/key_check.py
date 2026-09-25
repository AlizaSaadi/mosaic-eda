"""Check that the Gemini key works with every model in both pools.

Run with:  python -m mosaic.llm.key_check
Uses one request per model (6 in total with the default pools).
"""

from __future__ import annotations

import sys
import time

from google import genai

from mosaic.config import get_settings

PROMPT = "Reply with the single word OK."


def check_model(client: genai.Client, model: str) -> tuple[bool, str, float]:
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=PROMPT)
        text = (response.text or "").strip()
        return True, text[:40] or "(empty reply)", time.perf_counter() - start
    except Exception as exc:  # report every failure the same way
        first_line = str(exc).splitlines()[0][:160] if str(exc) else ""
        return False, f"{type(exc).__name__}: {first_line}", time.perf_counter() - start


def main() -> int:
    settings = get_settings()
    if not settings.has_gemini_key:
        print("GEMINI_API_KEY is not set. Add it to the .env file in the project root.")
        return 1
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value().strip())
    pools = {"lite": settings.gemini_lite_pool, "flash": settings.gemini_flash_pool}
    failures = 0
    for pool, models in pools.items():
        for model in models:
            ok, detail, seconds = check_model(client, model)
            failures += not ok
            status = "ok  " if ok else "FAIL"
            print(f"[{status}] {pool:<5} {model:<24} {seconds:5.1f}s  {detail}")
    print("All models reachable." if not failures else f"{failures} model(s) failed.")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())

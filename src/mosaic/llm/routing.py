"""Which model pool each agent role uses, in order of preference."""

from __future__ import annotations

from mosaic.config import Settings
from mosaic.llm.quota import PoolRule, QuotaTracker


def build_routes(settings: Settings) -> dict[str, list[PoolRule]]:
    reserve = settings.flash_reserve_pct / 100
    return {
        # The Reviewer gets Flash until the pool is empty
        "reviewer": [PoolRule("flash"), PoolRule("lite")],
        # The Strategist gets Flash only while the pool is above the reserve
        "strategist": [PoolRule("flash", min_fraction_left=reserve), PoolRule("lite")],
        # Everything else runs on Lite
        "default": [PoolRule("lite")],
    }


def build_tracker(settings: Settings) -> QuotaTracker:
    margin = settings.rpm_safety_margin
    return QuotaTracker(
        {
            "lite": (settings.gemini_lite_pool, settings.lite_rpm - margin, settings.lite_rpd),
            "flash": (settings.gemini_flash_pool, settings.flash_rpm - margin, settings.flash_rpd),
        }
    )

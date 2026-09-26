"""Application settings, read from environment variables and the local .env file."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets
    gemini_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    # Model pools, in fallback order (comma-separated in env vars)
    gemini_lite_pool: Annotated[list[str], NoDecode] = Field(
        default=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    )
    gemini_flash_pool: Annotated[list[str], NoDecode] = Field(
        default=["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]
    )
    lite_rpm: int = 15
    lite_rpd: int = 500
    flash_rpm: int = 5
    flash_rpd: int = 20
    # Share of the Flash pool's daily total kept back for the Reviewer
    flash_reserve_pct: int = Field(default=30, ge=0, le=100)
    # Headroom below the published per-minute limit
    rpm_safety_margin: int = 1

    # Jobs and limits
    max_concurrent_jobs: int = 1
    max_input_mb: int = 200
    max_sampled_files: int = 500
    max_unzip_mb: int = 2048
    max_zip_entries: int = 10_000
    max_zip_depth: int = 2
    max_compression_ratio: float = 200.0
    max_transcribe_seconds: float = 600.0
    max_job_seconds: float = 600.0  # model calls stop after this; the run ends with a message
    workspace_root: Path = Path(tempfile.gettempdir()) / "mosaic-jobs"
    job_ttl_minutes: int = 60

    reports_repo: str | None = None
    log_level: str = "INFO"

    @field_validator("gemini_lite_pool", "gemini_flash_pool", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()

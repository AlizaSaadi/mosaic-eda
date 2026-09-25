"""Ingestion entry point: upload or URL in, file manifest out."""

from __future__ import annotations

import shutil
from pathlib import Path

from mosaic.config import Settings
from mosaic.ingest.downloader import SafeDownloader, safe_filename
from mosaic.ingest.manifest import build_manifest
from mosaic.ingest.models import FileManifest, IngestError, Modality
from mosaic.ingest.safe_unzip import UnzipLimits, safe_extract
from mosaic.ingest.sniffer import sniff
from mosaic.workspace import JobWorkspace


def limits_from(settings: Settings) -> UnzipLimits:
    return UnzipLimits(
        max_total_bytes=settings.max_unzip_mb << 20,
        max_entries=settings.max_zip_entries,
        max_depth=settings.max_zip_depth,
        max_ratio=settings.max_compression_ratio,
    )


def fetch_input(
    source: str | Path,
    ws: JobWorkspace,
    settings: Settings,
    downloader: SafeDownloader | None = None,
) -> Path:
    """Put the user's file into the workspace. `source` is an upload path or a URL."""
    max_bytes = settings.max_input_mb << 20
    if isinstance(source, str) and source.strip().lower().startswith(("http://", "https://")):
        downloader = downloader or SafeDownloader(max_bytes=max_bytes)
        return downloader.download(source.strip(), ws.input).path

    path = Path(source)
    if not path.is_file():
        raise IngestError("missing_upload", "The uploaded file couldn't be found. Try again.")
    if path.stat().st_size > max_bytes:
        raise IngestError(
            "too_large", f"The file is larger than the {settings.max_input_mb} MB limit."
        )
    if path.stat().st_size == 0:
        raise IngestError("empty_file", "The uploaded file is empty.")
    target = ws.input / safe_filename(path.name)
    shutil.copyfile(path, target)
    return target


def ingest(
    source: str | Path,
    ws: JobWorkspace,
    settings: Settings,
    downloader: SafeDownloader | None = None,
) -> FileManifest:
    """Fetch, unpack if needed, and inventory the input. Raises IngestError on bad input."""
    input_path = fetch_input(source, ws, settings, downloader)
    detection = sniff(input_path)
    if detection.modality == Modality.ARCHIVE:
        extracted = ws.work / "extracted"
        report = safe_extract(input_path, extracted, limits_from(settings))
        manifest = build_manifest(extracted, input_path.name, report.skipped)
    else:
        single = ws.work / "extracted" / input_path.name
        single.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, single)
        manifest = build_manifest(single.parent, input_path.name)
    if manifest.rejected:
        raise IngestError("no_supported_files", manifest.reject_reason)
    return manifest

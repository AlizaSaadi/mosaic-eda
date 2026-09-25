"""Detect what a file really is from its bytes. The extension is only a hint."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

import filetype

from mosaic.ingest.models import Detection, Modality

HEAD_BYTES = 64 * 1024

IMAGE_FORMATS = {"jpg", "png", "gif", "webp", "heic", "bmp", "tif"}
AUDIO_FORMATS = {"mp3", "wav", "m4a", "ogg", "flac", "aac"}
VIDEO_FORMATS = {"mp4", "mov", "webm", "mkv", "avi", "m4v"}
DOC_NAMES = re.compile(r"^(readme|license|licence|changelog|citation)(\.\w+)?$", re.I)
DOC_EXTENSIONS = {".md", ".rst"}
LOG_LINE = re.compile(
    r"^\s*(\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\[?(INFO|WARN|WARNING|ERROR|DEBUG)\]?\b)", re.I
)

_FILETYPE_MAP = {
    "jpg": (Modality.IMAGE, "jpg"),
    "png": (Modality.IMAGE, "png"),
    "gif": (Modality.IMAGE, "gif"),
    "webp": (Modality.IMAGE, "webp"),
    "heic": (Modality.IMAGE, "heic"),
    "bmp": (Modality.IMAGE, "bmp"),
    "tif": (Modality.IMAGE, "tif"),
    "mp3": (Modality.AUDIO, "mp3"),
    "wav": (Modality.AUDIO, "wav"),
    "m4a": (Modality.AUDIO, "m4a"),
    "ogg": (Modality.AUDIO, "ogg"),
    "flac": (Modality.AUDIO, "flac"),
    "aac": (Modality.AUDIO, "aac"),
    "mp4": (Modality.VIDEO, "mp4"),
    "mov": (Modality.VIDEO, "mov"),
    "webm": (Modality.VIDEO, "webm"),
    "mkv": (Modality.VIDEO, "mkv"),
    "avi": (Modality.VIDEO, "avi"),
    "m4v": (Modality.VIDEO, "m4v"),
}

OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _zip_kind(path: Path) -> Detection:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return Detection(modality=Modality.UNKNOWN, format="zip", detail="damaged zip file")
    if "[Content_Types].xml" in names and any(n.startswith("xl/") for n in names):
        return Detection(modality=Modality.TABLE, format="xlsx")
    if "[Content_Types].xml" in names:
        return Detection(
            modality=Modality.UNKNOWN, format="office", detail="Word or PowerPoint file"
        )
    return Detection(modality=Modality.ARCHIVE, format="zip")


def _decode(head: bytes) -> str | None:
    if b"\x00" in head[:8192]:
        return None  # binary, or UTF-16, which we don't accept in v1
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = head.decode(encoding)
        except UnicodeDecodeError:
            continue
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
        if text and printable / len(text) > 0.95:
            return text
    return None


def _table_delimiter(text: str) -> str | None:
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    if len(lines) < 2:
        return None
    if len(lines) > 2 and not text.endswith(("\n", "\r")):
        lines = lines[:-1]  # the last line may be cut off by the head limit
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t|")
    except csv.Error:
        return None
    rows = list(csv.reader(io.StringIO("\n".join(lines)), dialect))
    widths = {len(r) for r in rows}
    if len(widths) == 1 and next(iter(widths)) >= 2:
        return dialect.delimiter
    # tolerate a few ragged rows, but most rows must agree on a width of 2+
    common = max(widths, key=lambda w: sum(len(r) == w for r in rows))
    if common >= 2 and sum(len(r) == common for r in rows) / len(rows) >= 0.9:
        return dialect.delimiter
    return None


def _text_kind(text: str) -> Detection:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return Detection(modality=Modality.TEXT, format="txt", detail="empty")
    json_lines = 0
    for line in lines[:20]:
        try:
            json.loads(line)
            json_lines += line.lstrip().startswith(("{", "["))
        except ValueError:
            pass
    if json_lines >= max(2, int(min(len(lines), 20) * 0.9)):
        return Detection(modality=Modality.TABLE, format="jsonl")
    delimiter = _table_delimiter(text)
    if delimiter:
        name = {",": "csv", "\t": "tsv"}.get(delimiter, "csv")
        return Detection(modality=Modality.TABLE, format=name, detail=f"delimiter {delimiter!r}")
    log_share = sum(bool(LOG_LINE.match(ln)) for ln in lines[:50]) / min(len(lines), 50)
    if log_share >= 0.6:
        return Detection(modality=Modality.TEXT, format="log")
    return Detection(modality=Modality.TEXT, format="txt")


def sniff(path: Path) -> Detection:
    with path.open("rb") as handle:
        head = handle.read(HEAD_BYTES)
    ext = path.suffix.lower()
    if not head:
        return Detection(modality=Modality.UNKNOWN, format="empty", detail="empty file")

    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return _zip_kind(path)
    if head.startswith(OLE2_MAGIC):
        if ext in {".xls", ""}:
            return Detection(modality=Modality.TABLE, format="xls")
        return Detection(modality=Modality.UNKNOWN, format="ole2", detail="old Office file")

    kind = filetype.guess(head)
    if kind is not None:
        mapped = _FILETYPE_MAP.get(kind.extension)
        if mapped:
            modality, fmt = mapped
            detail = "" if ext.lstrip(".") in {fmt, "jpeg"} or not ext else f"extension says {ext}"
            return Detection(modality=modality, format=fmt, detail=detail)
        return Detection(modality=Modality.UNKNOWN, format=kind.extension, detail=kind.mime)

    text = _decode(head)
    if text is None:
        return Detection(
            modality=Modality.UNKNOWN, format="binary", detail="not a supported format"
        )
    if DOC_NAMES.match(path.name) or ext in DOC_EXTENSIONS:
        return Detection(modality=Modality.DOC, format=ext.lstrip(".") or "txt")
    detection = _text_kind(text)
    if ext and detection.format not in ext:
        detection.detail = (detection.detail + f"; extension says {ext}").strip("; ")
    return detection

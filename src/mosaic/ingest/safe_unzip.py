"""Extract zip files without trusting anything inside them.

Protects against path traversal ("zip slip"), absolute paths, symlinks, zip bombs
(checked from the headers and again while streaming, since headers can lie),
too many entries, encrypted entries, and endless nesting.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import stat
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from mosaic.ingest.models import IngestError, SkippedEntry

JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
JUNK_PREFIXES = ("__MACOSX/",)
_WINDOWS_BAD = re.compile(r'[<>:"|?*\x00-\x1f]')
_DRIVE = re.compile(r"^[A-Za-z]:")
COPY_CHUNK = 1 << 16


@dataclass(frozen=True)
class UnzipLimits:
    max_total_bytes: int = 2 << 30
    max_entries: int = 10_000
    max_depth: int = 2
    max_ratio: float = 200.0
    ratio_min_bytes: int = 1 << 20  # small files can have high ratios harmlessly


@dataclass
class UnzipReport:
    root: Path
    extracted: list[str] = field(default_factory=list)
    skipped: list[SkippedEntry] = field(default_factory=list)
    total_bytes: int = 0
    entries: int = 0


def _bomb() -> IngestError:
    return IngestError(
        "zip_bomb",
        "The zip expands to far more data than it contains, so it was rejected as unsafe.",
    )


def _entry_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not info.flag_bits & 0x800:  # not flagged as UTF-8: try to recover UTF-8 names
        with contextlib.suppress(UnicodeEncodeError, UnicodeDecodeError):
            name = name.encode("cp437").decode("utf-8")
    return unicodedata.normalize("NFC", name.replace("\\", "/"))


def safe_relative_path(name: str) -> PurePosixPath:
    """Validate an archive path. Raises IngestError for anything that could escape."""
    if name.startswith("/") or _DRIVE.match(name):
        raise IngestError("unsafe_zip", "The zip contains absolute file paths, so it was rejected.")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise IngestError("unsafe_zip", "The zip contains paths that point outside it ('../').")
    cleaned = [_WINDOWS_BAD.sub("_", p).rstrip(" .") or "_" for p in parts]
    return PurePosixPath(*cleaned)


def _is_junk(rel: PurePosixPath, raw: str) -> bool:
    return (
        raw.startswith(JUNK_PREFIXES) or rel.name.lower() in JUNK_NAMES or rel.name.startswith("._")
    )


def _unique(target: Path) -> Path:
    if not target.exists():
        return target
    for i in range(1, 10_000):
        candidate = target.with_name(f"{target.stem}_{i}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise IngestError("unsafe_zip", "The zip contains too many files with the same name.")


def safe_extract(
    zip_path: Path,
    dest: Path,
    limits: UnzipLimits | None = None,
    *,
    _depth: int = 1,
    _report: UnzipReport | None = None,
    _prefix: str = "",
) -> UnzipReport:
    limits = limits or UnzipLimits()
    report = _report or UnzipReport(root=dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = report.root.resolve()

    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise IngestError("bad_zip", "The zip file is damaged or incomplete.") from exc

    nested: list[Path] = []
    with archive:
        infos = archive.infolist()
        if report.entries + len(infos) > limits.max_entries:
            raise IngestError(
                "too_many_files",
                f"The zip has more than {limits.max_entries:,} entries, which is over the limit.",
            )
        for info in infos:
            report.entries += 1
            raw = _entry_name(info)
            if info.is_dir():
                safe_relative_path(raw)  # still reject unsafe directory names
                continue
            rel = safe_relative_path(raw)
            shown = f"{_prefix}{rel}"
            if _is_junk(rel, raw):
                continue
            if info.flag_bits & 0x1:
                raise IngestError(
                    "encrypted_zip",
                    "The zip is password-protected. Upload an unprotected zip instead.",
                )
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                report.skipped.append(
                    SkippedEntry(path=shown, reason="symbolic link (not allowed)")
                )
                continue
            if info.file_size > limits.ratio_min_bytes and info.file_size > limits.max_ratio * max(
                info.compress_size, 1
            ):
                raise _bomb()
            if report.total_bytes + info.file_size > limits.max_total_bytes:
                raise _bomb()

            target = (dest / Path(*rel.parts)).resolve()
            if root not in target.parents:
                raise IngestError("unsafe_zip", "The zip contains paths that point outside it.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target = _unique(target)

            written = 0
            try:
                with archive.open(info) as src, target.open("wb") as out:
                    while chunk := src.read(COPY_CHUNK):
                        written += len(chunk)
                        if report.total_bytes + written > limits.max_total_bytes:
                            raise _bomb()
                        out.write(chunk)
            except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
                target.unlink(missing_ok=True)
                raise IngestError("bad_zip", "The zip file is damaged or incomplete.") from exc
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            report.total_bytes += written

            if target.suffix.lower() == ".zip":
                nested.append(target)
            else:
                report.extracted.append(target.relative_to(root).as_posix())

    for inner in nested:
        shown = inner.relative_to(root).as_posix()
        if _depth >= limits.max_depth:
            report.skipped.append(SkippedEntry(path=shown, reason="zip nested too deeply"))
            inner.unlink(missing_ok=True)
            continue
        folder = _unique(inner.with_suffix(""))
        try:
            safe_extract(
                inner, folder, limits, _depth=_depth + 1, _report=report, _prefix=f"{shown}/"
            )
        except IngestError as exc:
            if exc.code in ("zip_bomb", "too_many_files"):
                raise
            shutil.rmtree(folder, ignore_errors=True)
            report.skipped.append(SkippedEntry(path=shown, reason=exc.message))
        finally:
            inner.unlink(missing_ok=True)
    return report

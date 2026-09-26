"""Inventory extracted files: detected type, size, and folder group for each file."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mosaic.ingest.models import (
    ANALYZABLE,
    FileEntry,
    FileManifest,
    Modality,
    SkippedEntry,
)
from mosaic.ingest.sniffer import sniff


def _group(rel_parts: tuple[str, ...]) -> str:
    return rel_parts[0] if len(rel_parts) > 1 else ""


def _strip_single_top_folder(files: list[FileEntry]) -> list[FileEntry]:
    """Zips often wrap everything in one folder ("dataset/cats/..."). Group one level down."""
    tops = {f.path.split("/")[0] for f in files}
    if len(tops) != 1 or not all("/" in f.path for f in files):
        return files
    for f in files:
        inner = f.path.split("/")[1:]
        f.group = inner[0] if len(inner) > 1 else ""
    return files


def build_manifest(
    root: Path,
    source_name: str,
    skipped: list[SkippedEntry] | None = None,
) -> FileManifest:
    root = root.resolve()
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    base = root.parent if root.is_file() else root

    files: list[FileEntry] = []
    skipped = list(skipped or [])
    for path in paths:
        rel = path.relative_to(base)
        detection = sniff(path)
        entry = FileEntry(
            path=rel.as_posix(),
            size=path.stat().st_size,
            modality=detection.modality,
            format=detection.format,
            group=_group(rel.parts),
            detail=detection.detail,
        )
        if detection.modality in (Modality.UNKNOWN, Modality.ARCHIVE):
            reason = detection.detail or f"unsupported format ({detection.format})"
            skipped.append(SkippedEntry(path=entry.path, reason=reason))
            continue
        files.append(entry)

    files = _strip_single_top_folder(files)
    manifest = FileManifest(
        root=str(base),
        source_name=source_name,
        files=files,
        skipped=skipped,
        total_bytes=sum(f.size for f in files),
    )
    return _summarize(manifest)


def subset_manifest(manifest: FileManifest, modality: Modality) -> FileManifest:
    """The part of a mixed manifest with one data type, for group mode.

    It's re-rooted at the folder all of those files share ("bundle/images/circles/a.png"
    becomes "circles/a.png" under "bundle/images"), so the part looks exactly like a
    single-type upload and its first folder level is the label.
    """
    files = [f.model_copy() for f in manifest.files if f.modality == modality]
    dirs = [f.path.split("/")[:-1] for f in files]
    common: list[str] = []
    for level in zip(*dirs, strict=False) if dirs else []:
        if len(set(level)) != 1:
            break
        common.append(level[0])
    prefix = "/".join(common)
    for f in files:
        f.path = f.path[len(prefix) + 1 :] if prefix else f.path
        parts = f.path.split("/")
        f.group = parts[0] if len(parts) > 1 else ""
    part = FileManifest(
        root=str(Path(manifest.root) / prefix) if prefix else manifest.root,
        source_name=f"{manifest.source_name} ({modality.value})",
        files=files,
        total_bytes=sum(f.size for f in files),
    )
    return _summarize(part)


def _summarize(manifest: FileManifest) -> FileManifest:
    """Counts, the dominant type, and whether folders look like labels."""
    files = manifest.files
    counts = Counter(f.modality for f in files)
    manifest.counts = dict(counts)

    analyzable = {m: n for m, n in counts.items() if m in ANALYZABLE}
    if not analyzable:
        manifest.rejected = True
        manifest.reject_reason = (
            "No supported files were found. MOSAIC accepts CSV, Excel, text, images, "
            "audio, and video, or a zip of them."
        )
        return manifest

    manifest.dominant = max(analyzable, key=lambda m: (analyzable[m], m != Modality.TEXT))
    # README-style notes are Modality.DOC, so they never make a zip "mixed"
    manifest.is_mixed = len(analyzable) > 1

    dominant_files = manifest.files_of(manifest.dominant)
    groups = Counter(f.group for f in dominant_files)
    manifest.groups = dict(groups)
    grouped = sum(n for g, n in groups.items() if g)
    manifest.folders_as_labels = (
        len([g for g in groups if g]) >= 2 and grouped / max(len(dominant_files), 1) >= 0.9
    )
    return manifest

"""Turn share links (Google Drive, Dropbox, Hugging Face, GitHub) into direct-download URLs."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

_DRIVE_FILE = re.compile(r"/file/d/([A-Za-z0-9_-]{10,})")


def drive_file_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname not in {"drive.google.com", "docs.google.com"}:
        return None
    match = _DRIVE_FILE.search(parsed.path)
    if match:
        return match.group(1)
    ids = parse_qs(parsed.query).get("id")
    return ids[0] if ids else None


def resolve_url(url: str) -> str:
    """Return a direct-download URL, or the original URL when no rule applies."""
    url = url.strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    file_id = drive_file_id(url)
    if file_id:
        return f"https://drive.usercontent.google.com/download?id={file_id}&export=download"

    if host in {"www.dropbox.com", "dropbox.com"}:
        query = parse_qs(parsed.query)
        query.pop("dl", None)
        query["dl"] = ["1"]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

    if host == "huggingface.co" and "/blob/" in parsed.path:
        return urlunparse(parsed._replace(path=parsed.path.replace("/blob/", "/resolve/", 1)))

    if host == "github.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            user, repo, _, ref, *rest = parts
            return f"https://raw.githubusercontent.com/{user}/{repo}/{ref}/{'/'.join(rest)}"

    return url

"""Stream a file from a URL to disk with SSRF, size, and content checks."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlencode, urljoin, urlparse

import httpx

from mosaic.ingest.models import IngestError
from mosaic.ingest.net_guard import Resolver, check_url, is_public_ip, system_resolver
from mosaic.ingest.url_resolver import resolve_url

USER_AGENT = "MOSAIC-EDA/0.1 (+https://huggingface.co/spaces/lizconquers/mosaic-eda)"
CHUNK = 1 << 16
MAX_HTML_BYTES = 2 << 20
DRIVE_HOSTS = {"drive.google.com", "drive.usercontent.google.com", "docs.google.com"}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")
_CD_STAR = re.compile(r"filename\*\s*=\s*[^']*'[^']*'([^;]+)", re.I)
_CD_PLAIN = re.compile(r'filename\s*=\s*"?([^";]+)"?', re.I)
_FORM_ACTION = re.compile(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', re.I)
_HIDDEN_INPUT = re.compile(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"', re.I)


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    filename: str
    size: int
    final_url: str
    content_type: str


def safe_filename(name: str, fallback: str = "download") -> str:
    name = Path(unquote(name).replace("\\", "/")).name
    name = _SAFE_NAME.sub("_", name).strip(" .")
    return (name or fallback)[:120]


def filename_from_headers(headers: httpx.Headers, url: str) -> str:
    disposition = headers.get("content-disposition", "")
    match = _CD_STAR.search(disposition) or _CD_PLAIN.search(disposition)
    if match:
        return safe_filename(match.group(1))
    return safe_filename(urlparse(url).path.rsplit("/", 1)[-1])


def looks_like_html(content_type: str, head: bytes) -> bool:
    if "text/html" in content_type.lower():
        return True
    start = head[:1024].lstrip().lower()
    return start.startswith((b"<!doctype html", b"<html"))


def drive_confirm_url(page: str) -> str | None:
    """Build the 'download anyway' URL from Drive's 'can't scan for viruses' page."""
    action = _FORM_ACTION.search(page)
    if not action:
        return None
    params = {html.unescape(k): html.unescape(v) for k, v in _HIDDEN_INPUT.findall(page)}
    if "id" not in params:
        return None
    return f"{html.unescape(action.group(1))}?{urlencode(params)}"


class SafeDownloader:
    def __init__(
        self,
        *,
        max_bytes: int,
        timeout: float = 60.0,
        max_redirects: int = 5,
        resolver: Resolver = system_resolver,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.resolver = resolver
        self.transport = transport

    def _check_peer(self, response: httpx.Response) -> None:
        """Re-check the address actually connected to (defends against DNS rebinding)."""
        stream = response.extensions.get("network_stream")
        if stream is None:
            return
        server = stream.get_extra_info("server_addr")
        if server and not is_public_ip(str(server[0])):
            raise IngestError(
                "private_address",
                "That link points to a private or local network address, which isn't allowed.",
            )

    def _too_big(self) -> IngestError:
        return IngestError(
            "too_large",
            f"The file is larger than the {self.max_bytes // (1 << 20)} MB limit.",
        )

    def download(self, url: str, dest_dir: Path) -> DownloadResult:
        url = resolve_url(url)
        dest_dir.mkdir(parents=True, exist_ok=True)
        client = httpx.Client(
            transport=self.transport,
            follow_redirects=False,
            timeout=self.timeout,
            trust_env=False,  # no proxies: the peer check must see the real server
            headers={"User-Agent": USER_AGENT},
        )
        with client:
            for _ in range(self.max_redirects + 1):
                check_url(url, self.resolver)
                with client.stream("GET", url) as response:
                    self._check_peer(response)
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise IngestError(
                                "bad_redirect", "The server sent an invalid redirect."
                            )
                        url = urljoin(url, location)
                        continue
                    self._check_status(response)
                    result = self._save(response, url, dest_dir)
                    if isinstance(result, DownloadResult):
                        return result
                    url = result  # Drive's confirm link: fetch it on the next pass
        raise IngestError("too_many_redirects", "The link redirected too many times.")

    def _check_status(self, response: httpx.Response) -> None:
        code = response.status_code
        if code in (401, 403):
            raise IngestError(
                "not_public",
                "The file isn't public. Set its sharing to 'Anyone with the link' and try again.",
            )
        if code == 404:
            raise IngestError("not_found", "Nothing was found at that link (404).")
        if code == 429:
            raise IngestError(
                "remote_rate_limited", "The file host is rate-limiting downloads. Try later."
            )
        if code >= 400:
            raise IngestError("http_error", f"The file host returned an error ({code}).")

    def _save(self, response: httpx.Response, url: str, dest_dir: Path) -> DownloadResult | str:
        """Write the body to disk. Returns a follow-up URL instead for Drive's confirm page."""
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            raise self._too_big()

        content_type = response.headers.get("content-type", "")
        chunks = response.iter_bytes(CHUNK)
        first = next(chunks, b"")

        if looks_like_html(content_type, first):
            page = first
            for chunk in chunks:
                page += chunk
                if len(page) > MAX_HTML_BYTES:
                    break
            text = page.decode("utf-8", errors="replace")
            if (urlparse(url).hostname or "") in DRIVE_HOSTS:
                confirm = drive_confirm_url(text)
                if confirm:
                    return confirm
                raise IngestError(
                    "not_public",
                    "Google Drive didn't return the file. Set sharing to 'Anyone with the link'.",
                )
            raise IngestError(
                "html_page",
                "That link opens a web page, not a file. Use a direct download link.",
            )

        filename = filename_from_headers(response.headers, url)
        target = dest_dir / filename
        size = 0
        try:
            with target.open("wb") as handle:
                size += len(first)
                handle.write(first)
                for chunk in chunks:
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise self._too_big()
                    handle.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        if size > self.max_bytes:
            target.unlink(missing_ok=True)
            raise self._too_big()
        if size == 0:
            target.unlink(missing_ok=True)
            raise IngestError("empty_file", "The link returned an empty file.")
        return DownloadResult(target, filename, size, url, content_type)

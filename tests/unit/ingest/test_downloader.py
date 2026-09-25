import httpx
import pytest

from mosaic.ingest.downloader import SafeDownloader, drive_confirm_url, safe_filename
from mosaic.ingest.models import IngestError

from .fakes import CSV

PUBLIC = lambda host, port: ["93.184.215.14"]  # noqa: E731


def downloader(handler, *, max_bytes=1 << 20, resolver=PUBLIC):
    return SafeDownloader(
        max_bytes=max_bytes, resolver=resolver, transport=httpx.MockTransport(handler)
    )


def test_downloads_file_with_name_from_headers(tmp_path):
    def handler(request):
        return httpx.Response(
            200, content=CSV, headers={"content-disposition": 'attachment; filename="sales.csv"'}
        )

    result = downloader(handler).download("https://example.com/get?id=1", tmp_path)
    assert result.filename == "sales.csv"
    assert result.path.read_bytes() == CSV


def test_name_falls_back_to_url_path(tmp_path):
    result = downloader(lambda r: httpx.Response(200, content=CSV)).download(
        "https://example.com/data/my%20file.csv", tmp_path
    )
    assert result.filename == "my file.csv"


def test_rejects_declared_size_over_limit(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"x" * 10, headers={"content-length": str(5 << 20)})

    with pytest.raises(IngestError) as info:
        downloader(handler).download("https://example.com/big.csv", tmp_path)
    assert info.value.code == "too_large"


def test_rejects_streamed_size_over_limit_and_cleans_up(tmp_path):
    def handler(request):
        # a generator body has no content-length, so only the streaming check can catch it
        return httpx.Response(200, content=(b"a,b\n" * 1000 for _ in range(100)))

    with pytest.raises(IngestError) as info:
        downloader(handler, max_bytes=100_000).download("https://example.com/big.csv", tmp_path)
    assert info.value.code == "too_large"
    assert list(tmp_path.iterdir()) == []


def test_rejects_html_page(tmp_path):
    def handler(request):
        return httpx.Response(
            200, content=b"<!DOCTYPE html><html>login</html>", headers={"content-type": "text/html"}
        )

    with pytest.raises(IngestError) as info:
        downloader(handler).download("https://example.com/share/abc", tmp_path)
    assert info.value.code == "html_page"


def test_redirect_to_private_address_is_blocked(tmp_path):
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://internal.local/secret"})
        return httpx.Response(200, content=b"secret")

    def resolver(host, port):
        return ["10.1.2.3"] if host == "internal.local" else ["93.184.215.14"]

    with pytest.raises(IngestError) as info:
        downloader(handler, resolver=resolver).download("https://example.com/f", tmp_path)
    assert info.value.code == "private_address"


def test_follows_safe_redirects(tmp_path):
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "/final/data.csv"})
        return httpx.Response(200, content=CSV)

    result = downloader(handler).download("https://example.com/start", tmp_path)
    assert result.final_url == "https://example.com/final/data.csv"


def test_too_many_redirects(tmp_path):
    def handler(request):
        return httpx.Response(302, headers={"location": "/loop"})

    with pytest.raises(IngestError) as info:
        downloader(handler).download("https://example.com/loop", tmp_path)
    assert info.value.code == "too_many_redirects"


@pytest.mark.parametrize(
    "status, code", [(403, "not_public"), (404, "not_found"), (500, "http_error")]
)
def test_http_errors_have_friendly_codes(tmp_path, status, code):
    with pytest.raises(IngestError) as info:
        downloader(lambda r: httpx.Response(status)).download("https://example.com/x", tmp_path)
    assert info.value.code == code


DRIVE_WARNING = """<html><body>
<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
<input type="hidden" name="id" value="FILEID123">
<input type="hidden" name="export" value="download">
<input type="hidden" name="confirm" value="t">
<input type="hidden" name="uuid" value="u-1">
</form></body></html>"""


def test_drive_confirm_page_is_followed(tmp_path):
    def handler(request):
        if "confirm" in request.url.params:
            return httpx.Response(
                200, content=CSV, headers={"content-disposition": 'attachment; filename="big.csv"'}
            )
        return httpx.Response(
            200, content=DRIVE_WARNING.encode(), headers={"content-type": "text/html"}
        )

    result = downloader(handler).download(
        "https://drive.google.com/file/d/FILEID1234567/view", tmp_path
    )
    assert result.filename == "big.csv"
    assert result.path.read_bytes() == CSV


def test_private_drive_file_gives_sharing_hint(tmp_path):
    def handler(request):
        return httpx.Response(
            200, content=b"<html>Sign in</html>", headers={"content-type": "text/html"}
        )

    with pytest.raises(IngestError) as info:
        downloader(handler).download("https://drive.google.com/file/d/FILEID1234567/view", tmp_path)
    assert info.value.code == "not_public"


def test_drive_confirm_url_parsing():
    url = drive_confirm_url(DRIVE_WARNING)
    assert url.startswith("https://drive.usercontent.google.com/download?")
    assert "id=FILEID123" in url and "confirm=t" in url


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../etc/passwd", "passwd"),
        ("a\\b\\evil.csv", "evil.csv"),
        ("", "download"),
        ("we:ird*name?.csv", "we_ird_name_.csv"),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_empty_response_is_rejected(tmp_path):
    with pytest.raises(IngestError) as info:
        downloader(lambda r: httpx.Response(200, content=b"")).download(
            "https://example.com/e", tmp_path
        )
    assert info.value.code == "empty_file"

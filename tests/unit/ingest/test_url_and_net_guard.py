import pytest

from mosaic.ingest.models import IngestError
from mosaic.ingest.net_guard import check_url, is_public_ip
from mosaic.ingest.url_resolver import resolve_url

PUBLIC = lambda host, port: ["93.184.215.14"]  # noqa: E731


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing",
            "https://drive.usercontent.google.com/download?id=1AbCdEfGhIjKlMnOp&export=download",
        ),
        (
            "https://drive.google.com/open?id=1AbCdEfGhIjKlMnOp",
            "https://drive.usercontent.google.com/download?id=1AbCdEfGhIjKlMnOp&export=download",
        ),
        (
            "https://www.dropbox.com/scl/fi/abc/data.csv?rlkey=xyz&dl=0",
            "https://www.dropbox.com/scl/fi/abc/data.csv?rlkey=xyz&dl=1",
        ),
        (
            "https://huggingface.co/datasets/u/d/blob/main/data/train.csv",
            "https://huggingface.co/datasets/u/d/resolve/main/data/train.csv",
        ),
        (
            "https://github.com/u/r/blob/main/data/iris.csv",
            "https://raw.githubusercontent.com/u/r/main/data/iris.csv",
        ),
        ("https://example.com/files/data.zip", "https://example.com/files/data.zip"),
    ],
)
def test_resolve_url(url, expected):
    assert resolve_url(url) == expected


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.10",
        "169.254.169.254",
        "0.0.0.0",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "100.64.0.1",
        "224.0.0.1",
    ],
)
def test_private_and_special_addresses_are_not_public(address):
    assert not is_public_ip(address)


def test_public_address_is_public():
    assert is_public_ip("8.8.8.8")
    assert is_public_ip("2606:4700:4700::1111")


@pytest.mark.parametrize(
    "url, code",
    [
        ("file:///etc/passwd", "bad_scheme"),
        ("ftp://example.com/data.csv", "bad_scheme"),
        ("gopher://example.com/", "bad_scheme"),
        ("https://user:pass@example.com/data.csv", "credentials_in_url"),
        ("https://example.com:22/data.csv", "blocked_port"),
        ("http://", "bad_url"),
    ],
)
def test_rejected_urls(url, code):
    with pytest.raises(IngestError) as info:
        check_url(url, PUBLIC)
    assert info.value.code == code


@pytest.mark.parametrize(
    "resolved",
    [["127.0.0.1"], ["169.254.169.254"], ["93.184.215.14", "10.0.0.1"]],
)
def test_host_resolving_to_private_ip_is_blocked(resolved):
    with pytest.raises(IngestError) as info:
        check_url("https://innocent-looking.example/data.csv", lambda h, p: resolved)
    assert info.value.code == "private_address"


def test_allowed_url_passes():
    host, port, ips = check_url("https://example.com:8080/data.csv", PUBLIC)
    assert (host, port, ips) == ("example.com", 8080, ["93.184.215.14"])

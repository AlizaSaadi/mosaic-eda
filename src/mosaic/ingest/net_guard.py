"""Block URLs that point inside private networks (SSRF protection).

A URL is allowed only if it uses http or https, an allowed port, and every IP
address its host resolves to is public. The downloader calls this for the first
URL and again for every redirect, then checks the address it actually connected to.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlparse

from mosaic.ingest.models import IngestError

ALLOWED_SCHEMES = {"http", "https"}
# Hugging Face Spaces only allow outbound traffic on these ports
ALLOWED_PORTS = {80, 443, 8080}

Resolver = Callable[[str, int], list[str]]


def system_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise IngestError(
            "dns_failed", f"Couldn't find the server '{host}'. Check the link."
        ) from exc
    return sorted({info[4][0] for info in infos})


def is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def check_url(url: str, resolver: Resolver = system_resolver) -> tuple[str, int, list[str]]:
    """Validate a URL. Returns (host, port, resolved IPs) or raises IngestError."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise IngestError("bad_scheme", "Only http and https links are supported.")
    if parsed.username or parsed.password:
        raise IngestError(
            "credentials_in_url", "Links with a username or password aren't supported."
        )
    host = parsed.hostname
    if not host:
        raise IngestError("bad_url", "That link has no server name. Check the link.")
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise IngestError("bad_url", "That link has an invalid port number.") from exc
    if port not in ALLOWED_PORTS:
        raise IngestError(
            "blocked_port",
            f"Links on port {port} can't be reached from this app. Use a link on port 80 or 443.",
        )
    addresses = resolver(host, port)
    if not addresses:
        raise IngestError("dns_failed", f"Couldn't find the server '{host}'. Check the link.")
    for address in addresses:
        if not is_public_ip(address):
            raise IngestError(
                "private_address",
                "That link points to a private or local network address, which isn't allowed.",
            )
    return host, port, addresses

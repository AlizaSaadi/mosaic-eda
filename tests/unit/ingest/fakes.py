"""Tiny stand-in files: just enough header bytes for type detection."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 64
JPG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 64
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64
MP3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
XLS = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
CSV = b"id,name,price\n1,apple,1.20\n2,pear,0.80\n3,plum,2.10\n"
EURO_CSV = b"id;name;price\n1;apple;1,20\n2;pear;0,80\n3;plum;2,10\n"
PROSE = b"The quick brown fox jumps over the lazy dog. " * 20
JSONL = b'{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n{"a": 3, "b": "z"}\n'
LOG = b"".join(f"2026-09-25 10:0{i}:00 INFO request handled in {i}ms\n".encode() for i in range(8))


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def xlsx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("xl/workbook.xml", "<workbook/>")
    return buf.getvalue()


def make_zip(path: Path, entries: dict[str, bytes], *, compression=zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", compression) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()

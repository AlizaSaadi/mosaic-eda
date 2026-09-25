"""Malicious and awkward zip files."""

import stat
import zipfile

import pytest

from mosaic.ingest.models import IngestError
from mosaic.ingest.safe_unzip import UnzipLimits, safe_extract, safe_relative_path

from .fakes import CSV, PNG, make_zip, zip_bytes


def extract(tmp_path, entries, limits=None):
    z = make_zip(tmp_path / "in.zip", entries)
    return safe_extract(z, tmp_path / "out", limits)


def test_normal_zip_extracts_with_folders(tmp_path):
    report = extract(tmp_path, {"cats/1.png": PNG, "dogs/2.png": PNG, "meta.csv": CSV})
    assert sorted(report.extracted) == ["cats/1.png", "dogs/2.png", "meta.csv"]
    assert (tmp_path / "out" / "cats" / "1.png").read_bytes() == PNG


@pytest.mark.parametrize(
    "name", ["../evil.txt", "a/../../evil.txt", "/etc/evil", "C:/evil.txt", "..\\evil.txt"]
)
def test_zip_slip_is_rejected(tmp_path, name):
    with pytest.raises(IngestError) as info:
        extract(tmp_path, {name: b"pwned"})
    assert info.value.code == "unsafe_zip"
    assert not (tmp_path / "evil.txt").exists()


def test_junk_files_are_ignored(tmp_path):
    report = extract(
        tmp_path,
        {
            "__MACOSX/._a.png": b"x",
            "a/.DS_Store": b"x",
            "Thumbs.db": b"x",
            "a/._b.png": b"x",
            "a/b.png": PNG,
        },
    )
    assert report.extracted == ["a/b.png"]


def test_symlink_entries_are_skipped(tmp_path):
    z = tmp_path / "in.zip"
    with zipfile.ZipFile(z, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "/etc/passwd")
        archive.writestr("real.csv", CSV)
    report = safe_extract(z, tmp_path / "out")
    assert report.extracted == ["real.csv"]
    assert report.skipped[0].reason.startswith("symbolic link")
    assert not (tmp_path / "out" / "link").exists()


def test_encrypted_entry_is_rejected(tmp_path):
    z = make_zip(tmp_path / "in.zip", {"secret.csv": CSV})
    # zipfile can't write encrypted entries, so set the "encrypted" bit in the headers directly
    data = bytearray(z.read_bytes())
    data[6] |= 0x1  # local file header flags
    cd = data.rfind(b"PK")
    data[cd + 8] |= 0x1  # central directory flags
    z.write_bytes(bytes(data))
    with pytest.raises(IngestError) as info:
        safe_extract(z, tmp_path / "out")
    assert info.value.code == "encrypted_zip"


def test_high_compression_ratio_is_a_bomb(tmp_path):
    limits = UnzipLimits(max_ratio=50, ratio_min_bytes=1000)
    with pytest.raises(IngestError) as info:
        extract(tmp_path, {"zeros.bin": b"\x00" * 2_000_000}, limits)
    assert info.value.code == "zip_bomb"


def test_total_size_limit_is_enforced_while_streaming(tmp_path):
    limits = UnzipLimits(max_total_bytes=50_000, max_ratio=1e9)
    entries = {f"f{i}.csv": b"x" * 20_000 for i in range(5)}
    with pytest.raises(IngestError) as info:
        extract(tmp_path, entries, limits)
    assert info.value.code == "zip_bomb"


def test_lying_header_is_caught_while_streaming(tmp_path):
    z = make_zip(tmp_path / "in.zip", {"big.csv": b"y" * 100_000}, compression=zipfile.ZIP_STORED)
    # Patch the central directory to claim a tiny file
    data = bytearray(z.read_bytes())
    cd = data.rfind(b"PK\x01\x02")
    data[cd + 24 : cd + 28] = (10).to_bytes(4, "little")  # uncompressed size field
    z.write_bytes(bytes(data))
    limits = UnzipLimits(max_total_bytes=50_000)
    with pytest.raises((IngestError, zipfile.BadZipFile)):
        safe_extract(z, tmp_path / "out", limits)
    assert not (tmp_path / "out" / "big.csv").exists()


def test_too_many_entries(tmp_path):
    with pytest.raises(IngestError) as info:
        extract(tmp_path, {f"{i}.txt": b"x" for i in range(30)}, UnzipLimits(max_entries=20))
    assert info.value.code == "too_many_files"


def test_nested_zip_extracted_once(tmp_path):
    inner = zip_bytes({"b.csv": CSV})
    report = extract(tmp_path, {"a.csv": CSV, "more/inner.zip": inner})
    assert sorted(report.extracted) == ["a.csv", "more/inner/b.csv"]
    assert not (tmp_path / "out" / "more" / "inner.zip").exists()


def test_deeper_nesting_is_skipped(tmp_path):
    level3 = zip_bytes({"deep.csv": CSV})
    level2 = zip_bytes({"l3.zip": level3, "mid.csv": CSV})
    report = extract(tmp_path, {"l2.zip": level2})
    assert report.extracted == ["l2/mid.csv"]
    assert report.skipped[0].path == "l2/l3.zip"
    assert "nested too deeply" in report.skipped[0].reason


def test_unsafe_inner_zip_is_skipped_not_extracted(tmp_path):
    bad_inner = zip_bytes({"../../escape.txt": b"pwned"})
    report = extract(tmp_path, {"ok.csv": CSV, "bad.zip": bad_inner})
    assert report.extracted == ["ok.csv"]
    assert report.skipped[0].path == "bad.zip"
    assert not any(p.name == "escape.txt" for p in tmp_path.rglob("*"))


def test_damaged_zip(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(IngestError) as info:
        safe_extract(bad, tmp_path / "out")
    assert info.value.code == "bad_zip"


@pytest.mark.filterwarnings("ignore:Duplicate name")
def test_duplicate_names_get_suffixes(tmp_path):
    z = tmp_path / "in.zip"
    with zipfile.ZipFile(z, "w") as archive:
        archive.writestr("a.csv", CSV)
        archive.writestr("a.csv", CSV)
    report = safe_extract(z, tmp_path / "out")
    assert sorted(report.extracted) == ["a.csv", "a_1.csv"]


def test_windows_unsafe_characters_are_replaced():
    assert str(safe_relative_path('data/we"ird:name?.csv')) == "data/we_ird_name_.csv"

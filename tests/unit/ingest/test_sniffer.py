import pytest

from mosaic.ingest.models import Modality
from mosaic.ingest.sniffer import sniff

from .fakes import CSV, EURO_CSV, JPG, JSONL, LOG, MP3, MP4, PNG, PROSE, WAV, XLS, write, xlsx_bytes


@pytest.mark.parametrize(
    "name, data, modality, fmt",
    [
        ("a.png", PNG, Modality.IMAGE, "png"),
        ("a.jpg", JPG, Modality.IMAGE, "jpg"),
        ("a.wav", WAV, Modality.AUDIO, "wav"),
        ("a.mp3", MP3, Modality.AUDIO, "mp3"),
        ("a.mp4", MP4, Modality.VIDEO, "mp4"),
        ("a.xls", XLS, Modality.TABLE, "xls"),
        ("a.csv", CSV, Modality.TABLE, "csv"),
        ("notes.txt", PROSE, Modality.TEXT, "txt"),
        ("events.jsonl", JSONL, Modality.TABLE, "jsonl"),
        ("server.log", LOG, Modality.TEXT, "log"),
    ],
)
def test_detects_by_content(tmp_path, name, data, modality, fmt):
    d = sniff(write(tmp_path / name, data))
    assert (d.modality, d.format) == (modality, fmt)


def test_xlsx_is_table_not_archive(tmp_path):
    d = sniff(write(tmp_path / "book.xlsx", xlsx_bytes()))
    assert (d.modality, d.format) == (Modality.TABLE, "xlsx")


def test_csv_hiding_in_txt_is_a_table(tmp_path):
    d = sniff(write(tmp_path / "export.txt", EURO_CSV))
    assert d.modality == Modality.TABLE
    assert "';'" in d.detail and "extension says .txt" in d.detail


def test_image_with_wrong_extension_is_an_image(tmp_path):
    d = sniff(write(tmp_path / "data.csv", PNG))
    assert d.modality == Modality.IMAGE
    assert "extension says .csv" in d.detail


def test_binary_junk_named_txt_is_unknown(tmp_path):
    d = sniff(write(tmp_path / "data.txt", bytes(range(256)) * 10))
    assert d.modality == Modality.UNKNOWN


def test_empty_file_is_unknown(tmp_path):
    assert sniff(write(tmp_path / "e.csv", b"")).modality == Modality.UNKNOWN


def test_readme_is_doc(tmp_path):
    assert (
        sniff(write(tmp_path / "README.md", b"# My dataset\nSome notes.")).modality == Modality.DOC
    )


def test_single_line_text_is_not_a_table(tmp_path):
    assert sniff(write(tmp_path / "one.txt", b"hello, world")).modality == Modality.TEXT


def test_latin1_csv_is_readable(tmp_path):
    data = "city;temp\nMünchen;12,5\nZürich;9,1\nKöln;11,0\n".encode("cp1252")
    assert sniff(write(tmp_path / "cities.csv", data)).modality == Modality.TABLE

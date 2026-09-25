import pytest

from mosaic.config import Settings
from mosaic.ingest.manifest import build_manifest
from mosaic.ingest.models import FileEntry, IngestError, Modality
from mosaic.ingest.sampler import allocate, stratified_sample
from mosaic.ingest.service import ingest
from mosaic.workspace import create_workspace

from .fakes import CSV, PNG, PROSE, WAV, make_zip, write


def test_image_classes_from_folders(tmp_path):
    for i in range(6):
        write(tmp_path / "ds" / "cats" / f"{i}.png", PNG)
    for i in range(3):
        write(tmp_path / "ds" / "dogs" / f"{i}.png", PNG)
    write(tmp_path / "ds" / "README.md", b"# notes")
    m = build_manifest(tmp_path / "ds", "ds.zip")
    assert m.dominant == Modality.IMAGE
    assert m.is_mixed is False  # the README is a DOC, not a second type
    assert m.folders_as_labels is True
    assert m.groups == {"cats": 6, "dogs": 3}


def test_single_wrapper_folder_is_looked_through(tmp_path):
    for label in ("a", "b"):
        for i in range(2):
            write(tmp_path / "root" / "dataset" / label / f"{i}.png", PNG)
    m = build_manifest(tmp_path / "root", "x.zip")
    assert m.groups == {"a": 2, "b": 2}
    assert m.folders_as_labels


def test_images_plus_metadata_csv_is_mixed(tmp_path):
    for i in range(5):
        write(tmp_path / "ds" / "img" / f"{i}.png", PNG)
    write(tmp_path / "ds" / "metadata.csv", CSV)
    m = build_manifest(tmp_path / "ds", "ds.zip")
    assert m.is_mixed and m.dominant == Modality.IMAGE
    assert m.counts == {Modality.IMAGE: 5, Modality.TABLE: 1}


def test_unsupported_files_are_listed_as_skipped(tmp_path):
    write(tmp_path / "ds" / "a.csv", CSV)
    write(tmp_path / "ds" / "blob.bin", bytes(range(256)) * 4)
    m = build_manifest(tmp_path / "ds", "ds.zip")
    assert [s.path for s in m.skipped] == ["blob.bin"]


def test_nothing_supported_is_rejected(tmp_path):
    write(tmp_path / "ds" / "blob.bin", bytes(range(256)) * 4)
    m = build_manifest(tmp_path / "ds", "ds.zip")
    assert m.rejected and "CSV" in m.reject_reason


def test_allocate_is_proportional_with_minimum():
    quota = allocate({"big": 900, "mid": 90, "tiny": 2}, total=100, min_per_group=5)
    assert sum(quota.values()) == 100
    assert quota["tiny"] == 2  # all of it, below the minimum
    assert quota["big"] > quota["mid"] >= 5


def test_allocate_returns_everything_under_the_cap():
    assert allocate({"a": 3, "b": 4}, total=100, min_per_group=5) == {"a": 3, "b": 4}


def test_allocate_more_groups_than_slots():
    quota = allocate({f"g{i}": 10 for i in range(20)}, total=8, min_per_group=1)
    assert sum(quota.values()) == 8


def entries(n_by_group):
    return [
        FileEntry(path=f"{g}/{i}.png", size=1, modality=Modality.IMAGE, format="png", group=g)
        for g, n in n_by_group.items()
        for i in range(n)
    ]


def test_sample_is_stratified_and_deterministic():
    files = entries({"cats": 800, "dogs": 150, "rare": 3})
    s1 = stratified_sample(files, Modality.IMAGE, 100, seed=7)
    s2 = stratified_sample(files, Modality.IMAGE, 100, seed=7)
    assert [f.path for f in s1.files] == [f.path for f in s2.files]
    assert len(s1.files) == 100 and s1.is_sample
    assert s1.per_group["rare"] == 3
    assert round(s1.ratio, 3) == round(100 / 953, 3)


def test_sample_keeps_everything_when_small():
    s = stratified_sample(entries({"a": 4}), Modality.IMAGE, 500)
    assert len(s.files) == 4 and not s.is_sample


# ---- service (upload path end to end) ----


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, workspace_root=tmp_path / "jobs", max_input_mb=1)


def test_ingest_zip_upload(tmp_path, settings):
    z = make_zip(
        tmp_path / "upload.zip",
        {"audio/a.wav": WAV, "audio/b.wav": WAV, "README.txt": PROSE, "notes/interview.txt": PROSE},
    )
    ws = create_workspace(settings.workspace_root)
    m = ingest(z, ws, settings)
    assert m.counts[Modality.AUDIO] == 2
    assert m.counts[Modality.DOC] == 1  # README is a note, not data
    assert m.is_mixed  # interview.txt is real text data alongside the audio
    assert m.source_name == "upload.zip"


def test_ingest_single_csv_upload(tmp_path, settings):
    ws = create_workspace(settings.workspace_root)
    m = ingest(write(tmp_path / "sales.csv", CSV), ws, settings)
    assert m.dominant == Modality.TABLE and len(m.files) == 1


def test_ingest_rejects_oversized_upload(tmp_path, settings):
    big = write(tmp_path / "big.csv", b"a,b\n" * 400_000)
    with pytest.raises(IngestError) as info:
        ingest(big, create_workspace(settings.workspace_root), settings)
    assert info.value.code == "too_large"


def test_ingest_rejects_unsupported_only(tmp_path, settings):
    junk = write(tmp_path / "junk.bin", bytes(range(256)) * 4)
    with pytest.raises(IngestError) as info:
        ingest(junk, create_workspace(settings.workspace_root), settings)
    assert info.value.code == "no_supported_files"

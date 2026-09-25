"""Data contracts for ingestion: detected types, the file manifest, and samples."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TABLE = "table"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    DOC = "doc"  # README, LICENSE and similar notes that ship with datasets
    UNKNOWN = "unknown"


ANALYZABLE = (Modality.TABLE, Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO)


class IngestError(Exception):
    """A problem with the user's input, with a message safe to show in the UI."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Detection(BaseModel):
    modality: Modality
    format: str  # "csv", "xlsx", "png", "mp3", ...
    detail: str = ""  # for example "delimiter ';'" or "extension says .txt"


class FileEntry(BaseModel):
    path: str  # relative to the manifest root, forward slashes
    size: int
    modality: Modality
    format: str
    group: str = ""  # top-level folder, used for labels and stratified sampling
    detail: str = ""


class SkippedEntry(BaseModel):
    path: str
    reason: str


class FileManifest(BaseModel):
    root: str
    source_name: str
    files: list[FileEntry] = Field(default_factory=list)
    skipped: list[SkippedEntry] = Field(default_factory=list)
    counts: dict[Modality, int] = Field(default_factory=dict)
    total_bytes: int = 0
    dominant: Modality | None = None
    is_mixed: bool = False
    folders_as_labels: bool = False
    groups: dict[str, int] = Field(default_factory=dict)  # files per group, dominant type
    rejected: bool = False
    reject_reason: str = ""

    def files_of(self, modality: Modality) -> list[FileEntry]:
        return [f for f in self.files if f.modality == modality]


class SampleSet(BaseModel):
    modality: Modality
    files: list[FileEntry]
    total_available: int
    per_group: dict[str, int] = Field(default_factory=dict)
    seed: int

    @property
    def ratio(self) -> float:
        return 1.0 if self.total_available == 0 else len(self.files) / self.total_available

    @property
    def is_sample(self) -> bool:
        return len(self.files) < self.total_available

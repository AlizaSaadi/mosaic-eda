"""The allowed cleaning operations for text.

Text is cleaned through a table of documents (one row per document). Operations rewrite
the 'text' column or remove rows, and each one is a code template, so the exported
pipeline is exactly what ran.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field, field_validator

from mosaic.images.ops import _CHECK_FILES
from mosaic.tables.ops import NoParams, OpSpec, Risk
from mosaic.text import pipeline_helpers


class ShareParams(NoParams):
    min_share: float = Field(0.3, ge=0.1, le=1.0)


class SimilarityParams(NoParams):
    threshold: float = Field(0.8, ge=0.5, le=1.0)


class MinWordsParams(NoParams):
    min_words: int = Field(5, ge=1, le=500)


class MaxWordsParams(NoParams):
    max_words: int = Field(2000, ge=50, le=100000)


LANGUAGE_CODES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "dutch": "nl", "russian": "ru", "greek": "el", "arabic": "ar",
    "hindi": "hi", "japanese": "ja", "korean": "ko", "chinese": "zh",
}  # fmt: skip


class LanguageParams(NoParams):
    languages: list[str] = Field(min_length=1, max_length=10)
    keep_unknown: bool = True

    @field_validator("languages")
    @classmethod
    def _codes(cls, languages: list[str]) -> list[str]:
        """Accept names ('English') as well as codes ('en'); reject anything else."""
        codes = []
        for value in languages:
            code = LANGUAGE_CODES.get(value.strip().lower(), value.strip().lower())
            if code not in LANGUAGE_CODES.values():
                raise ValueError(
                    f"'{value}' isn't a language code; use codes from the evidence such as "
                    f"{', '.join(sorted(LANGUAGE_CODES.values()))}"
                )
            codes.append(code)
        return codes


class DocumentsParams(NoParams):
    documents: list[str] = Field(min_length=1, max_length=500)


TEXT_OPS: dict[str, OpSpec] = {}


def _op(name, risk, description, params, template):
    TEXT_OPS[name] = OpSpec(name, risk, description, params, template, needs_columns=False)


_drop = ".reset_index(drop=True)"
_op(
    "fix_encoding",
    Risk.SAFE,
    "Repair garbled characters from a wrong encoding ('cafÃ©' -> 'café').",
    NoParams,
    lambda c, p: "df['text'] = df['text'].map(fix_mojibake)",
)
_op(
    "normalize_unicode",
    Risk.SAFE,
    "Unicode NFKC normalization (full-width letters, ligatures, compatibility forms).",
    NoParams,
    lambda c, p: "df['text'] = df['text'].map(normalize_unicode)",
)
_op(
    "strip_html",
    Risk.SAFE,
    "Remove HTML tags and decode entities such as &amp;.",
    NoParams,
    lambda c, p: "df['text'] = df['text'].map(strip_html)",
)
_op(
    "normalize_whitespace",
    Risk.SAFE,
    "Collapse repeated spaces and blank lines.",
    NoParams,
    lambda c, p: "df['text'] = df['text'].map(normalize_whitespace)",
)
_op(
    "strip_boilerplate",
    Risk.LOSSY,
    "Remove lines repeated in at least min_share of documents (signatures, disclaimers).",
    ShareParams,
    lambda c, p: f"df = strip_boilerplate(df, min_share={p.min_share!r})",
)
_op(
    "mask_pii",
    Risk.LOSSY,
    "Replace emails, phone numbers, card numbers, SSNs, and IP addresses with tags.",
    NoParams,
    lambda c, p: "df['text'] = df['text'].map(mask_pii)",
)
_op(
    "truncate_long_documents",
    Risk.LOSSY,
    "Cut documents after max_words words.",
    MaxWordsParams,
    lambda c, p: f"df['text'] = df['text'].map(lambda t: truncate_words(t, {p.max_words}))",
)
_op(
    "drop_exact_duplicates",
    Risk.DESTRUCTIVE,
    "Keep one copy of documents with the same text (ignoring case and punctuation).",
    NoParams,
    lambda c, p: f"df = df[~df['text'].map(dedupe_key).duplicated()]{_drop}",
)
_op(
    "drop_near_duplicates",
    Risk.DESTRUCTIVE,
    "Keep one document per group of near-identical documents (estimated word 3-gram "
    "Jaccard similarity at least threshold).",
    SimilarityParams,
    lambda c, p: f"df = drop_near_duplicates(df, threshold={p.threshold!r})",
)
_op(
    "drop_cross_label_duplicates",
    Risk.DESTRUCTIVE,
    "Remove documents whose near-duplicates appear under more than one label.",
    SimilarityParams,
    lambda c, p: f"df = df[~cross_label_mask(df, threshold={p.threshold!r})]{_drop}",
)
_op(
    "drop_short_documents",
    Risk.DESTRUCTIVE,
    "Remove documents with fewer than min_words words (min_words 1 removes empty ones).",
    MinWordsParams,
    lambda c, p: f"df = df[df['text'].map(word_count) >= {p.min_words}]{_drop}",
)
_op(
    "filter_language",
    Risk.DESTRUCTIVE,
    "Keep only documents written in the listed languages (language codes such as 'en', "
    "'es'; keep_unknown keeps documents too short to tell).",
    LanguageParams,
    lambda c, p: (
        f"_lang = df['text'].map(detect_language)\n"
        f"df = df[_lang.isin({[*p.languages, *(['unknown'] if p.keep_unknown else [])]!r})]{_drop}"
    ),
)
_op(
    "drop_documents",
    Risk.DESTRUCTIVE,
    "Remove specific documents by ID (the path column), for example confirmed mislabels.",
    DocumentsParams,
    lambda c, p: (
        _CHECK_FILES.format(files=p.documents)
        + f"df = df[~df['path'].isin({p.documents!r})]{_drop}"
    ),
)
_op(
    "flag_suspected_mislabels",
    Risk.SAFE,
    "Mark documents for human review in the manifest (nothing is removed).",
    DocumentsParams,
    lambda c, p: (
        _CHECK_FILES.format(files=p.documents)
        + f"df['suspected_mislabel'] = df['suspected_mislabel'] | "
        f"df['path'].isin({p.documents!r})"
    ),
)


def text_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"pd": pd}
    ns.update({k: v for k, v in vars(pipeline_helpers).items() if not k.startswith("_")})
    return ns

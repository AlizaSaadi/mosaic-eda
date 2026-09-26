"""Profile a text dataset in code and save every result to the evidence store.

One row per document. Everything here is computed without a model: lengths, language
mix, encoding and markup problems, PII, repeated boilerplate, duplicates (MinHash),
top terms, NMF topics, readability, and a nearest-centroid label check. Quoted text is
always PII-masked and short.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mosaic.evidence.store import EvidenceStore
from mosaic.tables.profile import r4
from mosaic.text.pipeline_helpers import (
    STOPWORDS,
    dedupe_key,
    detect_language,
    has_html,
    has_mojibake,
    mask_pii,
    near_groups,
    pii_matches,
    repeated_lines,
    sentence_count,
    words,
)
from mosaic.ui.palette import HARVEST

TOO_SHORT_WORDS = 5
NEAR_THRESHOLD = 0.8
MISLABEL_MARGIN = 0.05
MAX_VOCAB = 2000
EXTRA_STOPWORDS = {
    "also",
    "get",
    "got",
    "would",
    "could",
    "should",
    "one",
    "two",
    "like",
    "any",
    "only",
    "while",
    "after",
    "before",
    "still",
    "since",
    "hi",
    "hello",
    "dear",
    "thanks",
    "thank",
    "please",
    "regards",
    "best",
    "cheers",
    "team",
}
CAPITALIZED = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", re.M)
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"(?<!\w)@\w{2,}")
HASHTAG = re.compile(r"(?<!\w)#[A-Za-z]\w+")
MONEY = re.compile(r"[$€£¥]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:USD|EUR|GBP|dollars|euros)\b", re.I)


@dataclass
class TextProfile:
    artifact_ids: list[str] = field(default_factory=list)
    chart_ids: list[str] = field(default_factory=list)
    quality: float = 0.0
    categories: dict[str, list[str]] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    main_language: str = "unknown"


def snippet(text: str, n: int = 90) -> str:
    """A short, PII-masked, single-line quote."""
    flat = re.sub(r"\s+", " ", mask_pii(text or "")).strip()
    return flat if len(flat) <= n else flat[: n - 3].rstrip() + "..."


def _stats(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {}
    return {"min": r4(values.min()), "median": r4(values.median()), "max": r4(values.max())}


def syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    count = len(groups) - (word.lower().endswith("e") and len(groups) > 1)
    return max(count, 1)


def flesch(text: str) -> float | None:
    tokens = words(text)
    if len(tokens) < 20:
        return None
    per_sentence = len(tokens) / sentence_count(text)
    per_word = sum(syllables(t) for t in tokens) / len(tokens)
    return round(206.835 - 1.015 * per_sentence - 84.6 * per_word, 1)


def doc_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-document measurements used by the profile (not part of the cleaning table)."""
    text = df["text"].fillna("")
    out = pd.DataFrame(index=df.index)
    out["words"] = text.map(lambda t: len(words(t)))
    out["chars"] = text.str.len()
    out["sentences"] = text.map(sentence_count)
    out["language"] = text.map(detect_language)
    out["mojibake"] = text.map(has_mojibake)
    out["html"] = text.map(has_html)
    out["pii"] = text.map(pii_matches)
    return out


def categorize(df: pd.DataFrame, m: pd.DataFrame, main_language: str) -> dict[str, list[str]]:
    """One main problem per document, in a fixed order."""
    long_limit = max(1000, 10 * float(m["words"].median() or 0))
    rules = [
        ("empty", m["words"] == 0),
        ("encoding_damage", m["mojibake"]),
        ("html_markup", m["html"]),
        ("too_short", m["words"] < TOO_SHORT_WORDS),
        (
            "other_language",
            (m["language"] != main_language) & (m["language"] != "unknown")
            if main_language != "unknown"
            else pd.Series(False, index=m.index),
        ),
        ("very_long", m["words"] > long_limit),
    ]
    cats: dict[str, list[str]] = {}
    taken: set[str] = set()
    for name, mask in rules:
        files = [p for p in df.loc[mask, "path"] if p not in taken]
        cats[name] = files
        taken.update(files)
    return cats


def _fig(fig: go.Figure, title: str) -> dict:
    import json

    fig.update_layout(
        title=title,
        template="plotly_white",
        colorway=HARVEST["chart"],
        height=340,
        font={"family": "Inter, system-ui, sans-serif", "color": HARVEST["text"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    return json.loads(fig.to_json())


def _chart(store: EvidenceStore, name: str, title: str, fig: go.Figure, stage: str) -> str:
    return store.add(
        "chart", "chart", name, f"Chart: {title} ({stage})", {"figure": _fig(fig, title)}
    ).id


# ---- vector space: terms, topics, label check ----


def tokens_for_terms(text: str, stop: set[str]) -> list[str]:
    return [
        t
        for t in (w.lower() for w in words(text))
        if len(t) >= 3 and t not in stop and not t.isdigit()
    ]


def tfidf(token_lists: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    n = len(token_lists)
    doc_freq = Counter(t for tokens in token_lists for t in set(tokens))
    upper = 0.5 * n if n >= 10 else n
    vocab = [t for t, d in doc_freq.most_common() if 2 <= d <= upper][:MAX_VOCAB]
    if not vocab:
        return np.zeros((n, 0), np.float32), []
    index = {t: i for i, t in enumerate(vocab)}
    x = np.zeros((n, len(vocab)), np.float32)
    for row, tokens in enumerate(token_lists):
        for t, c in Counter(tokens).items():
            if t in index:
                x[row, index[t]] = 1 + np.log(c)
    idf = np.log((1 + n) / (1 + np.array([doc_freq[t] for t in vocab]))) + 1
    x *= idf.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms), vocab


def nmf(x: np.ndarray, k: int, iters: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative matrix factorization (multiplicative updates): x ~ w @ h."""
    rng = np.random.default_rng(seed)
    w = rng.random((x.shape[0], k)).astype(np.float32) + 0.1
    h = rng.random((k, x.shape[1])).astype(np.float32) + 0.1
    eps = 1e-9
    for _ in range(iters):
        h *= (w.T @ x) / (w.T @ w @ h + eps)
        w *= (x @ h.T) / (w @ h @ h.T + eps)
    return w, h


def label_check(
    x: np.ndarray, labels: list[str], words_per_doc: list[int], usable: list[bool]
) -> tuple[list[dict], float | None]:
    """Leave-one-out nearest class centroid. Returns (suspects, agreement %)."""
    classes = [c for c, n in Counter(labels).items() if c and n >= 3]
    if len(classes) < 2 or x.shape[1] == 0:
        return [], None
    sums = {c: x[[i for i, lab in enumerate(labels) if lab == c]].sum(axis=0) for c in classes}
    sizes = Counter(labels)
    suspects, agree, checked = [], 0, 0
    for i, label in enumerate(labels):
        if not usable[i] or label not in sums or words_per_doc[i] < TOO_SHORT_WORDS:
            continue
        sims = {}
        for c in classes:
            centroid = (sums[c] - x[i]) / (sizes[c] - 1) if c == label else sums[c] / sizes[c]
            norm = np.linalg.norm(centroid)
            sims[c] = float(x[i] @ centroid / norm) if norm else 0.0
        best = max(sims, key=sims.get)
        checked += 1
        agree += best == label
        if best != label and sims[best] - sims[label] >= MISLABEL_MARGIN:
            suspects.append(
                {
                    "index": i,
                    "label": label,
                    "closest": best,
                    "margin": r4(sims[best] - sims[label]),
                }
            )
    return suspects, (r4(100 * agree / checked) if checked else None)


# ---- the profile ----


def profile_text(
    df: pd.DataFrame,
    store: EvidenceStore,
    *,
    structure: str,
    total_docs: int,
    stage: str = "raw",
) -> TextProfile:
    result = TextProfile()
    m = doc_metrics(df)
    known = Counter(lang for lang in m["language"] if lang != "unknown")
    main = known.most_common(1)[0][0] if known else "unknown"
    result.main_language = main
    cats = categorize(df, m, main)
    result.categories = cats
    n = max(len(df), 1)
    labels = [str(c) for c in df["class"].fillna("")]
    labelled = {k: v for k, v in Counter(labels).items() if k}
    label_word = "speaker" if structure == "transcript" else "label"

    # overview
    token_lists = [[w.lower() for w in words(t)] for t in df["text"].fillna("")]
    vocabulary = len({t for tokens in token_lists for t in tokens})
    total_words = int(m["words"].sum())
    english = [
        flesch(t) for t, lang in zip(df["text"], m["language"], strict=False) if lang == "en"
    ]
    english = [v for v in english if v is not None]
    overview = {
        "documents": total_docs,
        "analyzed": len(df),
        "structure": structure,
        "source_files": int(df["source_file"].nunique()),
        "labels": len(labelled),
        "words_per_document": _stats(m["words"]),
        "total_words": total_words,
        "vocabulary": vocabulary,
        "type_token_ratio": r4(vocabulary / max(total_words, 1)),
        "sentences_per_document": _stats(m["sentences"]),
        "reading_ease_median": r4(float(np.median(english))) if english else None,
    }
    sample = "all analyzed" if len(df) >= total_docs else f"{len(df)} sampled"
    shapes = {
        "corpus": "one document per file",
        "paragraphs": "one file split into paragraphs",
        "lines": "one file, one document per line",
        "transcript": "a transcript, one document per speaker turn",
    }
    reading = (
        f"; median Flesch reading ease {overview['reading_ease_median']} for English documents "
        "(reading_ease_median; 60-70 is plain English, below 30 is very hard)"
        if english
        else ""
    )
    a = store.add(
        "txt_overview",
        "profile",
        "text_inventory",
        f"Text ({stage}): {total_docs} documents ({sample}; 'documents' is the count), "
        f"{shapes.get(structure, structure)}; {overview['source_files']} source files; "
        f"{len(labelled)} {label_word}s (labels). Words per document "
        f"{overview['words_per_document']} (words_per_document.median etc.); total_words "
        f"{total_words}; vocabulary {vocabulary} distinct words; type_token_ratio "
        f"{overview['type_token_ratio']}{reading}.",
        overview,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    # quality problems
    counts = {k: len(v) for k, v in cats.items()}
    quality = {
        "counts": counts,
        "files": {k: v[:20] for k, v in cats.items()},
        "thresholds": {"too_short_words": TOO_SHORT_WORDS},
        "main_language": main,
    }
    listed = "; ".join(f"{k}: {', '.join(v[:5])}" for k, v in cats.items() if v)
    a = store.add(
        "txt_quality",
        "profile",
        "text_quality",
        f"Text problems ({stage}), one main problem per document: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
        + f" (keys counts.<problem>; too_short means under {TOO_SHORT_WORDS} words; "
        f"encoding_damage means garbled characters such as 'Ã©'). Documents: {listed or 'none'}.",
        quality,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    # languages
    langs = Counter(m["language"])
    language = {
        "languages": {k: {"count": v, "share": r4(100 * v / n)} for k, v in langs.most_common()},
        "main_language": main,
        "other_language_documents": counts["other_language"],
    }
    text = ", ".join(f"{k} {v['count']} ({v['share']}%)" for k, v in language["languages"].items())
    a = store.add(
        "txt_language",
        "profile",
        "language_mix",
        f"Language mix ({stage}, detected by code from common words and writing system): {text} "
        f"(keys languages.<code>.count/.share); main language '{main}'; "
        f"{counts['other_language']} documents in another language (other_language_documents). "
        "'unknown' means too short to tell.",
        language,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    # PII (counts and masked examples only)
    kinds: Counter = Counter()
    docs_with = Counter()
    examples = []
    for path, text_, found in zip(df["path"], df["text"], m["pii"], strict=False):
        for kind, hits in found.items():
            kinds[kind] += len(hits)
            docs_with[kind] += 1
        if found and len(examples) < 6:
            hit = next(iter(found.values()))[0]
            at = text_.find(hit)
            examples.append(
                {"document": path, "context": snippet(text_[max(0, at - 40) : at + 60])}
            )
    pii_docs = int(sum(bool(f) for f in m["pii"]))
    pii = {
        "documents_with_pii": pii_docs,
        "share": r4(100 * pii_docs / n),
        "occurrences": dict(kinds),
        "documents_by_kind": dict(docs_with),
        "examples": examples,
    }
    a = store.add(
        "txt_pii",
        "profile",
        "pii_scan",
        f"PII scan ({stage}, regex; card numbers pass a Luhn check): {pii_docs} documents "
        f"({pii['share']}%) contain personal data (documents_with_pii, share); documents by "
        f"kind {dict(docs_with)} (documents_by_kind.<kind>); occurrences {dict(kinds)} "
        f"(occurrences.<kind>). Examples are masked: "
        + "; ".join(f"{e['document']}: {e['context']}" for e in examples[:3]),
        pii,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    # duplicates
    keys = df["text"].fillna("").map(dedupe_key)
    nonempty = keys != ""
    exact = df[nonempty & keys.duplicated(keep=False)].groupby(keys)["path"].apply(list)
    groups = near_groups(df, NEAR_THRESHOLD)
    near = [g for g in df[nonempty].groupby(groups[nonempty])["path"].apply(list) if len(g) > 1]
    cross = []
    if labelled:
        label_sets = df[nonempty].groupby(groups[nonempty])["class"].agg(lambda s: set(s) - {""})
        cross = [
            df.loc[groups == g, "path"].tolist() for g, labs in label_sets.items() if len(labs) > 1
        ]
    dupes = {
        "exact_groups": len(exact),
        "exact_extra_copies": int(sum(len(g) - 1 for g in exact)),
        "near_groups": len(near),
        "near_extra_copies": int(sum(len(g) - 1 for g in near)),
        "cross_label_groups": len(cross),
        "cross_label_documents": int(sum(len(g) for g in cross)),
        "threshold": NEAR_THRESHOLD,
        "examples": near[:10],
        "cross_label_examples": cross[:10],
    }
    a = store.add(
        "txt_dupes",
        "profile",
        "duplicates",
        f"Duplicates ({stage}): {dupes['exact_groups']} groups of identical text "
        f"({dupes['exact_extra_copies']} extra copies, exact_extra_copies); {dupes['near_groups']}"
        f" near-duplicate groups at MinHash similarity {NEAR_THRESHOLD}+, including exact ones "
        f"({dupes['near_extra_copies']} extra copies, near_extra_copies): {near[:5]}"
        + (
            f"; {dupes['cross_label_groups']} groups appear under more than one {label_word} "
            f"(cross_label_groups, {dupes['cross_label_documents']} documents): {cross[:4]}"
            if labelled
            else ""
        ),
        dupes,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    # label balance
    if labelled:
        total = sum(labelled.values())
        balance = {
            "classes": {k: {"count": v, "share": r4(100 * v / total)} for k, v in labelled.items()},
            "imbalance_ratio": r4(max(labelled.values()) / max(min(labelled.values()), 1)),
        }
        text = ", ".join(f"{k} {v['count']} ({v['share']}%)" for k, v in balance["classes"].items())
        a = store.add(
            "txt_balance",
            "profile",
            "label_balance",
            f"{label_word.capitalize()} balance ({total} documents; {label_word}s come from "
            f"{'speaker names' if structure == 'transcript' else 'folder names'}): {text}; "
            f"largest/smallest ratio {balance['imbalance_ratio']} (imbalance_ratio). "
            "Keys: classes.<name>.count/.share.",
            balance,
            {"stage": stage},
        )
        result.artifact_ids.append(a.id)
        fig = go.Figure(
            go.Bar(x=list(labelled), y=list(labelled.values()), marker_color=HARVEST["accent"])
        )
        result.chart_ids.append(
            _chart(store, "label_balance", f"Documents per {label_word}", fig, stage)
        )

    # length histogram
    values = m["words"]
    if len(values):
        hist, edges = np.histogram(values, bins=min(30, max(5, len(values) // 4)))
        fig = go.Figure(
            go.Bar(
                x=(edges[:-1] + edges[1:]) / 2,
                y=hist,
                width=np.diff(edges),
                marker_color=HARVEST["chart"][1],
            )
        )
        result.chart_ids.append(_chart(store, "words_hist", "Words per document", fig, stage))

    if stage == "raw":
        result.artifact_ids += _raw_only(df, m, store, result, main, labels, token_lists, stage)

    # quality score
    low = sum(counts[k] for k in ("empty", "encoding_damage", "html_markup", "too_short"))
    imbalance = labelled and max(labelled.values()) / max(min(labelled.values()), 1)
    parts = {
        "low_quality": -min(25.0, 100 * low / n * 1.5),
        "duplicates": -min(15.0, 100 * dupes["near_extra_copies"] / n * 1.5),
        "privacy": -min(10.0, 100 * pii_docs / n * 0.5),
        "other_language": -min(10.0, 100 * counts["other_language"] / n),
        "label_noise": -min(10.0, 100 * len(result.mismatches) / n * 1.5),
        "label_imbalance": -min(10.0, max(0.0, (imbalance or 1) - 1.5) * 3),
    }
    parts = {k: round(v, 1) for k, v in parts.items()}
    score = round(max(0.0, 100 + sum(parts.values())), 1)
    result.quality = score
    a = store.add(
        "txt_quality_score",
        "profile",
        "quality_score",
        f"Text data quality score ({stage}): {score}/100 (score); deductions {parts}.",
        {"score": score, "parts": parts},
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)
    return result


def _raw_only(
    df: pd.DataFrame,
    m: pd.DataFrame,
    store: EvidenceStore,
    result: TextProfile,
    main: str,
    labels: list[str],
    token_lists: list[list[str]],
    stage: str,
) -> list[str]:
    """Terms, topics, label check, boilerplate, and entities (raw data only)."""
    ids: list[str] = []
    n = len(df)
    texts = df["text"].fillna("").tolist()

    # repeated boilerplate lines
    if n >= 5:
        lines = repeated_lines(texts, 0.3, 3)
        ranked = sorted(lines.items(), key=lambda kv: -kv[1])[:8]
        boiler = {  # named keys, so "line_1" in the text is lines.line_1 in the data
            "lines": {
                f"line_{i}": {"line": snippet(line, 80), "documents": c, "share": r4(100 * c / n)}
                for i, (line, c) in enumerate(ranked, 1)
            },
            "count": len(lines),
        }
        text = "; ".join(
            f"{key}: '{b['line']}' in {b['documents']} documents ({b['share']}%)"
            for key, b in list(boiler["lines"].items())[:5]
        )
        ids.append(
            store.add(
                "txt_boilerplate",
                "profile",
                "boilerplate",
                f"Repeated lines found in at least 30% of documents (signatures, disclaimers, "
                f"templates): {len(lines)} (count){': ' + text if text else ''}. Keys "
                "lines.line_<n>.share and lines.line_<n>.documents.",
                boiler,
                {"stage": stage},
            ).id
        )

    # terms
    stop = STOPWORDS["en"] | STOPWORDS.get(main, set()) | EXTRA_STOPWORDS
    term_lists = [tokens_for_terms(t, stop) for t in texts]
    # document frequency (documents containing the word), so one long document can't dominate
    unigrams = Counter(t for tokens in term_lists for t in set(tokens))
    bigrams: Counter = Counter()
    for tokens in token_lists:
        bigrams.update(
            {
                f"{a} {b}"
                for a, b in pairwise(tokens)
                if len(a) > 2 and len(b) > 2 and a not in stop and b not in stop
            }
        )
    labelled = sorted({lab for lab in labels if lab})
    by_label = {}
    if len(labelled) >= 2:
        for lab in labelled:
            members = [set(t) for t, x in zip(term_lists, labels, strict=False) if x == lab]
            inside = Counter(t for tokens in members for t in tokens)
            scored = {
                t: (c / len(members)) / ((unigrams[t] + 1) / n)
                for t, c in inside.items()
                if c >= max(2, 0.1 * len(members))
            }
            by_label[lab] = [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:6]]
    terms = {
        "top_words": dict(unigrams.most_common(15)),
        "top_bigrams": dict(bigrams.most_common(10)),
        "distinctive_by_label": by_label,
    }
    ids.append(
        store.add(
            "txt_terms",
            "profile",
            "ngram_frequencies",
            f"Most common words by number of documents containing them (stop words removed): "
            f"{terms['top_words']} (top_words.<word>); "
            f"word pairs: {terms['top_bigrams']} (top_bigrams.<pair>)"
            + (f"; most distinctive words per label: {by_label}" if by_label else "")
            + ".",
            terms,
            {"stage": stage},
        ).id
    )
    top = unigrams.most_common(12)[::-1]
    if top:
        fig = go.Figure(
            go.Bar(
                x=[c for _, c in top],
                y=[t for t, _ in top],
                orientation="h",
                marker_color=HARVEST["chart"][2],
            )
        )
        result.chart_ids.append(
            _chart(store, "top_words", "Documents containing each word", fig, stage)
        )

    # language chart
    langs = Counter(m["language"])
    if len(langs) > 1:
        fig = go.Figure(
            go.Bar(x=list(langs), y=list(langs.values()), marker_color=HARVEST["chart"][3])
        )
        result.chart_ids.append(_chart(store, "languages", "Documents per language", fig, stage))

    # topics and label check share one TF-IDF matrix
    x, vocab = tfidf(term_lists)
    if n >= 20 and len(vocab) >= 20:
        k = min(6, max(2, len(labelled) or n // 25))
        w, h = nmf(x, k)
        assigned = np.where(w.max(axis=1) > 0, w.argmax(axis=1), -1)
        found = sorted(
            (
                {
                    "words": [vocab[j] for j in np.argsort(-h[t])[:6]],
                    "share": r4(100 * float((assigned == t).sum()) / n),
                }
                for t in range(k)
            ),
            key=lambda t: -t["share"],
        )
        # named keys, numbered by size, so "topic 1" in the text is topics.topic_1 in the data
        topics = {f"topic_{i}": t for i, t in enumerate(found, 1)}
        text = "; ".join(
            f"{key} ({t['share']}%): {', '.join(t['words'])}" for key, t in topics.items()
        )
        ids.append(
            store.add(
                "txt_topics",
                "profile",
                "topic_model",
                f"Topics found by NMF on TF-IDF ({k} topics, computed by code, largest first; "
                f"words are the topic's strongest terms, share = % of documents mainly about "
                f"it): {text}. Keys topics.topic_<n>.share.",
                {"k": k, "topics": topics},
                {"stage": stage},
            ).id
        )

    # documents in another language would look "mislabeled" to a vocabulary check
    usable = [
        bool(v.any()) and lang in (main, "unknown")
        for v, lang in zip(x, m["language"], strict=False)
    ]
    suspects, agreement = label_check(x, labels, m["words"].tolist(), usable)
    if agreement is not None:
        rows = [
            {
                "document": df["path"].iloc[s["index"]],
                "label": s["label"],
                "closest": s["closest"],
                "margin": s["margin"],
                "text": snippet(texts[s["index"]], 100),
            }
            for s in sorted(suspects, key=lambda s: -s["margin"])
        ]
        result.mismatches = [r["document"] for r in rows]
        listed = "; ".join(
            f"{r['document']} (label '{r['label']}', reads like '{r['closest']}')" for r in rows[:8]
        )
        ids.append(
            store.add(
                "txt_labels",
                "profile",
                "label_check",
                f"Label check by code (each document compared with every label's average TF-IDF "
                f"vector, leaving itself out): {agreement}% of documents are closest to their own "
                f"label (centroid_agreement_pct; higher means labels are easy to tell apart); "
                f"{len(rows)} documents read more like another label (label_mismatches): "
                f"{listed or 'none'}. These are likely mislabels, not certainties.",
                {
                    "centroid_agreement_pct": agreement,
                    "label_mismatches": len(rows),
                    "mismatches": rows[:20],
                    "margin": MISLABEL_MARGIN,
                },
                {"stage": stage},
            ).id
        )

    # simple entities
    def docs_matching(pattern: re.Pattern) -> int:
        return sum(bool(pattern.search(t)) for t in texts)

    # names are counted, never listed: they're often people
    entities = {
        "documents_with_urls": docs_matching(URL),
        "documents_with_mentions": docs_matching(MENTION),
        "documents_with_hashtags": docs_matching(HASHTAG),
        "documents_with_money": docs_matching(MONEY),
        "documents_with_names": docs_matching(CAPITALIZED),
    }
    ids.append(
        store.add(
            "txt_entities",
            "profile",
            "entity_scan",
            "Entities found by pattern matching (documents containing each): URLs "
            f"{entities['documents_with_urls']} (documents_with_urls), @mentions "
            f"{entities['documents_with_mentions']}, #hashtags "
            f"{entities['documents_with_hashtags']}, money amounts "
            f"{entities['documents_with_money']} (documents_with_money), capitalized names of "
            f"people, products, or places {entities['documents_with_names']} "
            "(documents_with_names; names aren't listed because they may be people).",
            entities,
            {"stage": stage},
        ).id
    )
    return ids

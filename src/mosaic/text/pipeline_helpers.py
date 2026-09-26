"""Helper functions for text cleaning.

This module's source is copied verbatim into every exported cleaning_pipeline.py for
text, so it only imports the standard library, numpy, and pandas.
One row per document: cleaning operations rewrite the 'text' column or filter rows, and
export_text() writes the cleaned documents as JSON Lines plus a manifest CSV.
"""

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_EXTENSIONS = {".txt", ".text"}
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
SPEAKER_LINE = re.compile(r"^\s*(?:\[[\d:.]+\]\s*)?([A-Z][\w .'-]{0,30}?):\s+\S")
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
HTML_TAG = re.compile(
    r"</?(?:p|br|div|span|a|b|i|strong|em|li|ul|ol|table|tr|td|h\d|html|body)\b[^>]*>", re.I
)
HTML_ENTITY = re.compile(r"&(?:[a-z]{2,8}|#\d{2,5});", re.I)
MOJIBAKE = re.compile(
    r"[\u00c2\u00c3\u00e2][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d"
    r"\u017e\u0192\u02c6\u02dc\u2013-\u2122]"
)
PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(
        r"(?<![\w+(])(?<!\d[ .-])(?:\+?\d{1,3}[ .-])?\(?\d{2,4}\)?[ .-]\d{3,4}"
        r"[ .-]\d{3,4}\b(?![ .-]?\d)"
    ),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _set(text):
    return set(text.split())


STOPWORDS = {
    "en": _set(
        "the and to of a in is it you that for on was with are be this have not but at as i"
        " my we your they he she or from by an so if me our can will just do all there what"
        " been has had would when about their them out up one no which more were"
    ),
    "es": _set(
        "el la de que y en los se del las un por con no una su para es al lo como más pero"
        " sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay"
        " donde quien desde todo nos durante estoy mi pedido"
    ),
    "fr": _set(
        "le la les de des et en un une du est que qui dans pour pas sur au avec il elle ce"
        " ne se plus par mais ou nous vous je mon ma mes son sa ses leur été être avoir"
    ),
    "de": _set(
        "der die das und ist nicht ein eine zu den von mit sich des auf für im dem es ich"
        " sie wir ihr mein meine auch als an bei aus nach wie noch aber oder wird"
    ),
    "it": _set(
        "il lo la i gli le di da in con su per tra fra un una che non è sono mi ho ha del"
        " della dei delle nel nella al alla come ma anche questo quella mio mia"
    ),
    "pt": _set(
        "o a os as de do da dos das em no na um uma que não é por para com se mais mas foi"
        " ao meu minha você eu está como seu sua pelo pela isso"
    ),
    "nl": _set(
        "de het een en van in is dat op te zijn met voor niet aan er ik je hij zij wij mijn"
        " maar ook als bij door naar dan nog wel om"
    ),
}
SCRIPTS = (
    ("ru", "Ѐ", "ӿ"),
    ("el", "Ͱ", "Ͽ"),
    ("ar", "؀", "ۿ"),
    ("hi", "ऀ", "ॿ"),
    ("ja", "぀", "ヿ"),
    ("ko", "가", "힯"),
    ("zh", "一", "鿿"),
)
SHINGLE = 3
NUM_HASHES = 64
BANDS = 16


def read_text(path):
    raw = Path(path).read_bytes()
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def words(text):
    return WORD.findall(text or "")


def word_count(text):
    return len(words(text))


def sentence_count(text):
    text = (text or "").strip()
    return max(len(SENTENCE_END.findall(text)), 1) if text else 0


def detect_structure(text):
    """'transcript' (Speaker: line), 'paragraphs' (blank-line separated), or 'lines'."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "paragraphs"
    speakers = [m.group(1) for m in map(SPEAKER_LINE.match, lines) if m]
    if len(speakers) >= 0.6 * len(lines) and 2 <= len(set(speakers)) <= 20:
        return "transcript"
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    return "paragraphs" if len(blocks) >= 3 or len(lines) < 3 else "lines"


def split_documents(text, structure=None):
    """(document text, speaker) pairs for one file."""
    structure = structure or detect_structure(text)
    if structure == "transcript":
        docs, speaker, buf = [], "", []
        for line in text.splitlines():
            m = SPEAKER_LINE.match(line)
            if m:
                if buf:
                    docs.append((" ".join(buf).strip(), speaker))
                speaker = m.group(1).strip()
                buf = [line[line.index(":", m.start(1)) + 1 :].strip()]
            elif line.strip():
                buf.append(line.strip())
        if buf:
            docs.append((" ".join(buf).strip(), speaker))
        return docs
    if structure == "lines":
        return [(ln.strip(), "") for ln in text.splitlines() if ln.strip()]
    return [(b.strip(), "") for b in re.split(r"\n\s*\n", text) if b.strip()]


def list_text(folder):
    """(relative path, class) for every text file; the class is the first folder level."""
    folder = Path(folder)
    files = sorted(
        p
        for p in folder.rglob("*")
        if p.suffix.lower() in TEXT_EXTENSIONS and not p.name.lower().startswith("readme")
    )
    rels = [p.relative_to(folder).parts for p in files]
    tops = {r[0] for r in rels if len(r) > 1}
    skip = 1 if len(tops) == 1 and all(len(r) > 2 for r in rels) else 0  # one wrapper folder
    return [("/".join(r), r[skip] if len(r) > skip + 1 else "") for r in rels]


def detect_language(text):
    """A small, dependency-free guess: writing system first, then common-word counts."""
    letters = [ch for ch in text or "" if ch.isalpha()]
    if len(letters) < 12:
        return "unknown"
    for code, lo, hi in SCRIPTS:
        if sum(lo <= ch <= hi for ch in letters) / len(letters) > 0.3:
            return code
    tokens = [w.lower() for w in words(text)]
    if len(tokens) < 3:
        return "unknown"
    scores = {code: sum(t in stop for t in tokens) for code, stop in STOPWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] < 2 and scores[best] / len(tokens) < 0.15:
        return "unknown"
    return best


def has_mojibake(text):
    return bool(MOJIBAKE.search(text or ""))


def has_html(text):
    return bool(HTML_TAG.search(text or "")) or len(HTML_ENTITY.findall(text or "")) >= 2


def _luhn(digits):
    total, flip = 0, False
    for d in reversed(digits):
        n = int(d) * (2 if flip else 1)
        total += n - 9 if n > 9 else n
        flip = not flip
    return total % 10 == 0


def pii_matches(text):
    """{kind: [matched strings]}; card numbers must pass the Luhn check."""
    found = {}
    rest = text or ""
    for kind, pattern in PII_PATTERNS.items():
        hits = pattern.findall(rest)
        if kind == "card":
            hits = [h for h in hits if _luhn(re.sub(r"\D", "", h))]
        if kind == "ip":
            hits = [h for h in hits if all(int(p) <= 255 for p in h.split("."))]
        if hits:
            found[kind] = hits
            for h in hits:  # a card number shouldn't also count as a phone number
                rest = rest.replace(h, " ")
    return found


def mask_pii(text):
    for kind, hits in pii_matches(text).items():
        for h in sorted(set(hits), key=len, reverse=True):
            text = text.replace(h, f"[{kind.upper()}]")
    return text


def fix_mojibake(text):
    """Repair UTF-8 text that was decoded as Windows-1252 ('cafÃ©' -> 'café')."""
    if not has_mojibake(text):
        return text
    try:
        fixed = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return fixed if not has_mojibake(fixed) else text


def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text or "")


def normalize_whitespace(text):
    text = re.sub(r"[ \t ]+", " ", text or "")
    return re.sub(r"\s*\n\s*(\n\s*)+", "\n\n", text).strip()


def strip_html(text):
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text or "", flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(html.unescape(text))


def truncate_words(text, max_words):
    spans = list(WORD.finditer(text or ""))
    return text if len(spans) <= max_words else text[: spans[max_words - 1].end()]


def repeated_lines(texts, min_share=0.3, min_docs=3):
    """Lines (such as signatures and disclaimers) found in at least min_share of documents."""
    counts = {}
    for text in texts:
        for line in {ln.strip() for ln in (text or "").splitlines() if len(ln.strip()) >= 8}:
            counts[line] = counts.get(line, 0) + 1
    need = max(min_docs, int(np.ceil(min_share * len(texts))))
    return {line: n for line, n in counts.items() if n >= need}


def strip_boilerplate(df, min_share=0.3):
    lines = set(repeated_lines(df["text"].tolist(), min_share))
    if lines:
        df["text"] = df["text"].map(
            lambda t: "\n".join(ln for ln in t.splitlines() if ln.strip() not in lines).strip()
        )
    return df


def dedupe_key(text):
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def minhash(text):
    """MinHash signature of word 3-grams (hex), for estimating Jaccard similarity."""
    tokens = [w.lower() for w in words(text)]
    shingles = {" ".join(tokens[i : i + SHINGLE]) for i in range(max(len(tokens) - SHINGLE + 1, 1))}
    shingles.discard("")
    if not shingles:
        return ""
    hashes = np.array(
        [
            int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
            for s in shingles
        ],
        dtype=np.uint64,
    )
    rng = np.random.default_rng(7)
    a = rng.integers(1, 2**61, NUM_HASHES, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**61, NUM_HASHES, dtype=np.uint64)
    signature = ((hashes[:, None] * a[None, :] + b[None, :]) >> np.uint64(32)).min(axis=0)
    return signature.astype(np.uint32).tobytes().hex()


def similarity(sig_a, sig_b):
    a = np.frombuffer(bytes.fromhex(sig_a), np.uint32)
    b = np.frombuffer(bytes.fromhex(sig_b), np.uint32)
    return float((a == b).mean())


def near_groups(df, threshold=0.8):
    """Group documents whose estimated word-3-gram Jaccard similarity is at least threshold.

    Band hashing (LSH) finds candidate pairs, so this stays fast on large corpora.
    """
    sigs = [minhash(t) for t in df["text"]]
    parent = list(range(len(sigs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows = NUM_HASHES // BANDS
    buckets = {}
    for i, sig in enumerate(sigs):
        if not sig:
            continue
        for band in range(BANDS):
            buckets.setdefault((band, sig[band * rows * 8 : (band + 1) * rows * 8]), []).append(i)
    checked = set()
    for members in buckets.values():
        for x, i in enumerate(members):
            for j in members[x + 1 :]:
                if (i, j) in checked or find(i) == find(j):
                    continue
                checked.add((i, j))
                if similarity(sigs[i], sigs[j]) >= threshold:
                    parent[find(i)] = find(j)
    return pd.Series([find(i) for i in range(len(sigs))], index=df.index)


def drop_near_duplicates(df, threshold=0.8):
    return df[~near_groups(df, threshold).duplicated()].reset_index(drop=True)


def cross_label_mask(df, threshold=0.8):
    """Documents whose near-duplicates appear under more than one label."""
    groups = near_groups(df, threshold)
    labels = df.groupby(groups)["class"].transform("nunique")
    return (labels > 1) & (df["class"] != "")


def text_record(text, path, label, source_file):
    return {
        "path": path,
        "class": label,
        "source_file": source_file,
        "text": text,
        "suspected_mislabel": False,
    }


def build_table(source, files=None):
    """One row per document. A folder gives one document per file (class = folder); a
    single file is split into documents (paragraphs, lines, or transcript turns)."""
    source = Path(source)
    rows = []
    if source.is_file():
        for i, (doc, speaker) in enumerate(split_documents(read_text(source)), 1):
            rows.append(text_record(doc, f"{source.name}#{i}", speaker, source.name))
    else:
        for rel, label in files or list_text(source):
            rows.append(text_record(read_text(source / rel), rel, label, rel))
    columns = ["path", "class", "source_file", "text", "suspected_mislabel"]
    return pd.DataFrame(rows, columns=columns)


def export_text(df, source, out_folder):
    """Write cleaned_documents.jsonl, a manifest CSV, and (for folders) the cleaned files."""
    source, out_folder = Path(source), Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    with open(out_folder / "cleaned_documents.jsonl", "w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            record = {"id": row.path, "label": row._1, "text": row.text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if source.is_dir():
        for row in df.itertuples(index=False):
            target = out_folder / "documents" / row.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(row.text, encoding="utf-8")
    manifest = df.drop(columns=["text"]).assign(words=df["text"].map(word_count))
    manifest.to_csv(out_folder / "text_manifest.csv", index=False)
    return out_folder

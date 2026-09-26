import pandas as pd

from mosaic.ingest.models import Modality
from mosaic.ingest.sniffer import sniff
from mosaic.tables.pipeline_helpers import parse_log_lines
from mosaic.text.pipeline_helpers import (
    build_table,
    detect_language,
    detect_structure,
    drop_near_duplicates,
    fix_mojibake,
    has_mojibake,
    mask_pii,
    pii_matches,
    split_documents,
    strip_boilerplate,
    strip_html,
)
from mosaic.text.profile import label_check, nmf, snippet, tfidf


def test_language_guess_uses_common_words_and_scripts():
    assert detect_language("The package did not arrive and I want a refund for it") == "en"
    assert detect_language("Mi pedido no ha llegado y quiero saber donde esta") == "es"
    assert detect_language("Je ne peux plus me connecter depuis la mise à jour") == "fr"
    assert detect_language("Моя посылка не пришла, и я хочу вернуть деньги") == "ru"
    assert detect_language("help??") == "unknown"


def test_pii_is_found_and_masked_but_card_numbers_need_luhn():
    text = (
        "Mail a.b@example.com, call (415) 555-0132, card 4111 1111 1111 1111, "
        "SSN 123-45-6789, order 2026-09-25, not a card 1234 5678 9012 3456"
    )
    found = pii_matches(text)
    assert found["card"] == ["4111 1111 1111 1111"]
    assert set(found) == {"email", "phone", "card", "ssn"}
    masked = mask_pii(text)
    assert "[EMAIL]" in masked and "[PHONE]" in masked and "[CARD]" in masked
    assert "4111" not in masked and "(" not in masked.split("call")[1][:3]
    assert "2026-09-25" in masked  # dates aren't phone numbers


def test_garbled_encoding_is_detected_and_repaired():
    for good in ("The café in Zürich", "Price 49,00 € today", "the app’s sync button – again"):
        bad = good.encode("utf-8").decode("cp1252")
        assert has_mojibake(bad) and not has_mojibake(good)
        assert fix_mojibake(bad) == good
    assert fix_mojibake("plain text") == "plain text"


def test_html_and_boilerplate_are_stripped():
    assert strip_html("<p>Hello&nbsp;<b>world</b></p>") == "Hello world"
    notice = "CONFIDENTIAL: this message is for the named recipient only."
    df = pd.DataFrame({"text": [f"ticket {i}\n{notice}" for i in range(5)] + ["other"]})
    assert notice not in " ".join(strip_boilerplate(df, 0.3)["text"])


def test_structure_detection_and_splitting(tmp_path):
    transcript = "Agent: Hello there\nCustomer: My order is late\nand still not here\nAgent: Sorry"
    assert detect_structure(transcript) == "transcript"
    assert split_documents(transcript)[1] == ("My order is late and still not here", "Customer")
    assert detect_structure("First paragraph.\n\nSecond one.\n\nThird one.") == "paragraphs"
    path = tmp_path / "notes.txt"
    path.write_text("one line\nanother line\na third line\n", encoding="utf-8")
    df = build_table(path)
    assert df["path"].tolist() == ["notes.txt#1", "notes.txt#2", "notes.txt#3"]


def test_near_duplicates_are_grouped_by_minhash():
    base = "I was charged twice for my subscription this month and need a refund for the charge"
    df = pd.DataFrame(
        {
            "text": [base, base + " please", "My parcel never arrived at the office address"],
            "class": ["a", "a", "b"],
        }
    )
    assert len(drop_near_duplicates(df, 0.8)) == 2


def test_topics_and_label_check_find_the_misfit():
    billing = ["charged", "refund", "invoice", "payment", "card", "subscription", "billing"]
    shipping = ["parcel", "delivery", "courier", "tracking", "address", "package", "shipped"]
    # each document uses 5 of its label's 7 words
    texts = [billing[i : i + 5] for i in range(3)] + [shipping[i : i + 5] for i in range(3)]
    texts.append(billing[1:6])  # a billing ticket filed under shipping
    labels = ["billing"] * 3 + ["shipping"] * 4
    x, vocab = tfidf(texts)
    suspects, agreement = label_check(x, labels, [5] * 7, [True] * 7)
    assert [s["index"] for s in suspects] == [6] and agreement < 100
    w, h = nmf(x, 2)
    assert w.shape == (7, 2) and h.shape == (2, len(vocab))


def test_snippets_are_masked_and_short():
    quote = snippet("Contact me at jane@example.com " + "word " * 50, 60)
    assert "jane@" not in quote and len(quote) <= 60


def test_logs_are_parsed_into_records(tmp_path):
    app = tmp_path / "app.log"
    app.write_text(
        "2026-09-25 10:00:01,120 INFO [main] Server started\n"
        "2026-09-25 10:00:02,340 ERROR db.pool: Connection failed\n"
        "Traceback (most recent call last):\n"
        '  File "db.py", line 12\n'
        "2026-09-25 10:00:03,001 WARN [worker-1] Retrying\n",
        encoding="utf-8",
    )
    df = parse_log_lines(app)
    assert df.attrs["log_format"] == "app" and len(df) == 3
    assert df["level"].tolist() == ["INFO", "ERROR", "WARN"]
    assert df.loc[1, "extra_lines"] == "2" and "Traceback" in df.loc[1, "message"]

    access = tmp_path / "access.log"
    access.write_text(
        '10.0.0.1 - - [25/Sep/2026:10:00:01 +0000] "GET /index.html HTTP/1.1" 200 512 "-" "curl"\n'
        '10.0.0.2 - bob [25/Sep/2026:10:00:02 +0000] "POST /api HTTP/1.1" 500 17 "-" "Mozilla"\n',
        encoding="utf-8",
    )
    df = parse_log_lines(access)
    assert df.attrs["log_format"] == "access" and df["status"].tolist() == ["200", "500"]


def test_sniffer_routes_logs_prose_and_empty_text(tmp_path):
    def sniffed(name, text):
        path = tmp_path / name
        path.write_bytes(text.encode("utf-8"))
        return sniff(path)

    lines = [
        f'10.0.0.{i} - - [25/Sep/2026:10:00:0{i} +0000] "GET /a HTTP/1.1" 200 5' for i in range(5)
    ]
    assert sniffed("access.log", "\n".join(lines)).format == "log"
    prose = "Hi,\nMy parcel, which I ordered in May, never arrived at my office.\nRegards, Sam\n"
    assert sniffed("ticket.txt", prose).modality == Modality.TEXT
    dated = "\n".join(["timestamp,sensor,value"] + [f"2026-09-25 10:0{i},s1,{i}" for i in range(6)])
    assert sniffed("readings.txt", dated).modality == Modality.TABLE  # a date column isn't a log
    assert sniffed("empty.txt", "").modality == Modality.TEXT


def test_filter_language_takes_codes_or_names_and_the_dry_run_names_the_culprit():
    import pytest

    from mosaic.tables.cleaning import execute_plan
    from mosaic.tables.ops import CleaningOp, CleaningPlan
    from mosaic.text.ops import TEXT_OPS, LanguageParams, text_namespace

    assert LanguageParams(languages=["English", "es"]).languages == ["en", "es"]
    with pytest.raises(ValueError, match="isn't a language code"):
        LanguageParams(languages=["klingon"])
    df = pd.DataFrame(
        {
            "path": [f"d{i}" for i in range(6)],
            "class": [""] * 6,
            "text": ["Hola, mi pedido no ha llegado y quiero saber donde esta"] * 6,
            "suspected_mislabel": [False] * 6,
        }
    )
    plan = CleaningPlan(
        summary="s",
        ops=[
            CleaningOp(
                op="filter_language",
                params={"languages": ["en"]},
                rationale="r",
                evidence=["e"],
            )
        ],
    )
    run = execute_plan(plan, df, catalog=TEXT_OPS, namespace=text_namespace(), unit="documents")
    assert "Step 1 (filter_language) alone removed 6" in run.errors[0]


def test_numbered_evidence_keys_match_the_numbers_in_the_text(tmp_path):
    from mosaic.evidence.store import EvidenceStore
    from mosaic.text.profile import profile_text

    notice = "CONFIDENTIALITY NOTICE: for the named recipient only."
    topics = [
        "refund charge invoice payment card billing subscription receipt plan price fee tax",
        "parcel delivery courier tracking box address package shipment driver door route van",
    ]
    rows = []
    for i in range(30):
        words = topics[i % 2].split()
        body = " ".join(words[(i + j) % 12] for j in range(8))
        rows.append(
            {
                "path": f"d{i}.txt",
                "class": ["billing", "shipping"][i % 2],
                "source_file": f"d{i}.txt",
                "text": f"Ticket {i}: {body}.\n{notice if i % 2 else ''}",
                "suspected_mislabel": False,
            }
        )
    store = EvidenceStore(tmp_path)
    profile_text(pd.DataFrame(rows), store, structure="corpus", total_docs=30)
    boiler = store.get("txt_boilerplate_001")
    assert "line_1:" in boiler.summary
    assert store.lookup_all(["txt_boilerplate_001"], "lines.line_1.share") == [50.0]
    topic = store.get("txt_topics_001")
    first = topic.data["topics"]["topic_1"]["share"]
    assert f"topic_1 ({first}%)" in topic.summary

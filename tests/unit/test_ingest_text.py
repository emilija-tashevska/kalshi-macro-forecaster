"""Unit tests for Phase 1.4 Fed text-corpus ingestion.

A fake client maps deterministic URLs to canned HTML (and ``None`` for
URLs that "don't exist"), so we cover the probe-and-skip behavior, the
HTML→text parser, dedup/idempotency, and that the FTS index is queryable
— all without a network.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from kalshi_train.data.ingest_text import (
    _INDEX_LINK_PATTERNS,
    DOC_BEIGE_BOOK,
    DOC_FED_SPEECH,
    DOC_FOMC_MINUTES,
    DOC_FOMC_STATEMENT,
    beige_book_url,
    build_candidates,
    discover_indexed_candidates,
    extract_index_links,
    fomc_minutes_url,
    fomc_statement_url,
    run_text_ingest,
)
from kalshi_train.data.sources.fed_text import build_document, extract_title, html_to_text
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import TextDocument, bulk_insert_documents

_STATEMENT_HTML = """
<html><head><title>Federal Reserve issues FOMC statement</title></head>
<body>
  <nav>site navigation we do not want</nav>
  <div id="article">
    <h3 class="title">Federal Reserve issues FOMC statement</h3>
    <p>Recent indicators suggest that economic activity has continued to expand
    at a solid pace. The Committee decided to lower the target range for the
    federal funds rate by 1/4 percentage point to 4-3/4 to 5 percent.</p>
    <p>The Committee will continue to assess incoming data and the evolving
    outlook as it considers the extent and timing of additional adjustments.</p>
  </div>
  <footer>footer junk</footer>
  <script>var x = 1;</script>
</body></html>
"""

_MINUTES_HTML = """
<html><head><title>Minutes of the FOMC</title></head>
<body><div id="content">
  <h3 class="title">Minutes of the Federal Open Market Committee</h3>
  <p>The manager turned first to a review of developments in financial markets.
  Participants observed that inflation had eased over the past year but remained
  somewhat elevated relative to the Committee's longer-run goal of 2 percent.</p>
  <p>In their consideration of monetary policy, members agreed that it would be
  appropriate to maintain the target range.</p>
</div></body></html>
"""

# Body must exceed BEIGE_MIN_BODY_CHARS (2000), mirroring real reports and
# the TOC-stub filter; repeat a paragraph to get there.
_BEIGE_PARA = (
    "<p>Overall economic activity rose slightly in most Districts. Consumer "
    "spending was mixed, with several Districts noting softening demand for "
    "discretionary goods. Employment grew modestly and price pressures "
    "continued to moderate across manufacturing and services alike.</p>"
)
_BEIGE_HTML = f"""
<html><head><title>Beige Book</title></head>
<body><div id="article">
  <h2 class="title">Summary of Commentary on Current Economic Conditions</h2>
  {_BEIGE_PARA * 10}
</div></body></html>
"""


class _FakeFedClient:
    """Duck-typed FedTextClient: serves canned HTML, 404s everything else."""

    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages
        self.requested: list[str] = []

    async def __aenter__(self) -> _FakeFedClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def fetch(self, url: str) -> str | None:
        self.requested.append(url)
        return self._pages.get(url)


# ── pure parser ───────────────────────────────────────────────────────


def test_html_to_text_strips_chrome_and_keeps_body() -> None:
    text = html_to_text(_STATEMENT_HTML)
    assert "federal funds rate" in text
    assert "site navigation" not in text
    assert "footer junk" not in text
    assert "var x" not in text


def test_extract_title_prefers_heading() -> None:
    assert extract_title(_STATEMENT_HTML) == "Federal Reserve issues FOMC statement"


def test_build_document_rejects_thin_pages() -> None:
    assert build_document(
        "<html><body><div id='article'>tiny</div></body></html>",
        source="fed",
        document_type="fomc_statement",
        url="/x.htm",
        published_date="2024-01-31",
    ) is None


def test_build_document_builds_textdocument() -> None:
    doc = build_document(
        _STATEMENT_HTML,
        source="fed",
        document_type=DOC_FOMC_STATEMENT,
        url="https://www.federalreserve.gov/x.htm",
        published_date="2024-09-18",
        effective_date="2024-09-18",
        title="FOMC Statement",
    )
    assert doc is not None
    assert doc.document_type == DOC_FOMC_STATEMENT
    assert "lower the target range" in doc.body


# ── URL builders ──────────────────────────────────────────────────────


def test_url_builders_format_dates() -> None:
    m = date(2024, 9, 18)
    assert fomc_statement_url(m) == "/newsevents/pressreleases/monetary20240918a.htm"
    assert fomc_minutes_url(m) == "/monetarypolicy/fomcminutes20240918.htm"
    assert beige_book_url(2024, 3) == "/monetarypolicy/beigebook202403.htm"


def test_build_candidates_uses_calendar(tmp_db: Path, tmp_path: Path) -> None:
    cal = tmp_path / "fomc.txt"
    cal.write_text("2024-01-31\n2024-03-20\n")
    # tmp_db has an empty event_calendar, so prefer_db falls back to the
    # static file rather than reading the real project DB.
    cands = build_candidates(
        start="2024-01-01",
        end="2024-12-31",
        document_types=[DOC_FOMC_STATEMENT, DOC_FOMC_MINUTES],
        db_path=tmp_db,
        calendar_path=cal,
    )
    statements = [c for c in cands if c.document_type == DOC_FOMC_STATEMENT]
    minutes = [c for c in cands if c.document_type == DOC_FOMC_MINUTES]
    assert len(statements) == 2
    assert len(minutes) == 2
    # Minutes get an approximate +21d publication date; effective stays the meeting.
    jan_minutes = next(c for c in minutes if c.effective_date == date(2024, 1, 31))
    assert jan_minutes.published_date == date(2024, 2, 21)


# ── orchestrator ──────────────────────────────────────────────────────


def _make_client(cal_dates: list[str]) -> _FakeFedClient:
    pages: dict[str, str] = {}
    for d in cal_dates:
        m = date.fromisoformat(d)
        pages[fomc_statement_url(m)] = _STATEMENT_HTML
        pages[fomc_minutes_url(m)] = _MINUTES_HTML
    pages[beige_book_url(2024, 3)] = _BEIGE_HTML  # only one beige book "exists"
    return _FakeFedClient(pages)


async def test_run_text_ingest_stores_documents(tmp_db: Path, tmp_path: Path) -> None:
    cal = tmp_path / "fomc.txt"
    cal.write_text("2024-01-31\n2024-03-20\n")
    client = _make_client(["2024-01-31", "2024-03-20"])

    report = await run_text_ingest(
        start="2024-01-01",
        end="2024-12-31",
        db_path=tmp_db,
        calendar_path=cal,
        client=client,
    )

    # 2 statements + 2 minutes + 1 beige book = 5 stored.
    assert report.total_stored == 5
    by_type = {r.document_type: r for r in report.results}
    assert by_type[DOC_FOMC_STATEMENT].stored == 2
    assert by_type[DOC_FOMC_MINUTES].stored == 2
    assert by_type[DOC_BEIGE_BOOK].stored == 1
    # Most beige-book months 404'd → counted as missing.
    assert by_type[DOC_BEIGE_BOOK].missing >= 10

    with connect(tmp_db, read_only=True) as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM text_documents").fetchone()["c"]
        assert count == 5


async def test_run_text_ingest_is_idempotent(tmp_db: Path, tmp_path: Path) -> None:
    cal = tmp_path / "fomc.txt"
    cal.write_text("2024-01-31\n")
    kwargs = {"start": "2024-01-01", "end": "2024-12-31", "db_path": tmp_db, "calendar_path": cal}

    await run_text_ingest(client=_make_client(["2024-01-31"]), **kwargs)
    await run_text_ingest(client=_make_client(["2024-01-31"]), **kwargs)

    with connect(tmp_db, read_only=True) as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM text_documents").fetchone()["c"]
    assert count == 3  # 1 statement + 1 minutes + 1 beige, not doubled


_SPEECH_INDEX_HTML = """
<html><body><div id="article">
  <ul>
    <li><a href="/newsevents/speech/powell20240315a.htm">Economic Outlook</a></li>
    <li><a href="/newsevents/speech/waller20240620a.htm">Inflation and Policy</a></li>
    <li><a href="/newsevents/speech/notadate.htm">Bio page (ignored)</a></li>
  </ul>
</div></body></html>
"""

_SPEECH_HTML = (
    "<html><head><title>Speech</title></head><body><div id='article'>"
    "<h3 class='title'>Economic Outlook</h3>"
    + "<p>Inflation has moderated while the labor market stays resilient. "
    "The Committee remains data dependent and prepared to adjust policy. </p>" * 8
    + "</div></body></html>"
)


def test_extract_index_links_dedups_and_sorts() -> None:
    links = extract_index_links(_SPEECH_INDEX_HTML, _INDEX_LINK_PATTERNS[DOC_FED_SPEECH])
    assert links == [
        "/newsevents/speech/powell20240315a.htm",
        "/newsevents/speech/waller20240620a.htm",
    ]  # the non-dated bio link is excluded


async def test_discover_indexed_candidates_parses_dates() -> None:
    client = _FakeFedClient({"/newsevents/speech/2024-speeches.htm": _SPEECH_INDEX_HTML})
    cands = await discover_indexed_candidates(
        client, DOC_FED_SPEECH, date(2024, 1, 1), date(2024, 12, 31)
    )
    dates = sorted(c.published_date for c in cands)
    assert dates == [date(2024, 3, 15), date(2024, 6, 20)]
    assert all(c.document_type == DOC_FED_SPEECH for c in cands)


async def test_run_text_ingest_crawls_speeches(tmp_db: Path) -> None:
    pages = {
        "/newsevents/speech/2024-speeches.htm": _SPEECH_INDEX_HTML,
        "/newsevents/speech/powell20240315a.htm": _SPEECH_HTML,
        "/newsevents/speech/waller20240620a.htm": _SPEECH_HTML,
    }
    client = _FakeFedClient(pages)
    report = await run_text_ingest(
        start="2024-01-01",
        end="2024-12-31",
        document_types=[DOC_FED_SPEECH],
        db_path=tmp_db,
        client=client,
    )
    by_type = {r.document_type: r for r in report.results}
    assert by_type[DOC_FED_SPEECH].stored == 2
    with connect(tmp_db, read_only=True) as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS c FROM text_documents WHERE document_type='fed_speech'"
        ).fetchone()["c"]
    assert n == 2


def test_bulk_insert_documents_survives_natural_key_collision(tmp_db: Path) -> None:
    """Two distinct URLs sharing (source, type, date, title) must not crash."""
    body = "x" * 400
    docs = [
        TextDocument(
            source="fed",
            document_type="fed_speech",
            title="Welcoming Remarks",
            published_date="2024-05-01",
            body=body,
            url="https://x/a.htm",
        ),
        TextDocument(
            source="fed",
            document_type="fed_speech",
            title="Welcoming Remarks",  # same title+date, different URL
            published_date="2024-05-01",
            body=body,
            url="https://x/b.htm",
        ),
    ]
    with connect(tmp_db) as conn:
        written = bulk_insert_documents(conn, docs)
        conn.commit()
        n = conn.execute("SELECT COUNT(*) AS c FROM text_documents").fetchone()["c"]
    assert written == 1  # second collided on the natural key and was skipped
    assert n == 1


async def test_fts_index_is_searchable(tmp_db: Path, tmp_path: Path) -> None:
    cal = tmp_path / "fomc.txt"
    cal.write_text("2024-01-31\n")
    await run_text_ingest(
        start="2024-01-01", end="2024-12-31", db_path=tmp_db, calendar_path=cal,
        client=_make_client(["2024-01-31"]),
    )
    with connect(tmp_db, read_only=True) as conn:
        rows = conn.execute(
            "SELECT doc_id FROM text_documents_fts WHERE text_documents_fts MATCH ?",
            ("inflation",),
        ).fetchall()
    assert len(rows) >= 1  # minutes mention inflation

"""Tests for focused unit parsing and context budget management."""

from __future__ import annotations

from pathlib import Path

from google.genai.types import Content, Part

import fs_explorer.document_cache as document_cache_module
import fs_explorer.document_parsing as document_parsing_module
import fs_explorer.fs as fs_module
from fs_explorer.context_budget import ContextBudgetManager
from fs_explorer.document_parsing import (
    ParsedDocument,
    ParsedUnit,
    parse_document,
    resolve_requested_unit_nos,
)
from fs_explorer.fs import clear_document_cache, parse_file
from fs_explorer.indexing.pipeline import IndexingPipeline
from fs_explorer.storage import DuckDBStorage


def test_parse_document_selector_anchor_window(tmp_path: Path) -> None:
    path = tmp_path / "memo.md"
    path.write_text(
        "\n\n".join(
            [
                "# Intro\nOverview section.",
                "# Scope\nScope details.",
                "# Pricing\nPurchase price and payment schedule.",
                "# Risks\nRisk disclosures.",
                "# Closing\nClosing requirements.",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_document(str(path))
    assert len(parsed.units) >= 3

    focused = parse_document(
        str(path),
        selector={"anchor": 3, "window": 1, "max_units": 3},
    )
    assert [unit.unit_no for unit in focused.units] == [2, 3, 4]
    assert any("Purchase price" in unit.markdown for unit in focused.units)


def test_parse_file_focused_mode_returns_unit_headers(tmp_path: Path) -> None:
    clear_document_cache()
    path = tmp_path / "report.md"
    path.write_text(
        "\n\n".join(
            [
                "# Intro\nBaseline summary.",
                "# Pricing\nPurchase price and earnout clause.",
                "# Annex\nAdditional notes.",
            ]
        ),
        encoding="utf-8",
    )

    content = parse_file(
        str(path),
        focus_hint="purchase price",
        anchor=2,
        window=0,
        max_units=1,
    )
    assert "FOCUSED PARSE" in content
    assert "[UNIT 2" in content
    assert "Purchase price" in content


def test_parse_document_pdf_selector_uses_requested_pages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    calls: list[dict[str, object]] = []

    class StubPyMuPDF4LLM:
        @staticmethod
        def to_markdown(doc, *, pages=None, page_chunks=False, **kwargs):  # noqa: ANN001, ANN003
            calls.append(
                {
                    "pages": list(pages) if pages is not None else None,
                    "page_chunks": page_chunks,
                }
            )
            return [
                {"page": 2, "text": "Page 2 text"},
                {"page": 3, "text": "Target page 3 text"},
                {"page": 4, "text": "Page 4 text"},
            ]

    monkeypatch.setattr(document_parsing_module, "pymupdf4llm", StubPyMuPDF4LLM)
    monkeypatch.setattr(document_parsing_module, "fitz", None)

    parsed = parse_document(
        str(pdf_path),
        selector={"anchor": 3, "window": 1, "max_units": 3},
    )

    assert calls == [{"pages": [1, 2, 3], "page_chunks": True}]
    assert [unit.unit_no for unit in parsed.units] == [2, 3, 4]


def test_parse_document_pdf_query_only_uses_probed_pages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "query-only.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    calls: list[dict[str, object]] = []

    class StubPage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, mode: str) -> str:
            if mode == "text":
                return self._text
            return f"markdown::{self._text}"

    class StubDocument:
        def __init__(self) -> None:
            self.page_count = 4
            self._pages = [
                StubPage("概述信息"),
                StubPage("公司现任董事会成员如下 董事会成员名单"),
                StubPage("监事和高级管理人员信息"),
                StubPage("其他附注"),
            ]

        def load_page(self, index: int) -> StubPage:
            return self._pages[index]

        def close(self) -> None:
            return None

    class StubFitz:
        @staticmethod
        def open(file_path: str) -> StubDocument:
            return StubDocument()

    class StubPyMuPDF4LLM:
        @staticmethod
        def to_markdown(doc, *, pages=None, page_chunks=False, **kwargs):  # noqa: ANN001, ANN003
            calls.append(
                {
                    "pages": list(pages) if pages is not None else None,
                    "page_chunks": page_chunks,
                }
            )
            return [
                {"page": 2, "text": "董事会成员名单"},
                {"page": 3, "text": "监事和高级管理人员"},
            ]

    monkeypatch.setattr(document_parsing_module, "fitz", StubFitz)
    monkeypatch.setattr(document_parsing_module, "pymupdf4llm", StubPyMuPDF4LLM)

    parsed = parse_document(
        str(pdf_path),
        selector={"query": "董事会成员 董事 监事 高级管理人员", "max_units": 2},
    )

    assert calls == [{"pages": [1, 2], "page_chunks": True}]
    assert [unit.unit_no for unit in parsed.units] == [2, 3]


def test_resolve_requested_unit_nos_probes_pdf_query_pages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "resolve-pages.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")

    class StubPage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return self._text

    class StubDocument:
        def __init__(self) -> None:
            self.page_count = 3
            self._pages = [
                StubPage("封面"),
                StubPage("董事会成员名单"),
                StubPage("监事会成员"),
            ]

        def load_page(self, index: int) -> StubPage:
            return self._pages[index]

        def close(self) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class StubFitz:
        @staticmethod
        def open(file_path: str) -> StubDocument:
            return StubDocument()

    monkeypatch.setattr(document_parsing_module, "fitz", StubFitz)

    requested = resolve_requested_unit_nos(
        str(pdf_path),
        {"query": "董事会成员 监事会", "max_units": 2},
    )

    assert requested == [2, 3]
    assert document_cache_module._requested_unit_nos(
        str(pdf_path),
        document_parsing_module.ParseSelector(query="董事会成员 监事会", max_units=2),
    ) == [2, 3]


def test_parse_file_focused_pdf_bypasses_full_document_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "focused.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")
    cache_calls = {"count": 0}

    def fake_cached_or_parse(file_path: str):  # noqa: ANN001
        cache_calls["count"] += 1
        return ParsedDocument(
            parser_name="pymupdf4llm",
            parser_version="m2-v1",
            units=(
                ParsedUnit(
                    unit_no=1,
                    markdown="full document parse",
                    content_hash="hash-full",
                    source_locator="page-1",
                ),
            ),
        )

    def fake_parse_document(file_path: str, selector=None):  # noqa: ANN001, ANN002
        return ParsedDocument(
            parser_name="pymupdf4llm",
            parser_version="m2-v1",
            units=(
                ParsedUnit(
                    unit_no=48,
                    markdown="focused page 48",
                    content_hash="hash-48",
                    source_locator="page-48",
                ),
            ),
        )

    monkeypatch.setattr(fs_module, "_get_cached_or_parse", fake_cached_or_parse)
    monkeypatch.setattr(fs_module, "parse_document", fake_parse_document)

    content = parse_file(
        str(pdf_path),
        focus_hint="董事会成员",
        anchor=48,
        window=1,
        max_units=4,
    )

    assert cache_calls["count"] == 0
    assert "FOCUSED PARSE" in content
    assert "[UNIT 48" in content


def test_indexing_writes_source_unit_no_for_chunks(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "agreement.md").write_text(
        "\n\n".join(
            [
                "# Intro\nBackground information.",
                "# Commercial Terms\nPurchase price is $45,000,000.",
                "# Closing\nClosing is subject to approvals.",
            ]
        ),
        encoding="utf-8",
    )

    storage = DuckDBStorage(str(tmp_path / "index.duckdb"))
    result = IndexingPipeline(storage=storage).index_folder(str(corpus), discover_schema=True)
    hits = storage.search_chunks(
        corpus_id=result.corpus_id,
        query="purchase price",
        limit=3,
    )
    assert hits
    assert "source_unit_no" in hits[0]
    assert hits[0]["source_unit_no"] is not None


def test_context_budget_manager_hard_limit_and_anchor_bias() -> None:
    manager = ContextBudgetManager(max_input_tokens=180, min_recent_messages=2)
    history = [
        Content(role="user", parts=[Part.from_text(text="Initial question")]),
        Content(
            role="user",
            parts=[
                Part.from_text(
                    text=(
                        "Tool result for parse_file:\n\n"
                        "[UNIT 2 | source=unit-2]\n"
                        + ("old context " * 120)
                    )
                )
            ],
        ),
        Content(
            role="user",
            parts=[
                Part.from_text(
                    text=(
                        "Tool result for parse_file:\n\n"
                        "[UNIT 8 | source=unit-8]\n"
                        + ("anchor context " * 120)
                    )
                )
            ],
        ),
        Content(role="user", parts=[Part.from_text(text="Need final answer now")]),
    ]

    compacted, stats = manager.compact_history(history, anchor_unit_no=8)
    assert int(stats["after_tokens"]) <= int(stats["hard_limit_tokens"])
    blob = "\n".join(
        "\n".join(str(getattr(part, "text", "")) for part in item.parts)
        for item in compacted
    )
    assert "[UNIT 8" in blob

from fs_explorer.context_state import ContextState


def test_context_state_tracks_parse_coverage_and_builds_pack() -> None:
    state = ContextState(task="请输出所有董事会成员")
    state.register_documents(
        [
            {
                "document_id": "doc-1",
                "label": "2024-report.pdf",
                "file_path": "data/object_store/doc-1/report.pdf",
            }
        ]
    )

    summary = state.ingest_parse_result(
        document_id="doc-1",
        file_path="data/object_store/doc-1/report.pdf",
        label="2024-report.pdf",
        units=[
            {
                "unit_no": 48,
                "source_locator": "page-48",
                "heading": "董事会",
                "markdown": "公司现任董事会为第四届董事会。",
            },
            {
                "unit_no": 49,
                "source_locator": "page-49",
                "heading": "董事会成员",
                "markdown": "董事会成员包括张三、李四、王五。",
            },
        ],
        total_units=80,
        focus_hint="董事会成员",
        anchor=48,
        window=2,
        max_units=10,
    )

    assert summary["returned_unit_nos"] == [48, 49]
    snapshot = state.snapshot()
    assert snapshot["context_scope"]["active_document_id"] == "doc-1"
    assert snapshot["coverage_by_document"]["doc-1"]["retrieved_ranges"] == [
        {"start": 48, "end": 49}
    ]

    context_pack, stats = state.build_context_pack(anchor_unit_no=48, max_chars=4000)
    assert "STRUCTURED CONTEXT PACK" in context_pack
    assert "Coverage by document" in context_pack
    assert stats["context_scope"]["active_document_id"] == "doc-1"


def test_context_state_flags_repeated_no_new_coverage() -> None:
    state = ContextState(task="请找董事会成员")
    state.register_documents(
        [
            {
                "document_id": "doc-1",
                "label": "report.pdf",
                "file_path": "report.pdf",
            }
        ]
    )

    kwargs = {
        "document_id": "doc-1",
        "file_path": "report.pdf",
        "label": "report.pdf",
        "units": [
            {
                "unit_no": 48,
                "source_locator": "page-48",
                "heading": "董事会",
                "markdown": "董事会内容。",
            }
        ],
        "total_units": 90,
        "focus_hint": "董事会成员",
        "anchor": 48,
        "window": 1,
        "max_units": 5,
    }

    state.ingest_parse_result(**kwargs)
    state.ingest_parse_result(**kwargs)
    state.ingest_parse_result(**kwargs)

    snapshot = state.snapshot()
    assert any(gap["kind"] == "no_new_coverage" for gap in snapshot["open_gaps"])
    assert snapshot["context_scope"]["active_ranges"] == []

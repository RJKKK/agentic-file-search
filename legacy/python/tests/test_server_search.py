"""Tests for document-scope search APIs."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fs_explorer.server as server_module
from fs_explorer.blob_store import LocalBlobStore
from fs_explorer.server import app


@pytest.fixture()
def client_with_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        server_module,
        "_blob_store",
        LocalBlobStore(tmp_path / "object_store"),
    )
    return TestClient(app)


def _upload(
    client: TestClient,
    *,
    db_path: str,
    name: str,
    content: str,
) -> dict:
    response = client.post(
        "/api/documents",
        data={"db_path": db_path},
        files={"file": (name, content.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 201
    return response.json()["document"]


def test_search_requires_non_empty_selector(
    client_with_store: TestClient,
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "empty.duckdb")
    response = client_with_store.post(
        "/api/search",
        json={"query": "purchase price", "db_path": db_path},
    )
    assert response.status_code == 400


def test_search_supports_document_ids_collection_and_union(
    client_with_store: TestClient,
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "search.duckdb")
    alpha = _upload(
        client_with_store,
        db_path=db_path,
        name="agreement.md",
        content="# Agreement\n\nPurchase price is $45,000,000.\n",
    )
    beta = _upload(
        client_with_store,
        db_path=db_path,
        name="report.md",
        content="# Report\n\nLitigation exposure summary and risk register.\n",
    )

    create_collection = client_with_store.post(
        "/api/collections",
        params={"db_path": db_path},
        json={"name": "Due Diligence"},
    )
    collection_id = create_collection.json()["collection"]["id"]
    client_with_store.post(
        f"/api/collections/{collection_id}/documents",
        params={"db_path": db_path},
        json={"document_ids": [beta["id"]]},
    )

    by_doc = client_with_store.post(
        "/api/search",
        json={
            "query": "purchase price",
            "document_ids": [alpha["id"]],
            "db_path": db_path,
        },
    )
    assert by_doc.status_code == 200
    by_doc_payload = by_doc.json()
    assert by_doc_payload["lazy_indexing"]["triggered"] is False
    assert by_doc_payload["hits"]
    assert by_doc_payload["hits"][0]["doc_id"] == alpha["id"]
    assert by_doc_payload["hits"][0]["source_unit_no"] == 1
    assert by_doc_payload["hits"][0]["absolute_path"].endswith("page-0001.md")

    library_after_first_search = client_with_store.get(
        "/api/documents",
        params={"db_path": db_path},
    ).json()["items"]
    statuses = {item["id"]: item["status"] for item in library_after_first_search}
    assert statuses[alpha["id"]] == "pages_ready"
    assert statuses[beta["id"]] == "pages_ready"

    by_collection = client_with_store.post(
        "/api/search",
        json={
            "query": "litigation",
            "collection_id": collection_id,
            "db_path": db_path,
        },
    )
    assert by_collection.status_code == 200
    assert by_collection.json()["hits"][0]["doc_id"] == beta["id"]

    union_response = client_with_store.post(
        "/api/search",
        json={
            "query": "summary",
            "document_ids": [alpha["id"], beta["id"]],
            "collection_id": collection_id,
            "db_path": db_path,
        },
    )
    assert union_response.status_code == 200
    union_payload = union_response.json()
    assert sorted(union_payload["document_ids"]) == sorted([alpha["id"], beta["id"]])
    assert len({hit["doc_id"] for hit in union_payload["hits"]}) >= 1

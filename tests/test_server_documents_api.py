"""Tests for document-library and collection APIs."""

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


def _upload_markdown(
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


def test_document_library_list_parse_and_delete(
    client_with_store: TestClient,
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "library.duckdb")
    alpha = _upload_markdown(
        client_with_store,
        db_path=db_path,
        name="alpha.md",
        content="# Alpha\n\nPurchase price is $45,000,000.\n",
    )
    beta = _upload_markdown(
        client_with_store,
        db_path=db_path,
        name="beta.md",
        content="## Beta\n\nLitigation exposure remains manageable.\n",
    )

    list_response = client_with_store.get(
        "/api/documents",
        params={"db_path": db_path, "q": "alpha"},
    )
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["original_filename"] == "alpha.md"
    assert payload["items"][0]["status"] == "pages_ready"

    parse_response = client_with_store.post(
        f"/api/documents/{alpha['id']}/parse",
        params={"db_path": db_path},
        json={"mode": "incremental", "force": False, "anchor": 1, "window": 0, "max_units": 1},
    )
    assert parse_response.status_code == 200
    parse_payload = parse_response.json()
    assert parse_payload["document_id"] == alpha["id"]
    assert parse_payload["parsed_units"] >= 1

    detail_response = client_with_store.get(
        f"/api/documents/{alpha['id']}",
        params={"db_path": db_path},
    )
    assert detail_response.status_code == 200
    assert detail_response.json()["document"]["original_filename"] == "alpha.md"

    pages_response = client_with_store.get(
        f"/api/documents/{alpha['id']}/pages",
        params={"db_path": db_path, "page": 1, "page_size": 1},
    )
    assert pages_response.status_code == 200
    assert pages_response.json()["total"] >= 1
    assert "Purchase price" in pages_response.json()["items"][0]["markdown"]

    patch_response = client_with_store.patch(
        f"/api/documents/{beta['id']}",
        params={"db_path": db_path},
        json={"metadata": {"owner": "legal", "priority": "high"}},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["document"]["metadata"]["owner"] == "legal"

    delete_response = client_with_store.delete(
        f"/api/documents/{beta['id']}",
        params={"db_path": db_path},
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    deleted_list = client_with_store.get(
        "/api/documents",
        params={"db_path": db_path, "include_deleted": "true"},
    )
    assert deleted_list.status_code == 200
    listed = {
        item["id"]: item
        for item in deleted_list.json()["items"]
    }
    assert listed[beta["id"]]["status"] == "deleted"


def test_collection_crud_and_document_membership(
    client_with_store: TestClient,
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "collections.duckdb")
    alpha = _upload_markdown(
        client_with_store,
        db_path=db_path,
        name="alpha.md",
        content="# Alpha\n\nBoard members and executive summary.\n",
    )
    beta = _upload_markdown(
        client_with_store,
        db_path=db_path,
        name="beta.md",
        content="# Beta\n\nAdditional appendix.\n",
    )

    create_response = client_with_store.post(
        "/api/collections",
        params={"db_path": db_path},
        json={"name": "Board Pack"},
    )
    assert create_response.status_code == 200
    collection = create_response.json()["collection"]

    attach_response = client_with_store.post(
        f"/api/collections/{collection['id']}/documents",
        params={"db_path": db_path},
        json={"document_ids": [alpha["id"], beta["id"]]},
    )
    assert attach_response.status_code == 200
    assert attach_response.json()["attached"] >= 2

    list_response = client_with_store.get(
        f"/api/collections/{collection['id']}/documents",
        params={"db_path": db_path},
    )
    assert list_response.status_code == 200
    assert {item["id"] for item in list_response.json()["items"]} == {
        alpha["id"],
        beta["id"],
    }

    rename_response = client_with_store.patch(
        f"/api/collections/{collection['id']}",
        params={"db_path": db_path},
        json={"name": "Board Pack Final"},
    )
    assert rename_response.status_code == 200
    assert rename_response.json()["collection"]["name"] == "Board Pack Final"

    detach_response = client_with_store.delete(
        f"/api/collections/{collection['id']}/documents/{beta['id']}",
        params={"db_path": db_path},
    )
    assert detach_response.status_code == 200
    assert detach_response.json()["removed"] is True

    client_with_store.delete(
        f"/api/documents/{alpha['id']}",
        params={"db_path": db_path},
    )
    remaining_response = client_with_store.get(
        f"/api/collections/{collection['id']}/documents",
        params={"db_path": db_path},
    )
    assert remaining_response.status_code == 200
    assert remaining_response.json()["items"] == []

    delete_response = client_with_store.delete(
        f"/api/collections/{collection['id']}",
        params={"db_path": db_path},
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True


def test_upload_failure_cleans_written_blobs(
    client_with_store: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = str(tmp_path / "broken.duckdb")
    original_upsert = server_module.PostgresStorage.upsert_document_stub

    def _boom(self, document):  # noqa: ANN001
        raise RuntimeError("simulated insert failure")

    monkeypatch.setattr(server_module.PostgresStorage, "upsert_document_stub", _boom)
    try:
        response = client_with_store.post(
            "/api/documents",
            data={"db_path": db_path},
            files={"file": ("alpha.md", b"# Alpha\n\nhello\n", "text/markdown")},
        )
    finally:
        monkeypatch.setattr(
            server_module.PostgresStorage,
            "upsert_document_stub",
            original_upsert,
        )

    assert response.status_code == 500
    object_store_root = tmp_path / "object_store"
    assert not (object_store_root / "documents" / "alpha.md").exists()

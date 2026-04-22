"""Tests for the M4 frontend hosting integration."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import fs_explorer.server as server_module
from fs_explorer.server import app


def test_root_prefers_built_frontend_over_legacy(monkeypatch, tmp_path: Path) -> None:
    dist_dir = tmp_path / "frontend-dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text(
        "<!doctype html><html><body>vite frontend</body></html>",
        encoding="utf-8",
    )
    legacy_path = tmp_path / "legacy.html"
    legacy_path.write_text("<html><body>legacy</body></html>", encoding="utf-8")

    monkeypatch.setattr(server_module, "_FRONTEND_DIST", dist_dir)
    monkeypatch.setattr(server_module, "_LEGACY_UI", legacy_path)

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "vite frontend" in response.text
    assert "legacy" not in response.text


def test_spa_fallback_serves_assets_and_preserves_api_404(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "frontend-dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text(
        "<!doctype html><html><body>spa shell</body></html>",
        encoding="utf-8",
    )
    (assets_dir / "app.js").write_text("console.log('ok');", encoding="utf-8")

    monkeypatch.setattr(server_module, "_FRONTEND_DIST", dist_dir)
    monkeypatch.setattr(server_module, "_LEGACY_UI", tmp_path / "missing-legacy.html")

    client = TestClient(app)

    asset_response = client.get("/assets/app.js")
    assert asset_response.status_code == 200
    assert "console.log('ok');" in asset_response.text

    spa_response = client.get("/documents/workbench")
    assert spa_response.status_code == 200
    assert "spa shell" in spa_response.text

    api_response = client.get("/api/not-a-real-route")
    assert api_response.status_code == 404
    assert "spa shell" not in api_response.text

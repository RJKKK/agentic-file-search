"""
Reference: legacy/python/src/fs_explorer/document_parsing.py
Reference: legacy/python/src/fs_explorer/fs.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "legacy" / "python" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fs_explorer.document_parsing import (  # noqa: E402
    DocumentParseError,
    ParsedDocument,
    ParsedImage,
    ParsedUnit,
    _coerce_selector,
    _convert_doc_to_docx,
    _html_to_markdown,
    _parse_pdf,
    _parse_with_docling,
    _read_text_file,
    _single_markdown_document,
    select_parsed_document,
)


def _serialize_image(image: ParsedImage) -> dict[str, Any]:
    return {
        "image_hash": image.image_hash,
        "page_no": int(image.page_no),
        "image_index": int(image.image_index),
        "mime_type": image.mime_type,
        "width": image.width,
        "height": image.height,
    }


def _serialize_unit(unit: ParsedUnit) -> dict[str, Any]:
    return {
        "unit_no": int(unit.unit_no),
        "markdown": str(unit.markdown),
        "content_hash": str(unit.content_hash),
        "heading": unit.heading,
        "source_locator": unit.source_locator,
        "images": [_serialize_image(image) for image in unit.images],
    }


def _serialize_document(document: ParsedDocument) -> dict[str, Any]:
    return {
        "parser_name": document.parser_name,
        "parser_version": document.parser_version,
        "units": [_serialize_unit(unit) for unit in document.units],
    }


def _convert_docx_to_pdf(file_path: str) -> tuple[str, str]:
    soffice = shutil.which("soffice")
    if soffice is None:
        raise DocumentParseError(
            file_path=file_path,
            code="docx_conversion_failed",
            message="LibreOffice headless is required for .docx to .pdf conversion. Install `soffice` and retry.",
        )

    output_dir = tempfile.mkdtemp(prefix="fs-explorer-docx-pdf-")
    try:
        completed = subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                output_dir,
                file_path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise DocumentParseError(
            file_path=file_path,
            code="docx_conversion_failed",
            message=str(exc),
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise DocumentParseError(
            file_path=file_path,
            code="docx_conversion_failed",
            message=stderr,
        )

    converted = Path(output_dir) / f"{Path(file_path).stem}.pdf"
    if not converted.exists():
        raise DocumentParseError(
            file_path=file_path,
            code="docx_conversion_failed",
            message="LibreOffice did not produce a PDF output file.",
        )
    return str(converted), output_dir


def _parse_document_for_bridge(
    file_path: str,
    selector: dict[str, Any] | None,
) -> ParsedDocument:
    resolved = str(Path(file_path).resolve())
    effective_selector = _coerce_selector(selector)
    ext = Path(resolved).suffix.lower()

    if ext == ".pdf":
        parsed = _parse_pdf(resolved, selector=effective_selector)
    elif ext == ".docx":
        converted_pdf, output_dir = _convert_docx_to_pdf(resolved)
        try:
            parsed = _parse_pdf(converted_pdf, selector=effective_selector)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
    elif ext == ".md":
        parsed = _single_markdown_document(
            parser_name="markdown",
            markdown=_read_text_file(resolved),
        )
    elif ext == ".html":
        parsed = _single_markdown_document(
            parser_name="html",
            markdown=_html_to_markdown(_read_text_file(resolved)),
        )
    elif ext == ".doc":
        converted = _convert_doc_to_docx(resolved)
        try:
            parsed = _parse_with_docling(converted, parser_name="libreoffice+docling")
        finally:
            try:
                Path(converted).unlink(missing_ok=True)
            except OSError:
                pass
    else:
        parsed = _parse_with_docling(resolved, parser_name="docling")

    return select_parsed_document(parsed, effective_selector)


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw or "{}")
        operation = str(payload.get("operation") or "").strip()
        if operation != "parse_document":
            raise ValueError(f"Unsupported operation: {operation or '<empty>'}")
        file_path = str(payload.get("file_path") or "").strip()
        selector = payload.get("selector")
        if not file_path:
            raise ValueError("file_path is required")
        document = _parse_document_for_bridge(file_path, selector if isinstance(selector, dict) else None)
        sys.stdout.write(json.dumps({"ok": True, "document": _serialize_document(document)}))
        return 0
    except DocumentParseError as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "file_path": exc.file_path,
                        "code": exc.code,
                        "message": exc.message,
                    },
                }
            )
        )
        return 0
    except Exception as exc:
        fallback_path = str(Path(payload.get("file_path") or "").resolve()) if "payload" in locals() else ""
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "file_path": fallback_path or "<unknown>",
                        "code": "bridge_failed",
                        "message": str(exc),
                    },
                }
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

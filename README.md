# Agentic File Search

This repository now has two tracks:

- `node/`: the new phase-1 Node.js agent core rewrite
- `legacy/python/`: the archived Python implementation kept for reference

## Current Default

The default active implementation work now lives in `node/`.

Phase 1 focuses on:

- agent core
- dynamic skill loading from folders
- page-first retrieval strategy
- SQLite-backed document storage for Node

Phase 1 does not include:

- HTTP API
- frontend
- semantic search
- lazy indexing
- metadata extraction
- chunk/schema/embedding storage

## Document Parsing

The Node rewrite now includes a decoupled document parsing runtime that calls Python externally instead of embedding parser/database logic in Node.

- Node entry point: `node/src/runtime/document-parsing.ts`
- Python bridge: `node/python/document_parsing_bridge.py`
- Node-owned parser copy: `node/python/document_parsing.py`
- Legacy reference parser: `legacy/python/src/fs_explorer/document_parsing.py`

Behavior:

- `.pdf`: follows the legacy PDF parsing path
- `.docx`: converts to PDF first, then follows the PDF parsing path
- `.doc`, `.pptx`, `.xlsx`, `.html`, `.md`: stay aligned with the legacy parsing path
- database writes and indexing-side storage sync are intentionally kept out of this runtime layer

## Database

The Node rewrite now has its own SQLite storage backend.

- Runtime module: `node/src/storage/sqlite.ts`
- Path resolution: `node/src/storage/resolve-db-path.ts`
- Driver: `better-sqlite3`
- Physical tables: `collections`, `collection_documents`, `documents`, `document_pages`, `image_semantics`

The legacy `corpora` concept is still present logically, but it is no longer a physical table. Node stores the `root_path <-> corpus_id` mapping as hidden `corpus_scope` rows inside `collections`, while user-visible collections continue to behave like normal reusable document sets.

## Node Phase 1 Skills

Active skills:

- `glob`
- `grep`
- `read`
- `list_indexed_documents`
- `get_document`

Outdated skills:

- `scan_folder`
- `semantic_search`

The preferred strategy remains:

1. `glob`
2. `grep`
3. `read`

Use `list_indexed_documents` and `get_document` only when a `doc_id` or full document body is actually needed.

## Paths

- Node rewrite: [node](./node)
- Node checklist: [docs/revamp/NODE_PHASE1_CHECKLIST.md](./docs/revamp/NODE_PHASE1_CHECKLIST.md)
- Legacy Python archive: [legacy/python](./legacy/python)
- Legacy Python architecture: [legacy/python/ARCHITECTURE.md](./legacy/python/ARCHITECTURE.md)

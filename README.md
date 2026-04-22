# Agentic File Search

This repository now has two tracks:

- repository root: the active phase-1 Node.js implementation
- `legacy/python/`: the archived Python implementation kept for reference

## Current Default

The default active implementation now lives at the repository root.

Phase 1 focuses on:

- agent core
- dynamic skill loading from folders
- real OpenAI-compatible LLM provider wiring
- page-first retrieval strategy
- SQLite-backed document storage for Node
- document library upload/storage services
- exploration session workflow with SSE
- page-manifest index search
- Fastify HTTP API
- migrated legacy Vue frontend served by the Node API

Phase 1 does not include:

- semantic search
- lazy indexing
- metadata extraction
- chunk/schema/embedding storage

## Quick Start

Install and verify the Node service:

```bash
npm install
npm run typecheck
npm test
```

Build the migrated frontend and start the combined API/UI service:

```bash
npm --prefix ./frontend install
npm run frontend:build
npm start
```

Defaults:

- API/UI host: `127.0.0.1`
- API/UI port: `8000`
- Health check: `/api/health`
- Frontend dist: `frontend/dist`

## Local Python Parser Environment

The active Node service still calls a local Python parser bridge for document parsing. Set up these Python packages in your local environment before uploading or reparsing documents:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "docling>=2.55.0" "pymupdf>=1.24.0" pymupdf4llm
```

On macOS/Linux, activate the venv with:

```bash
source .venv/bin/activate
```

Additional system dependency:

- LibreOffice must be installed and `soffice` must be available on `PATH` for `.docx -> .pdf` and legacy `.doc -> .docx` conversion paths.

Python selection:

- Node uses `FS_EXPLORER_PYTHON_BIN` first when it is set.
- If `FS_EXPLORER_PYTHON_BIN` is not set, Node defaults to `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux.
- If your packages are installed in a different Python environment, set `FS_EXPLORER_PYTHON_BIN` to that interpreter path.
- Do not leave `FS_EXPLORER_PYTHON_BIN=python` in your local `.env` unless that exact command resolves to the interpreter with `docling`, `pymupdf`, and `pymupdf4llm` installed in the same environment used by the Node service.

Example on Windows:

```powershell
$env:FS_EXPLORER_PYTHON_BIN = "C:\Users\41948\AppData\Local\Python\bin\python.exe"
npm start
```

Or put this in your local `.env`:

```env
FS_EXPLORER_PYTHON_BIN=C:\Users\41948\AppData\Local\Python\bin\python.exe
```

Notes:

- The Node-owned parser files live in `python/`.
- You do not need to install the full `legacy/python` project to run the Node mainline.
- If you intentionally want the full legacy Python toolchain for comparison, use `legacy/python/pyproject.toml`; that is separate from the Node parser bridge.

## Configuration

The root `.env.example` is copied from the legacy Python implementation so existing environment names remain recognizable during the migration.

Runtime entry points:

- executable server: `src/server/main.ts`
- reusable server factory: `src/server/http-server.ts`
- env loader: `src/runtime/env.ts`
- LLM config: `src/runtime/model-config.ts`
- OpenAI-compatible action model: `src/runtime/openai-compatible-model.ts`

LLM provider precedence follows the legacy names:

- model: `FS_EXPLORER_TEXT_MODEL`, then `TEXT_MODEL_NAME`, defaulting to `gpt-4o-mini`
- API key: `FS_EXPLORER_TEXT_API_KEY`, then `TEXT_API_KEY`, then `OPENAI_API_KEY`
- base URL: `FS_EXPLORER_TEXT_BASE_URL`, then `TEXT_BASE_URL`, then `OPENAI_BASE_URL`

Server-side production controls:

- `FS_EXPLORER_HOST`
- `FS_EXPLORER_PORT`
- `FS_EXPLORER_LOG_LEVEL`
- `FS_EXPLORER_CORS_ORIGINS`
- `FS_EXPLORER_BODY_LIMIT_BYTES`
- `FS_EXPLORER_UPLOAD_LIMIT_BYTES`

## Document Parsing

The Node rewrite now includes a decoupled document parsing runtime that calls Python externally instead of embedding parser/database logic in Node.

- Node entry point: `src/runtime/document-parsing.ts`
- Python bridge: `python/document_parsing_bridge.py`
- Node-owned parser copy: `python/document_parsing.py`
- Legacy reference parser: `legacy/python/src/fs_explorer/document_parsing.py`

Behavior:

- `.pdf`: follows the legacy PDF parsing path
- `.docx`: converts to PDF first, then follows the PDF parsing path
- `.doc`, `.pptx`, `.xlsx`, `.html`, `.md`: stay aligned with the legacy parsing path
- database writes and indexing-side storage sync are intentionally kept out of this runtime layer

## Database

The Node rewrite now has its own SQLite storage backend.

- Runtime module: `src/storage/sqlite.ts`
- Path resolution: `src/storage/resolve-db-path.ts`
- Driver: `better-sqlite3`
- Physical tables: `collections`, `collection_documents`, `documents`, `document_pages`, `image_semantics`

The legacy `corpora` concept is still present logically, but it is no longer a physical table. Node stores the `root_path <-> corpus_id` mapping as hidden `corpus_scope` rows inside `collections`, while user-visible collections continue to behave like normal reusable document sets.

## HTTP/SSE API

The Node rewrite now exposes a Fastify HTTP/SSE service layer.

- Server module: `src/server/http-server.ts`
- Startup module: `src/server/main.ts`
- Dependencies: `fastify`, `@fastify/multipart`, `@fastify/cors`, `@fastify/static`
- Exported entry points: `createHttpServer(...)`, `runServer(...)`

Implemented route groups:

- `/api/health`: service health probe
- `/api/documents`: upload, list, detail, metadata patch, delete, reparse, page listing
- `/api/collections`: collection CRUD plus document attach/detach
- `/api/search`: scoped page-manifest search
- `/api/explore/sessions`: create/get/reply plus `/events` SSE stream

Legacy folder indexing endpoints are present but return explicit `501` responses in phase 1.

## Frontend

The legacy Vue frontend has been copied into `frontend/` without rewriting its API assumptions.

- Source: `frontend/src`
- Dev server: `npm run frontend:dev`
- Production build: `npm run frontend:build`
- Static serving: Fastify serves `frontend/dist/assets` and returns `frontend/dist/index.html` for non-API routes

The frontend continues to call relative `/api/...` endpoints, so it can be served from the same Node process after `npm run frontend:build`.

## Document Library Storage

The Node rewrite now includes the document library storage chain.

- Local object store: `src/runtime/blob-store.ts`
- Page blob persistence: `src/runtime/page-store.ts`
- Page manifest readback: `src/runtime/document-pages.ts`
- Library scope helpers: `src/runtime/document-library.ts`
- Upload/delete/reparse orchestration: `src/runtime/document-library-service.ts`

Uploads preserve the legacy sequence: validate filename, store source blob, parse through the Python bridge, write page blobs, sync SQLite manifests, write image semantic placeholders, and update parse state.

## Exploration Sessions

The Node rewrite now includes an exploration session workflow exposed through HTTP/SSE.

- Session manager: `src/runtime/explore-sessions.ts`
- Workflow service: `src/runtime/exploration-workflow.ts`
- Trace helpers: `src/runtime/exploration-trace.ts`

The workflow keeps the legacy selected-document scope, `start/context_scope_updated/tool_call/go_deeper/ask_human/complete/error` event stream, ask-human pause/resume behavior, and page-first prompts.

## Index Search

The Node rewrite now includes the non-embedding page search path from the legacy server.

- Runtime module: `src/runtime/index-search.ts`
- Shared types: `src/types/search.ts`

This search path reads SQLite document/page manifests, loads page markdown from the blob store, applies the legacy query-term scoring/snippet rules, and returns ranked page hits. It also powers index-aware `glob`, `grep`, `read`, `list_indexed_documents`, and `get_document` behavior inside exploration sessions.

Still intentionally out of scope: `semantic_search`, embeddings, lazy indexing, metadata extraction, and metadata filter parsing.

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

- Node runtime: [src](./src)
- Dynamic skills: [skills](./skills)
- Parser bridge: [python](./python)
- Frontend: [frontend](./frontend)
- Env example: [.env.example](./.env.example)
- Node checklist: [docs/revamp/NODE_PHASE1_CHECKLIST.md](./docs/revamp/NODE_PHASE1_CHECKLIST.md)
- Legacy Python archive: [legacy/python](./legacy/python)
- Legacy Python architecture: [legacy/python/ARCHITECTURE.md](./legacy/python/ARCHITECTURE.md)

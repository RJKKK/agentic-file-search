# AGENT.md

## Purpose

This file is the working guide for coding agents operating in this repository.

The current active product is the Node.js and TypeScript implementation at the repository root. The Python implementation under `legacy/python/` is archived reference material unless a task explicitly targets legacy code.

## Source Of Truth

When docs disagree, prefer sources in this order:

1. `src/`
2. `tests/`
3. `README.md`
4. `ARCHITECTURE.md`
5. older docs under `docs/` or `legacy/`

Important example: some environment examples still mention older storage settings, but the active server uses SQLite by default via `src/storage/resolve-db-path.ts`.

## Active Stack

- Backend: Node.js + TypeScript + Fastify
- Frontend: Vue 3 + Vite in `frontend/`
- Storage: SQLite via `better-sqlite3`
- Parsing bridge: local Python code in `python/`
- Optional semantic retrieval dependency: Qdrant

## Repository Map

- `src/agent/`: agent loop, action generation, context handling
- `src/runtime/`: runtime services, skill loading, parsing, document library, retrieval, and exploration workflow
- `src/server/`: Fastify server entrypoints and HTTP/SSE routes
- `src/storage/`: SQLite implementation and DB path resolution
- `src/types/`: shared schemas and contracts
- `skills/active/`: only skills here are loaded at runtime
- `skills/outdated/`: reference only, not registered
- `python/`: active Python parser bridge used by the Node app
- `frontend/`: migrated legacy UI
- `tests/`: Node test suite
- `legacy/`: archived implementation, not the default behavior target

## Key Runtime Facts

- Server entrypoint: `src/server/main.ts`
- Default host: `127.0.0.1`
- Default port: `8000`
- Health endpoint: `/api/health`
- Frontend build output: `frontend/dist`
- Default SQLite DB path: `data/agentic-file-search.db`
- SQLite path env overrides actually used by code:
  - `FS_EXPLORER_SQLITE_PATH`
  - `FS_EXPLORER_DB_PATH` for legacy-compatible local file paths only

## Local Commands

Install and verify the main app:

```bash
npm install
npm run typecheck
npm test
```

Run the backend:

```bash
npm start
```

Run the backend in watch mode:

```bash
npm run dev
```

Build the frontend:

```bash
npm run frontend:build
```

Run the frontend dev server:

```bash
npm --prefix ./frontend install
npm run frontend:dev
```

## Python Parser Requirements

Document upload and reparse flows depend on the Python bridge in `python/`.

Typical local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "docling>=2.55.0" "pymupdf>=1.24.0" pymupdf4llm pillow opencv-python numpy
```

If the parser environment is not the default venv, set `FS_EXPLORER_PYTHON_BIN` to the correct interpreter.

LibreOffice on `PATH` is required for some office-document conversion flows.

## How To Approach Changes

For most tasks, identify the owning area first:

- API or route behavior: start in `src/server/http-server.ts`
- Retrieval or search behavior: start in `src/runtime/index-search.ts` or `src/runtime/traditional-rag.ts`
- Document upload, parse, or reparse: start in `src/runtime/document-library-service.ts` and `src/runtime/document-parsing.ts`
- Agent tool loading or skill registration: start in `src/runtime/load-skills.ts`
- Persistence behavior: start in `src/storage/sqlite.ts`
- Parser-specific bugs: inspect both `src/runtime/document-parsing.ts` and `python/document_parsing.py`
- Frontend behavior: work in `frontend/`, but confirm the API contract in `src/server/`

## Skill System Rules

- Runtime-loaded skills come only from `skills/active/`
- Each active skill folder should contain `manifest.json` and the module declared by `manifest.entry`
- `manifest.status` must be `active`
- `skills/outdated/` should not be wired back into the active registry unless a task explicitly requires a design change

## Testing Expectations

Run targeted tests whenever possible, then broader checks when shared behavior changes.

Common high-signal checks:

- `npm run typecheck`
- `npm test`

Useful test files by area:

- `tests/http-server.test.ts`: API and server integration behavior
- `tests/document-library-service.test.ts`: upload, storage, and reparse flows
- `tests/document-parsing.test.ts`: parser bridge and parsing behavior
- `tests/index-search.test.ts`: page-first search behavior
- `tests/openai-compatible-model.test.ts`: model wiring and config resolution
- `tests/sqlite-storage.test.ts`: persistence behavior

If a task affects the frontend contract, also build the frontend with `npm run frontend:build`.

## Guardrails

- Treat the repository root implementation as the default product
- Do not silently change behavior to match `legacy/` without checking whether tests and current Node routes still agree
- Do not move active logic into `legacy/`
- Prefer small, local changes over broad rewrites
- Preserve legacy-compatible API payload shapes when editing routes unless the task explicitly changes the contract
- Keep retrieval changes aligned with the current page-first flow: `glob -> grep -> read`

## Notes On Mixed Signals

- The repo contains both active and archived implementations, so confirm which one a task really targets
- Some environment examples still contain historical settings that are not the main active storage path for the Node server
- `README.md` and `ARCHITECTURE.md` are broadly accurate, but code and tests should settle any ambiguity

## Done Criteria

A change is usually ready when:

- the modified area has targeted test coverage or a clear manual verification path
- `npm run typecheck` passes for TypeScript changes
- `npm test` passes when backend behavior changed
- `npm run frontend:build` passes when frontend code changed
- any required parser or env assumptions are called out in the handoff

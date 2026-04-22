# Node Phase 1 Checklist

## Scope

- Move the existing Python implementation into `legacy/python/`
- Stand up a TypeScript agent core at the repository root
- Load skills dynamically from folder manifests
- Keep `get_document` and `list_indexed_documents` active
- Archive `scan_folder` and `semantic_search` under `skills/outdated/`

## Active Skills

- `glob`
- `grep`
- `read`
- `list_indexed_documents`
- `get_document`

## Outdated Skills

- `scan_folder`
- `semantic_search`

## Implementation Notes

- Every new Node module must include `Reference:` comments pointing to `legacy/python/...`
- Prompt/tool documentation must be generated from the active runtime registry
- Page-first remains the default strategy
- `list_indexed_documents` and `get_document` are auxiliary, not first-choice tools
- Document parsing is decoupled from database writes and runs through an external Python bridge
- The external Python bridge and parser copy live under `python/`, not `legacy/python/`
- `.pdf` keeps the legacy parser path; `.docx` converts to PDF before parsing; other supported types stay aligned with legacy behavior
- Node storage now uses SQLite via `better-sqlite3`
- Physical Node tables are limited to `collections`, `collection_documents`, `documents`, `document_pages`, `image_semantics`
- Logical corpus mapping is preserved through hidden `collections.kind='corpus_scope'` rows instead of a physical `corpora` table
- Chunk/schema/parsed-unit/embedding persistence is intentionally out of scope for phase 1
- Document library upload/storage is implemented and exposed through Fastify routes
- Local blob storage, page persistence, page readback, library scope resolution, upload, delete, and reparse are aligned with the legacy Python chain
- Exploration sessions are implemented and exposed through Fastify HTTP/SSE routes
- Session snapshots, event history, SSE-compatible event encoding, ask-human pause/resume, trace summaries, and completion stats are aligned with the legacy Python workflow
- Workflow prompts preserve the selected-document start prompt, go-deeper prompt, human-answer prompt, and post-tool page-continuation guidance
- Real LLM provider wiring is implemented through an OpenAI-compatible chat-completions adapter
- Root `.env.example` is copied from the legacy Python implementation so original env names remain visible
- Text model config preserves legacy precedence across `FS_EXPLORER_TEXT_*`, `TEXT_*`, and `OPENAI_*`
- Page-manifest index search is implemented and exposed through the scoped search route
- Search uses SQLite document/page manifests plus blob page markdown and preserves the legacy server scoring/snippet rules
- `glob`, `grep`, `read`, `list_indexed_documents`, and `get_document` now consume the index search service when an index context is available
- `semantic_search`, embeddings, lazy indexing, metadata extraction, and metadata filters remain out of scope
- Fastify HTTP/SSE service layer is implemented for document library, collections, scoped search, and exploration sessions
- Executable startup entry is implemented at `src/server/main.ts`
- Production API peripherals include health check, CORS, multipart/body/upload limits, optional logger, graceful shutdown, and SPA static serving
- Legacy Vue frontend is copied into `frontend/` and can be served from `frontend/dist`
- Legacy folder indexing and auto-profile endpoints return explicit phase-1 `501` responses
- Document upload route supports legacy multipart upload through `@fastify/multipart`

## Verification

- Skill loader ignores `outdated/`
- Duplicate skill ids fail fast
- Prompt includes only active tools
- Agent can complete a basic `glob -> grep -> read` flow
- Agent can use `list_indexed_documents -> get_document` when a `doc_id` is needed
- Node document parsing runtime preserves legacy selector behavior for focused parses
- Python bridge can parse a markdown file end-to-end through the external-call path
- SQLite storage creates only the intended five physical tables
- Hidden corpus-scope rows do not leak into normal collection APIs
- SQLite storage preserves document/page/image-semantic deletion and update behavior
- Blob store rejects root-escaping object keys and supports source/page blob lifecycle
- Document library upload writes source blobs, page blobs, SQLite manifests, and image placeholders
- Reparse uses parse-state cache unless forced
- Uploaded documents can be exposed through the document catalog with real local `pages_dir`
- Exploration sessions replay history and serialize legacy SSE event payloads
- Exploration workflow can complete a scoped `glob -> grep -> read` run
- Exploration workflow can pause on `ask_human` and resume after a human reply
- Index search returns ranked page hits from scoped documents
- Index-aware skills emit candidate-page/read-page events and update structured context
- HTTP routes cover document upload/list/detail/pages/delete, collection attach/detach, scoped search, and exploration session create/get/reply
- SSE route serializes exploration events using the legacy event/data payload format
- OpenAI-compatible action model resolves legacy env precedence and parses action JSON from chat completions
- `npm start` launches the Fastify service from `src/server/main.ts`
- `npm run frontend:build` builds the migrated Vue frontend
- Fastify serves the built frontend for `/` and SPA fallback paths while preserving `/api/...` errors

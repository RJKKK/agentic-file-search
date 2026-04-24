# Node Phase 2 Checklist

## Scope

- Move the existing Python implementation into `legacy/python/`
- Stand up a TypeScript agent core at the repository root
- Load skills dynamically from folder manifests
- Keep `get_document` and `list_indexed_documents` active
- Keep the phase-1 agent workflow stable while adding traditional RAG

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
- Page-first remains the default strategy for agent mode
- `list_indexed_documents` and `get_document` are auxiliary, not first-choice tools
- Document parsing is decoupled from database writes and runs through an external Python bridge
- The external Python bridge and parser copy live under `python/`, not `legacy/python/`
- `.pdf` keeps the legacy parser path; `.docx` converts to PDF before parsing; other supported types stay aligned with legacy behavior
- Node storage now uses SQLite via `better-sqlite3`
- Physical Node tables now include `document_chunks`, `retrieval_chunks`, `image_semantic_cache`, and the `retrieval_chunks_fts` virtual table in addition to the phase-1 tables
- Logical corpus mapping is preserved through hidden `collections.kind='corpus_scope'` rows instead of a physical `corpora` table
- Traditional RAG now persists structured document chunks, retrieval chunks, image semantic cache rows, and optional vector embeddings
- Document library upload/storage is implemented and exposed through Fastify routes
- Local blob storage, page persistence, page readback, library scope resolution, upload, delete, and reparse are aligned with the legacy Python chain
- Exploration sessions are implemented and exposed through Fastify HTTP/SSE routes
- Session snapshots, event history, SSE-compatible event encoding, ask-human pause/resume, trace summaries, and completion stats are aligned with the legacy Python workflow
- Workflow prompts preserve the selected-document start prompt, go-deeper prompt, human-answer prompt, and post-tool page-continuation guidance
- Real LLM provider wiring is implemented through an OpenAI-compatible chat-completions adapter
- Root `.env.example` is copied from the legacy Python implementation so original env names remain visible
- Text model config preserves legacy precedence across `FS_EXPLORER_TEXT_*`, `TEXT_*`, and `OPENAI_*`
- Page-manifest search remains implemented and exposed through the scoped search route for agent mode
- Traditional RAG adds `document_chunks`, `retrieval_chunks`, SQLite FTS5 keyword search, optional Qdrant embedding search, and hybrid scoring
- `glob`, `grep`, `read`, `list_indexed_documents`, and `get_document` now consume the index search service when an index context is available
- Traditional RAG adds keyword / semantic / hybrid retrieval, chunk content endpoint T, and image asset endpoint K
- Fastify HTTP/SSE service layer now also exposes `/api/rag/query`, `/api/document-chunks/:chunkId/content`, and `/api/assets/images/:imageHash`
- Executable startup entry is implemented at `src/server/main.ts`
- Production API peripherals include health check, CORS, multipart/body/upload limits, optional logger, graceful shutdown, and SPA static serving
- Vue frontend now supports both `Agent 检索` and `传统检索` modes on the QA page
- Legacy folder indexing and auto-profile endpoints return explicit phase-1 `501` responses
- Document upload route supports legacy multipart upload through `@fastify/multipart`

## Verification

- Skill loader ignores `outdated/`
- Duplicate skill ids fail fast
- Prompt includes only active tools
- Agent can complete a basic `glob -> grep -> read` flow
- Agent can use `list_indexed_documents -> get_document` when a `doc_id` is needed
- Node document parsing runtime preserves legacy selector behavior for focused parses and now reconstructs PDF page markdown from `pymupdf4llm.to_json`
- Python bridge can parse a markdown file end-to-end through the external-call path
- SQLite storage creates the traditional RAG tables and FTS virtual table alongside the phase-1 tables
- Hidden corpus-scope rows do not leak into normal collection APIs
- SQLite storage preserves document/page/image-semantic deletion and update behavior
- Blob store rejects root-escaping object keys and supports source/page blob lifecycle
- Document library upload writes source blobs, page blobs, SQLite manifests, and image placeholders
- Reparse uses parse-state cache unless forced
- Uploaded documents can be exposed through the document catalog with real local `pages_dir`
- Exploration sessions replay history and serialize legacy SSE event payloads
- Exploration workflow can complete a scoped `glob -> grep -> read` run
- Exploration workflow can pause on `ask_human` and resume after a human reply
- Traditional RAG returns ranked retrieval chunk hits and can collapse oversized split hits back to source chunk summaries
- Index-aware skills emit candidate-page/read-page events and update structured context
- HTTP routes cover document upload/list/detail/pages/delete, collection attach/detach, scoped search, traditional RAG query, chunk readback, image assets, and exploration session create/get/reply
- SSE route serializes exploration events using the legacy event/data payload format
- OpenAI-compatible action model resolves legacy env precedence and parses action JSON from chat completions
- `npm start` launches the Fastify service from `src/server/main.ts`
- `npm run frontend:build` builds the migrated Vue frontend
- Fastify serves the built frontend for `/` and SPA fallback paths while preserving `/api/...` errors

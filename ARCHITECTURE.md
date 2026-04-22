# Node Phase 1 Architecture

## Summary

The new phase-1 architecture is a TypeScript agent core with dynamic skill loading.

- Runtime: repository root
- Archived implementation: `legacy/python/`
- Skill loading model: `manifest + module`

## Runtime Layout

- `src/agent`: action loop, repeat guard, context budget, structured context
- `src/runtime`: skill loading, registry assembly, LLM provider config, document parsing, object storage, page storage, document library services, exploration sessions, and page-manifest index search
- `src/server`: Fastify HTTP/SSE service layer
- `src/storage`: SQLite persistence and database-path resolution
- `src/types`: Zod schemas and shared runtime contracts
- `skills/active`: runtime-loadable skills
- `skills/outdated`: archived skills that must not enter the active registry
- `frontend`: migrated legacy Vue frontend
- `python`: Node-owned Python parser bridge and parser copy

## Retrieval Strategy

The preferred retrieval flow is page-first:

1. `glob`
2. `grep`
3. `read`

Auxiliary paths:

- `list_indexed_documents`
- `get_document`

Outdated paths:

- `scan_folder`
- `semantic_search`

## Dynamic Skills

Each skill lives in its own folder:

- `manifest.json`
- `index.ts`

The loader only activates skills in `skills/active`.

Outdated skills remain versioned in-repo for reference and later replacement design, but they are never added to the runtime tool registry.

## Storage

The Node storage layer uses SQLite via `better-sqlite3`.

- Default runtime database file: `data/agentic-file-search.sqlite`
- Physical tables: `collections`, `collection_documents`, `documents`, `document_pages`, `image_semantics`
- Logical corpus mapping: hidden `collections.kind='corpus_scope'` rows
- User-visible collections: `collections.kind='user'`

The Node storage surface intentionally excludes legacy chunk, schema, parsed-unit, and embedding APIs in phase 1.

## Model Provider

The Node runtime includes an OpenAI-compatible action model for real agent execution.

- `.env` loading is handled by `src/runtime/env.ts`.
- Legacy model environment names are resolved by `src/runtime/model-config.ts`.
- Chat completion calls are implemented by `src/runtime/openai-compatible-model.ts`.
- `createHttpServer` uses the configured provider automatically and falls back to an explicit no-model final answer only when no text API key is configured.

Text-model precedence follows the legacy config: `FS_EXPLORER_TEXT_*`, then `TEXT_*`, then `OPENAI_*` fallbacks.

## Document Library

The Node library chain mirrors the legacy upload path and is exposed through the Fastify document routes.

- Source blobs and generated page blobs are stored in a local object store.
- The shared library corpus remains `blob://library/default`.
- Uploaded documents are persisted into SQLite documents/page manifests.
- `DocumentLibraryService` provides API-ready upload, delete, and reparse orchestration.
- `createLibraryDocumentCatalog` exposes uploaded documents to `list_indexed_documents` and `get_document` with real local `pages_dir` paths.

## Exploration Workflow

The Node exploration workflow mirrors the legacy in-memory workflow and is exposed through Fastify HTTP/SSE routes.

- `ExploreSessionManager` keeps session snapshots, event history, subscribers, retention cleanup, and SSE-compatible event encoding.
- `ExplorationWorkflowService` resolves the selected document or collection scope, materializes library documents, configures the agent with the legacy page-first prompts, and publishes legacy event names.
- Ask-human pauses keep the agent/runtime state in memory and can be resumed through `replyToSession`.
- Completion events include context budget stats, structured context snapshots, trace path, cited sources, and the same lazy-indexing placeholder shape with indexing disabled.

## HTTP/SSE Server

The HTTP layer uses Fastify plus `@fastify/multipart`, `@fastify/cors`, and `@fastify/static`, and keeps the legacy route groups where phase-1 services can back them.

- `src/server/main.ts` is the executable startup entry used by `npm start`.
- Startup loads `.env`, reads `FS_EXPLORER_HOST` / `FS_EXPLORER_PORT`, and installs graceful shutdown handlers.
- `/api/health` provides a production health probe.
- Document routes call `DocumentLibraryService` and preserve trace headers and legacy error-code payloads for upload/list/detail/page operations.
- Collection routes call the SQLite collection APIs while filtering out hidden corpus-scope collections.
- Search routes call `IndexSearchService` and return the legacy page-hit shape.
- Exploration routes call `ExplorationWorkflowService`; `/events` streams SSE payloads using the legacy `event:` and `data:` format with keepalives.
- Legacy folder indexing and metadata auto-profile routes return explicit `501` responses because folder indexing, schema discovery, embeddings, and metadata extraction remain out of scope.

## Frontend Serving

The legacy Vue frontend is copied into `frontend/` and remains API-compatible with the relative `/api/...` routes.

- Development uses the frontend package scripts inside `frontend/`.
- Production uses `npm run frontend:build`.
- Fastify serves `frontend/dist/assets` at `/assets/` and returns `frontend/dist/index.html` for non-API routes.
- API 404 behavior remains separate from the SPA fallback so missing `/api/...` routes still return API-style errors.

## Index Search

The phase-1 search layer mirrors the legacy server's page-manifest search path, not the old chunk/vector search path.

- `IndexSearchService` resolves corpus/document/collection scope from SQLite.
- Search loads page markdown through `document_pages` manifests and blob-store objects.
- Scoring follows the legacy `_search_terms`, `_page_matches_query`, and `_build_search_snippet` behavior.
- Index-aware skills resolve source paths or `pages_dir` paths back to stored documents, emit candidate/read runtime events, and update structured context.
- Metadata filters, embeddings, lazy indexing, and `semantic_search` remain intentionally disabled for this phase.

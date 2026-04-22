# Node Phase 1 Architecture

## Summary

The new phase-1 architecture is a TypeScript agent core with dynamic skill loading.

- Runtime: `node/`
- Archived implementation: `legacy/python/`
- Skill loading model: `manifest + module`

## Runtime Layout

- `node/src/agent`: action loop, repeat guard, context budget, structured context
- `node/src/runtime`: skill loading and registry assembly
- `node/src/storage`: SQLite persistence and database-path resolution
- `node/src/types`: Zod schemas and shared runtime contracts
- `node/skills/active`: runtime-loadable skills
- `node/skills/outdated`: archived skills that must not enter the active registry

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

The loader only activates skills in `node/skills/active`.

Outdated skills remain versioned in-repo for reference and later replacement design, but they are never added to the runtime tool registry.

## Storage

The Node storage layer uses SQLite via `better-sqlite3`.

- Physical tables: `collections`, `collection_documents`, `documents`, `document_pages`, `image_semantics`
- Logical corpus mapping: hidden `collections.kind='corpus_scope'` rows
- User-visible collections: `collections.kind='user'`

The Node storage surface intentionally excludes legacy chunk, schema, parsed-unit, and embedding APIs in phase 1.

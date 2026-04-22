# Node Phase 1 Checklist

## Scope

- Move the existing Python implementation into `legacy/python/`
- Stand up a TypeScript agent core in `node/`
- Load skills dynamically from folder manifests
- Keep `get_document` and `list_indexed_documents` active
- Archive `scan_folder` and `semantic_search` under `node/skills/outdated/`

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
- The external Python bridge and parser copy live under `node/python/`, not `legacy/python/`
- `.pdf` keeps the legacy parser path; `.docx` converts to PDF before parsing; other supported types stay aligned with legacy behavior
- Node storage now uses SQLite via `better-sqlite3`
- Physical Node tables are limited to `collections`, `collection_documents`, `documents`, `document_pages`, `image_semantics`
- Logical corpus mapping is preserved through hidden `collections.kind='corpus_scope'` rows instead of a physical `corpora` table
- Chunk/schema/parsed-unit/embedding persistence is intentionally out of scope for phase 1

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

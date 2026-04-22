# Node Rewrite Alignment Addendum

## Current Position

- The full historical Python implementation has been archived under `legacy/python/`
- The new default implementation now lives at the repository root

## Phase 1 Node Decisions

- `semantic_search` is not part of phase 1
- `scan_folder` is archived as outdated
- `get_document` remains active
- `list_indexed_documents` remains active
- The default strategy is page-first, not indexed retrieval

## Reference Policy

All new Node source files must declare their Python reference files with `Reference:` comments pointing into `legacy/python/`.

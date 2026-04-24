# Traditional RAG Design

## Overview

This repository now supports two retrieval modes on the QA page:

- Agent retrieval: the existing page-first exploration workflow
- Traditional retrieval: keyword, semantic, or hybrid retrieval over persisted retrieval chunks

The traditional path is built on top of the Node runtime and keeps the phase-1 document library flow unchanged at the API boundary.

## Parsing and Chunking

- PDF parsing now prefers `pymupdf4llm.to_json(...)`
- The Python bridge reconstructs page-level markdown from structured blocks so existing page blobs remain compatible with the agent workflow
- Structured blocks are persisted into `document_chunks`
- Query-time chunks are persisted into `retrieval_chunks`

### `document_chunks`

Each row stores:

- `document_id`
- `page_no`
- `document_index`
- `page_index`
- `block_type`
- `bbox_json`
- `content_md`
- `size_class`
- `summary_text`
- `merged_page_nos_json`
- `merged_bboxes_json`

### `retrieval_chunks`

- Small neighboring chunks are merged until they reach the normal range
- Normal chunks are stored as-is
- Oversized chunks are summarized at the source level and split into retrieval-size segments
- Embeddings and BM25 operate only on this table

## Images

- PDF images are extracted through the Python bridge
- OpenCV preprocessing runs before any vision-model request
- Images with no text and high interference are dropped
- Kept images are persisted into blob storage and exposed by `/api/assets/images/:imageHash`
- Vision outputs are cached in `image_semantic_cache`

## Storage and Retrieval

- SQLite stores document metadata, page manifests, structured chunks, retrieval chunks, image semantics, and image semantic cache rows
- SQLite FTS5 powers keyword retrieval over `retrieval_chunks`
- Qdrant stores vector embeddings in one collection per document: `doc_<document_id>`
- Hybrid retrieval normalizes keyword and semantic scores before combining them

## API Surface

- `POST /api/rag/query`
- `GET /api/document-chunks/:chunkId/content`
- `GET /api/assets/images/:imageHash`

## Frontend

The QA page now includes:

- `Agent 检索` for the original trace-heavy workflow
- `传统检索` for direct RAG answers with chunk citations, page numbers, and chunk links

/*
Reference: legacy/python/src/fs_explorer/blob_store.py
Reference: legacy/python/src/fs_explorer/page_store.py
Reference: legacy/python/src/fs_explorer/document_library.py
Reference: legacy/python/src/fs_explorer/document_pages.py
Reference: legacy/python/src/fs_explorer/server.py
*/

import type { DocumentCatalog } from "./skills.js";
import type {
  DocumentParseTaskStatus,
  DocumentParseTaskType,
  DocumentParseStageTiming,
  PublicCollectionRecord,
  RetrievalChunkingStrategy,
  SqliteStorageBackend,
  StoredDocument,
  StoredImageSemantic,
} from "./storage.js";
import type { ParsedDocument } from "./parsing.js";

export interface BlobHead {
  objectKey: string;
  storageUri: string;
  size: number;
  absolutePath: string;
}

export interface BlobStore {
  put(input: { objectKey: string; data: Uint8Array }): Promise<BlobHead>;
  get(input: { objectKey: string }): Promise<Uint8Array>;
  materialize(input: { objectKey: string }): Promise<string>;
  delete(input: { objectKey: string }): Promise<boolean>;
  head(input: { objectKey: string }): Promise<BlobHead | null>;
  deletePrefix(input: { prefix: string }): Promise<number>;
  listPrefix(input: { prefix: string }): Promise<BlobHead[]>;
}

export interface StoredPage {
  pageNo: number;
  objectKey: string;
  heading: string | null;
  sourceLocator: string | null;
  contentHash: string;
  charCount: number;
  isSyntheticPage: boolean;
}

export interface LoadedDocumentPage {
  unit_no: number;
  page_no: number;
  object_key: string;
  heading: string | null;
  source_locator: string | null;
  content_hash: string;
  char_count: number;
  chunk_count?: number;
  is_synthetic_page: boolean;
  page_label: string;
  markdown: string;
  file_path: string;
}

export interface DocumentScope {
  corpusId: string;
  documentIds: string[];
  documents: StoredDocument[];
  collection: PublicCollectionRecord | null;
  collections: PublicCollectionRecord[];
  isEmpty: boolean;
}

export interface UploadDocumentInput {
  filename: string;
  data: Uint8Array;
  contentType?: string | null;
  enableEmbedding?: boolean;
  enableImageSemantic?: boolean;
  chunkingStrategy?: RetrievalChunkingStrategy;
  fixedChunkChars?: number | null;
}

export interface DocumentSummaryPayload {
  id: string;
  corpus_id: string;
  relative_path: string;
  absolute_path: string;
  original_filename: string;
  object_key: string;
  source_object_key: string;
  pages_prefix: string;
  storage_uri: string;
  content_type: string | null;
  upload_status: string;
  page_count: number;
  file_size: number;
  file_mtime: number;
  content_sha256: string;
  parsed_content_sha256: string | null;
  parsed_is_complete: boolean;
  embedding_enabled: boolean;
  has_embeddings: boolean;
  image_semantic_enabled: boolean;
  retrieval_chunking_strategy: RetrievalChunkingStrategy;
  fixed_chunk_chars: number | null;
  is_deleted: boolean;
  status: string;
  metadata: Record<string, unknown>;
  last_indexed_at?: string;
}

export interface DocumentParseTaskPayload {
  id: string;
  document_id: string | null;
  document_filename: string;
  task_type: DocumentParseTaskType;
  status: DocumentParseTaskStatus;
  progress_percent: number;
  current_stage: string | null;
  options: Record<string, unknown>;
  stage_timings: DocumentParseStageTiming[];
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  updated_at: string;
  total_duration_ms: number | null;
}

export interface UploadDocumentResult {
  document: DocumentSummaryPayload;
  task: DocumentParseTaskPayload;
}

export interface ReparseDocumentResult {
  document: DocumentSummaryPayload;
  task: DocumentParseTaskPayload;
}

export interface DeleteDocumentResult {
  document: DocumentSummaryPayload;
  deleted: true;
}

export interface DocumentLibraryServices {
  storage: SqliteStorageBackend;
  blobStore: BlobStore;
  documentCatalog: DocumentCatalog;
}

export interface ParsedDocumentImageSemantic {
  images: StoredImageSemantic[];
  parsedDocument: ParsedDocument;
}


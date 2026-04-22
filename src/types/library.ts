/*
Reference: legacy/python/src/fs_explorer/blob_store.py
Reference: legacy/python/src/fs_explorer/page_store.py
Reference: legacy/python/src/fs_explorer/document_library.py
Reference: legacy/python/src/fs_explorer/document_pages.py
Reference: legacy/python/src/fs_explorer/server.py
*/

import type { DocumentCatalog } from "./skills.js";
import type {
  PublicCollectionRecord,
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
  isEmpty: boolean;
}

export interface UploadDocumentInput {
  filename: string;
  data: Uint8Array;
  contentType?: string | null;
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
  is_deleted: boolean;
  status: string;
  metadata: Record<string, unknown>;
  last_indexed_at?: string;
}

export interface UploadDocumentResult {
  document: DocumentSummaryPayload;
  uploadResult: {
    corpus_id: string;
    storage_uri: string;
    pages_generated: number;
    page_count: number;
    page_naming_scheme: string;
  };
}

export interface ReparseDocumentResult {
  documentId: string;
  pageCount: number;
  pagesUpdated: number;
  fromCache: boolean;
  pages: LoadedDocumentPage[];
  pageNamingScheme: string;
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


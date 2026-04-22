/*
Reference: legacy/python/src/fs_explorer/storage/base.py
Reference: legacy/python/src/fs_explorer/storage/postgres.py
*/

import type { DocumentCatalog } from "./skills.js";

export type CollectionKind = "user" | "corpus_scope";

export interface StorageDocumentRecord {
  id: string;
  corpusId: string;
  relativePath: string;
  absolutePath: string;
  content: string;
  metadataJson: string;
  fileMtime: number;
  fileSize: number;
  contentSha256: string;
  originalFilename?: string;
  objectKey?: string;
  sourceObjectKey?: string;
  pagesPrefix?: string;
  storageUri?: string;
  contentType?: string | null;
  uploadStatus?: string;
  pageCount?: number;
  parsedContentSha256?: string | null;
  parsedIsComplete?: boolean;
}

export interface StoredDocument {
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
  content: string;
  metadata_json: string;
  file_mtime: number;
  file_size: number;
  content_sha256: string;
  parsed_content_sha256: string | null;
  parsed_is_complete: boolean;
  last_indexed_at: string;
  is_deleted: boolean;
}

export interface StoredCollection {
  id: string;
  name: string;
  kind: CollectionKind;
  corpus_id: string | null;
  root_path: string | null;
  is_deleted: boolean;
  created_at: string;
  updated_at: string;
}

export interface PublicCollectionRecord {
  id: string;
  name: string;
  is_deleted: boolean;
  created_at: string;
  updated_at: string;
}

export interface StoredDocumentPage {
  document_id: string;
  page_no: number;
  object_key: string;
  content_hash: string;
  char_count: number;
  is_synthetic_page: boolean;
  heading: string | null;
  source_locator: string | null;
}

export interface StorageDocumentPageRecord {
  documentId: string;
  pageNo: number;
  objectKey: string;
  contentHash: string;
  charCount: number;
  isSyntheticPage: boolean;
  heading?: string | null;
  sourceLocator?: string | null;
}

export interface StoredImageSemantic {
  image_hash: string;
  source_document_id: string;
  source_page_no: number;
  source_image_index: number;
  mime_type: string | null;
  width: number | null;
  height: number | null;
  semantic_text: string | null;
  semantic_model: string | null;
}

export interface StorageImageSemanticRecord {
  imageHash: string;
  sourceDocumentId: string;
  sourcePageNo: number;
  sourceImageIndex: number;
  mimeType?: string | null;
  width?: number | null;
  height?: number | null;
  semanticText?: string | null;
  semanticModel?: string | null;
}

export interface SqliteStorageOptions {
  dbPath: string;
  readOnly?: boolean;
}

export interface SqliteStorageBackend {
  initialize(): void;
  close(): void;
  getOrCreateCorpus(rootPath: string): string;
  getCorpusId(rootPath: string): string | null;
  getCorpusRootPath(corpusId: string): string | null;
  upsertDocumentStub(document: StorageDocumentRecord): void;
  getDocument(docId: string): StoredDocument | null;
  listDocuments(corpusId: string, includeDeleted?: boolean): StoredDocument[];
  listDocumentsByIds(docIds: string[], includeDeleted?: boolean): StoredDocument[];
  setDocumentDeleted(docId: string, isDeleted: boolean): StoredDocument | null;
  deleteDocument(docId: string): StoredDocument | null;
  updateDocumentAbsolutePath(docId: string, absolutePath: string): StoredDocument | null;
  updateDocumentMetadata(docId: string, metadataJson: string): StoredDocument | null;
  updateDocumentParseState(
    docId: string,
    parsedContentSha256: string | null,
    parsedIsComplete: boolean,
  ): StoredDocument | null;
  syncDocumentPages(
    documentId: string,
    pages: StorageDocumentPageRecord[],
  ): { upserted: number; untouched: number; deleted: number };
  listDocumentPages(documentId: string, pageNos?: number[] | null): StoredDocumentPage[];
  upsertImageSemantics(images: StorageImageSemanticRecord[]): number;
  getImageSemantics(imageHashes: string[]): Record<string, StoredImageSemantic>;
  listImageSemanticsForDocument(
    documentId: string,
    pageNos?: number[] | null,
  ): StoredImageSemantic[];
  updateImageSemantic(imageHash: string, semanticText: string, semanticModel?: string | null): void;
  createCollection(name: string): PublicCollectionRecord;
  listCollections(includeDeleted?: boolean): PublicCollectionRecord[];
  getCollection(collectionId: string): PublicCollectionRecord | null;
  updateCollection(collectionId: string, name: string): PublicCollectionRecord | null;
  setCollectionDeleted(collectionId: string, isDeleted: boolean): PublicCollectionRecord | null;
  listCollectionDocuments(collectionId: string, includeDeleted?: boolean): StoredDocument[];
  attachDocumentsToCollection(collectionId: string, documentIds: string[]): number;
  detachDocumentFromCollection(collectionId: string, docId: string): boolean;
  removeDocumentFromAllCollections(docId: string): number;
  createDocumentCatalog(corpusId: string): DocumentCatalog;
}

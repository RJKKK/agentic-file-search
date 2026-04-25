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
  embeddingEnabled?: boolean;
  hasEmbeddings?: boolean;
  imageSemanticEnabled?: boolean;
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
  embedding_enabled: boolean;
  has_embeddings: boolean;
  image_semantic_enabled: boolean;
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
  chunk_count: number;
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
  chunkCount?: number;
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
  bbox_json: string | null;
  object_key: string | null;
  storage_uri: string | null;
  has_text: boolean | null;
  interference_score: number | null;
  is_dropped: boolean;
  recognizable: boolean | null;
  accessible_url: string | null;
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
  bboxJson?: string | null;
  objectKey?: string | null;
  storageUri?: string | null;
  hasText?: boolean | null;
  interferenceScore?: number | null;
  isDropped?: boolean;
  recognizable?: boolean | null;
  accessibleUrl?: string | null;
  semanticText?: string | null;
  semanticModel?: string | null;
}

export type DocumentChunkSizeClass = "small" | "normal" | "oversized";

export interface StoredDocumentChunk {
  id: string;
  document_id: string;
  ordinal: number;
  reference_retrieval_chunk_id: string | null;
  page_no: number;
  document_index: number;
  page_index: number;
  block_type: string;
  bbox_json: string;
  content_md: string;
  size_class: DocumentChunkSizeClass;
  summary_text: string | null;
  is_split_from_oversized: boolean;
  split_index: number;
  split_count: number;
  merged_page_nos_json: string;
  merged_bboxes_json: string;
}

export interface StorageDocumentChunkRecord {
  id: string;
  documentId: string;
  ordinal?: number;
  referenceRetrievalChunkId?: string | null;
  pageNo: number;
  documentIndex: number;
  pageIndex: number;
  blockType: string;
  bboxJson: string;
  contentMd: string;
  sizeClass: DocumentChunkSizeClass;
  summaryText?: string | null;
  isSplitFromOversized?: boolean;
  splitIndex?: number;
  splitCount?: number;
  mergedPageNosJson?: string;
  mergedBboxesJson?: string;
}

export interface StoredRetrievalChunk {
  id: string;
  document_id: string;
  ordinal: number;
  content_md: string;
  size_class: DocumentChunkSizeClass;
  summary_text: string | null;
  source_document_chunk_ids_json: string;
  page_nos_json: string;
  source_locator: string | null;
  bboxes_json: string;
}

export interface StorageRetrievalChunkRecord {
  id: string;
  documentId: string;
  ordinal: number;
  contentMd: string;
  sizeClass: DocumentChunkSizeClass;
  summaryText?: string | null;
  sourceDocumentChunkIdsJson: string;
  pageNosJson: string;
  sourceLocator?: string | null;
  bboxesJson?: string;
}

export interface StoredImageSemanticCache {
  image_hash: string;
  prompt_version: string;
  recognizable: boolean;
  image_kind: string | null;
  contains_text: boolean | null;
  visible_text: string | null;
  summary: string | null;
  entities_json: string;
  keywords_json: string;
  qa_hints_json: string;
  drop_reason: string | null;
  semantic_model: string | null;
}

export type DocumentParseTaskType = "upload_parse" | "reparse" | "embed_only";
export type DocumentParseTaskStatus = "queued" | "running" | "completed" | "failed";
export type DocumentParseStageStatus = "pending" | "running" | "completed" | "skipped" | "failed";

export interface DocumentParseStageTiming {
  stage: string;
  label: string;
  status: DocumentParseStageStatus;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
}

export interface StoredDocumentParseTask {
  id: string;
  document_id: string | null;
  document_filename: string;
  task_type: DocumentParseTaskType;
  status: DocumentParseTaskStatus;
  progress_percent: number;
  current_stage: string | null;
  options_json: string;
  stage_timings_json: string;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  updated_at: string;
  total_duration_ms: number | null;
}

export interface StorageDocumentParseTaskRecord {
  id: string;
  documentId?: string | null;
  documentFilename: string;
  taskType: DocumentParseTaskType;
  status: DocumentParseTaskStatus;
  progressPercent?: number;
  currentStage?: string | null;
  optionsJson?: string;
  stageTimingsJson?: string;
  errorMessage?: string | null;
  createdAt?: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  updatedAt?: string;
  totalDurationMs?: number | null;
}

export interface DocumentChunkKeywordHit {
  document_chunk_id: string;
  document_id: string;
  reference_retrieval_chunk_id: string | null;
  ordinal: number;
  content_md: string;
  size_class: DocumentChunkSizeClass;
  is_split_from_oversized: boolean;
  score: number;
}

export interface StorageImageSemanticCacheRecord {
  imageHash: string;
  promptVersion: string;
  recognizable: boolean;
  imageKind?: string | null;
  containsText?: boolean | null;
  visibleText?: string | null;
  summary?: string | null;
  entitiesJson?: string;
  keywordsJson?: string;
  qaHintsJson?: string;
  dropReason?: string | null;
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
  updateDocumentUploadStatus(docId: string, uploadStatus: string): StoredDocument | null;
  updateDocumentFeatureFlags(
    docId: string,
    input: {
      embeddingEnabled?: boolean;
      hasEmbeddings?: boolean;
      imageSemanticEnabled?: boolean;
    },
  ): StoredDocument | null;
  syncDocumentPages(
    documentId: string,
    pages: StorageDocumentPageRecord[],
  ): { upserted: number; untouched: number; deleted: number };
  listDocumentPages(documentId: string, pageNos?: number[] | null): StoredDocumentPage[];
  replaceDocumentChunks(
    documentId: string,
    chunks: StorageDocumentChunkRecord[],
  ): { inserted: number; deleted: number };
  listDocumentChunks(documentId: string): StoredDocumentChunk[];
  getDocumentChunk(chunkId: string): StoredDocumentChunk | null;
  replaceRetrievalChunks(
    documentId: string,
    chunks: StorageRetrievalChunkRecord[],
  ): { inserted: number; deleted: number };
  getRetrievalChunk(chunkId: string): StoredRetrievalChunk | null;
  listRetrievalChunks(documentId: string): StoredRetrievalChunk[];
  keywordSearchDocumentChunks(input: {
    query: string;
    documentIds: string[];
    limit: number;
  }): DocumentChunkKeywordHit[];
  upsertImageSemantics(images: StorageImageSemanticRecord[]): number;
  getImageSemantics(imageHashes: string[]): Record<string, StoredImageSemantic>;
  listImageSemanticsForDocument(
    documentId: string,
    pageNos?: number[] | null,
  ): StoredImageSemantic[];
  deleteImageSemanticsForDocument(documentId: string): number;
  updateImageSemantic(imageHash: string, semanticText: string, semanticModel?: string | null): void;
  getImageSemanticCache(
    imageHash: string,
    promptVersion: string,
  ): StoredImageSemanticCache | null;
  upsertImageSemanticCache(record: StorageImageSemanticCacheRecord): void;
  createDocumentParseTask(task: StorageDocumentParseTaskRecord): StoredDocumentParseTask;
  getDocumentParseTask(taskId: string): StoredDocumentParseTask | null;
  listDocumentParseTasks(input?: {
    status?: DocumentParseTaskStatus | null;
    taskType?: DocumentParseTaskType | null;
    documentId?: string | null;
    limit?: number;
    offset?: number;
  }): { items: StoredDocumentParseTask[]; total: number };
  updateDocumentParseTask(
    taskId: string,
    patch: {
      documentId?: string | null;
      status?: DocumentParseTaskStatus;
      progressPercent?: number;
      currentStage?: string | null;
      optionsJson?: string;
      stageTimingsJson?: string;
      errorMessage?: string | null;
      startedAt?: string | null;
      finishedAt?: string | null;
      totalDurationMs?: number | null;
    },
  ): StoredDocumentParseTask | null;
  deleteDocumentParseTask(taskId: string): boolean;
  deleteActiveDocumentParseTasks(): number;
  createCollection(name: string): PublicCollectionRecord;
  listCollections(includeDeleted?: boolean): PublicCollectionRecord[];
  getCollection(collectionId: string): PublicCollectionRecord | null;
  getActiveCollectionByName(name: string): PublicCollectionRecord | null;
  updateCollection(collectionId: string, name: string): PublicCollectionRecord | null;
  setCollectionDeleted(collectionId: string, isDeleted: boolean): PublicCollectionRecord | null;
  countCollectionDocuments(collectionId: string): number;
  listCollectionDocuments(collectionId: string, includeDeleted?: boolean): StoredDocument[];
  listDocumentCollections(docId: string, includeDeleted?: boolean): PublicCollectionRecord[];
  attachDocumentsToCollection(collectionId: string, documentIds: string[]): number;
  replaceCollectionDocuments(collectionId: string, documentIds: string[]): number;
  detachDocumentFromCollection(collectionId: string, docId: string): boolean;
  replaceDocumentCollections(docId: string, collectionIds: string[]): number;
  removeDocumentFromAllCollections(docId: string): number;
  createDocumentCatalog(corpusId: string): DocumentCatalog;
}

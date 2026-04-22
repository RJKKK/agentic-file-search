/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/agent.py
*/

import type { LoadedDocumentPage } from "./library.js";
import type { StoredDocument } from "./storage.js";

export interface LazyIndexingStats {
  triggered: boolean;
  indexed_documents: number;
  chunks_written: number;
  embeddings_written: number;
}

export interface IndexedSearchHit {
  doc_id: string;
  original_filename: string;
  relative_path: string;
  absolute_path: string;
  position: number;
  source_unit_no: number;
  text: string;
  semantic_score: number;
  metadata_score: number;
  score: number;
  matched_by: "page_content" | "page_content+metadata";
  heading?: string | null;
  match_count: number;
}

export interface IndexedSearchInput {
  query: string;
  filters?: string | null;
  limit?: number;
}

export interface IndexedSearchResult {
  query: string;
  document_ids: string[];
  collection_id: string | null;
  lazy_indexing: LazyIndexingStats;
  hits: IndexedSearchHit[];
}

export interface IndexStatusResult {
  indexed: boolean;
  corpus_id?: string;
  document_count?: number;
  schema_name?: string | null;
  has_metadata?: boolean;
  has_embeddings?: boolean;
  schema_fields?: string[];
}

export interface ResolvedPageScope {
  document: StoredDocument;
  pagesDir: string;
}

export interface PageSearchHit {
  doc_id: string;
  absolute_path: string;
  source_unit_no: number;
  score: number;
  text: string;
  match_count: number;
}

export interface PageSearchForTargetResult {
  document: StoredDocument;
  pagesDir: string;
  hits: PageSearchHit[];
  pages: LoadedDocumentPage[];
}

export interface IndexedDocumentReadResult {
  rendered: string;
  structured: {
    document_id: string;
    absolute_path: string;
    label: string;
    content: string;
  } | null;
}

export type RuntimeEventEmitter = (eventType: string, data: Record<string, unknown>) => void;

export interface IndexSearchServiceContract {
  scopeLabel(): string | null;
  listIndexedDocuments(): string;
  getDocument(docId: string): Promise<IndexedDocumentReadResult>;
  search(input: IndexedSearchInput): Promise<IndexedSearchResult>;
  renderSearchResult(result: IndexedSearchResult): string;
  resolveDocumentPageScope(target: string): ResolvedPageScope | null;
  searchPagesForTarget(target: string, pattern: string): Promise<PageSearchForTargetResult | null>;
  findPageByPath(filePath: string): Promise<{ document: StoredDocument; page: LoadedDocumentPage } | null>;
  emit(eventType: string, data: Record<string, unknown>): void;
}

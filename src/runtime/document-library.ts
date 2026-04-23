/*
Reference: legacy/python/src/fs_explorer/document_library.py
Reference: legacy/python/src/fs_explorer/server.py
*/

import { extname } from "node:path";

import type { DocumentCatalog, DocumentRecord, DocumentSummary } from "../types/skills.js";
import type { BlobStore, DocumentScope, DocumentSummaryPayload } from "../types/library.js";
import type { PublicCollectionRecord, SqliteStorageBackend, StoredDocument } from "../types/storage.js";
import { loadDocumentPages, resolvePagesDirectory } from "./document-pages.js";
import { buildDocumentPagesPrefix, buildDocumentSourceKey, validateStorageFilename } from "./page-store.js";

export const LIBRARY_CORPUS_ROOT = "blob://library/default";

function scopeResult(
  corpusId: string,
  documentIds: string[],
  documents: StoredDocument[],
  collections: PublicCollectionRecord[],
): DocumentScope {
  return {
    corpusId,
    documentIds,
    documents,
    collection: collections[0] ?? null,
    collections,
    isEmpty: documentIds.length === 0,
  };
}

function serializeSummary(document: StoredDocument, pageCount?: number): DocumentSummary {
  return {
    id: document.id,
    absolutePath: document.absolute_path,
    label: document.original_filename || document.relative_path,
    pageCount: pageCount ?? document.page_count,
  };
}

export function ensureLibraryCorpus(storage: SqliteStorageBackend): string {
  return storage.getOrCreateCorpus(LIBRARY_CORPUS_ROOT);
}

export function getLibraryCorpusId(
  storage: SqliteStorageBackend,
  input: { createIfMissing?: boolean } = {},
): string | null {
  if (input.createIfMissing) {
    return ensureLibraryCorpus(storage);
  }
  return storage.getCorpusId(LIBRARY_CORPUS_ROOT);
}

export function buildDocumentObjectKey(docId: string, filename: string): string {
  const validated = validateStorageFilename(filename);
  const extension = extname(validated) || ".bin";
  return buildDocumentSourceKey(docId, `source${extension}`);
}

export function buildDocumentPagesKeyPrefix(docId: string): string {
  return buildDocumentPagesPrefix(docId);
}

export async function materializeDocument(input: {
  storage: SqliteStorageBackend;
  blobStore: BlobStore;
  document: StoredDocument;
}): Promise<StoredDocument> {
  const objectKey = String(input.document.source_object_key || input.document.object_key || "").trim();
  if (!objectKey) {
    return input.document;
  }
  const materializedPath = await input.blobStore.materialize({ objectKey });
  if (input.document.absolute_path !== materializedPath) {
    const updated = input.storage.updateDocumentAbsolutePath(input.document.id, materializedPath);
    if (updated) {
      return updated;
    }
    return {
      ...input.document,
      absolute_path: materializedPath,
    };
  }
  return input.document;
}

export function resolveDocumentScope(input: {
  storage: SqliteStorageBackend;
  documentIds?: string[] | null;
  collectionId?: string | null;
  collectionIds?: string[] | null;
}): DocumentScope {
  const corpusId = getLibraryCorpusId(input.storage, { createIfMissing: true }) || "";
  const resolvedIds: string[] = [];
  const collections: PublicCollectionRecord[] = [];
  const collectionIds = [
    ...new Set(
      [
        ...(input.collectionIds ?? []),
        ...(input.collectionId ? [input.collectionId] : []),
      ]
        .map((item) => String(item).trim())
        .filter(Boolean),
    ),
  ];

  for (const collectionId of collectionIds) {
    const collection = input.storage.getCollection(collectionId);
    if (!collection || collection.is_deleted) {
      throw new Error("Collection not found.");
    }
    collections.push(collection);
    for (const document of input.storage.listCollectionDocuments(collectionId, false)) {
      if (!resolvedIds.includes(document.id)) {
        resolvedIds.push(document.id);
      }
    }
  }

  for (const docId of input.documentIds ?? []) {
    const normalized = String(docId).trim();
    if (normalized && !resolvedIds.includes(normalized)) {
      resolvedIds.push(normalized);
    }
  }

  const documents = input.storage.listDocumentsByIds(resolvedIds, false);
  const liveIds = documents.map((document) => document.id);
  return scopeResult(corpusId, liveIds, documents, collections);
}

export function serializeDocumentSummary(
  document: StoredDocument,
  input: { pageCount?: number } = {},
): DocumentSummaryPayload {
  const pageCount = input.pageCount ?? document.page_count ?? 0;
  const status = document.is_deleted ? "deleted" : document.upload_status || "uploaded";
  let metadata: Record<string, unknown> = {};
  try {
    metadata = JSON.parse(document.metadata_json || "{}") as Record<string, unknown>;
  } catch {
    metadata = {};
  }
  return {
    id: document.id,
    corpus_id: document.corpus_id,
    relative_path: document.relative_path,
    absolute_path: document.absolute_path,
    original_filename: document.original_filename || document.relative_path || "",
    object_key: document.object_key || "",
    source_object_key: document.source_object_key || document.object_key || "",
    pages_prefix: document.pages_prefix || "",
    storage_uri: document.storage_uri || "",
    content_type: document.content_type,
    upload_status: document.upload_status || "uploaded",
    page_count: pageCount,
    file_size: document.file_size || 0,
    file_mtime: document.file_mtime || 0,
    content_sha256: document.content_sha256 || "",
    parsed_content_sha256: document.parsed_content_sha256,
    parsed_is_complete: document.parsed_is_complete,
    is_deleted: document.is_deleted,
    status,
    metadata,
    last_indexed_at: document.last_indexed_at,
  };
}

export function createLibraryDocumentCatalog(input: {
  storage: SqliteStorageBackend;
  blobStore: BlobStore;
}): DocumentCatalog {
  return {
    listDocuments: async (): Promise<DocumentSummary[]> => {
      const corpusId = getLibraryCorpusId(input.storage, { createIfMissing: false });
      if (!corpusId) {
        return [];
      }
      return input.storage.listDocuments(corpusId, false).map((document) => ({
        ...serializeSummary(document),
        pagesDir: document.pages_prefix
          ? resolvePagesDirectory({ blobStore: input.blobStore, pagesPrefix: document.pages_prefix })
          : undefined,
      }));
    },
    getDocument: async (docId: string): Promise<DocumentRecord | null> => {
      const corpusId = getLibraryCorpusId(input.storage, { createIfMissing: false });
      if (!corpusId) {
        return null;
      }
      const document = input.storage.getDocument(docId);
      if (!document || document.is_deleted || document.corpus_id !== corpusId) {
        return null;
      }
      let content = document.content;
      if (!content.trim() && document.page_count > 0) {
        const pages = await loadDocumentPages({
          storage: input.storage,
          blobStore: input.blobStore,
          documentId: document.id,
        });
        content = pages.map((page) => page.markdown).filter((markdown) => markdown.trim()).join("\n\n");
      }
      return {
        id: document.id,
        absolutePath: document.absolute_path,
        label: document.original_filename || document.relative_path,
        pageCount: document.page_count,
        pagesDir: document.pages_prefix
          ? resolvePagesDirectory({ blobStore: input.blobStore, pagesPrefix: document.pages_prefix })
          : undefined,
        content,
      };
    },
  };
}

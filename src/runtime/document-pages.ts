/*
Reference: legacy/python/src/fs_explorer/document_pages.py
Reference: legacy/python/src/fs_explorer/page_store.py
*/

import { resolve } from "node:path";

import { LocalBlobStore, resolveObjectStoreDir } from "./blob-store.js";
import { parsePageFrontMatter } from "./page-store.js";
import type { BlobStore, LoadedDocumentPage, StoredPage } from "../types/library.js";
import type {
  SqliteStorageBackend,
  StorageDocumentPageRecord,
  StoredDocument,
  StoredDocumentPage,
} from "../types/storage.js";

export function pageRecordFromManifest(
  item: StoredPage | StoredDocumentPage,
  input: { documentId?: string | null } = {},
): StorageDocumentPageRecord {
  const documentId = input.documentId ?? ("document_id" in item ? item.document_id : null);
  if (!documentId) {
    throw new Error("document_id is required to build a page record.");
  }
  return {
    documentId,
    pageNo: "pageNo" in item ? item.pageNo : item.page_no,
    objectKey: "objectKey" in item ? item.objectKey : item.object_key,
    heading: item.heading ?? null,
    sourceLocator: "sourceLocator" in item ? item.sourceLocator ?? null : item.source_locator ?? null,
    contentHash: "contentHash" in item ? item.contentHash : item.content_hash,
    charCount: "charCount" in item ? item.charCount : item.char_count,
    isSyntheticPage:
      "isSyntheticPage" in item ? item.isSyntheticPage : item.is_synthetic_page,
  };
}

export function resolvePagesDirectory(input: {
  blobStore: BlobStore;
  pagesPrefix: string;
}): string {
  const prefix = String(input.pagesPrefix || "").trim().replace(/\/+$/, "");
  if (!prefix) {
    return "";
  }
  if (input.blobStore instanceof LocalBlobStore) {
    return resolve(input.blobStore.rootDir, prefix);
  }
  return resolve(resolveObjectStoreDir(), prefix);
}

export async function readPageContent(input: {
  blobStore: BlobStore;
  page: StoredDocumentPage;
}): Promise<LoadedDocumentPage> {
  const raw = Buffer.from(await input.blobStore.get({ objectKey: input.page.object_key })).toString("utf8");
  const [header, body] = parsePageFrontMatter(raw);
  return {
    unit_no: input.page.page_no,
    page_no: input.page.page_no,
    object_key: input.page.object_key,
    heading: input.page.heading ?? header.heading ?? null,
    source_locator: input.page.source_locator ?? header.source_locator ?? null,
    content_hash: input.page.content_hash,
    char_count: input.page.char_count || body.length,
    is_synthetic_page: input.page.is_synthetic_page,
    page_label: header.page_label ?? String(input.page.page_no),
    markdown: body,
    file_path: await input.blobStore.materialize({ objectKey: input.page.object_key }),
  };
}

export async function loadDocumentPages(input: {
  storage: SqliteStorageBackend;
  blobStore: BlobStore;
  documentId: string;
  pageNos?: number[] | null;
}): Promise<LoadedDocumentPage[]> {
  const pages = input.storage.listDocumentPages(input.documentId, input.pageNos ?? null);
  const loaded: LoadedDocumentPage[] = [];
  for (const page of pages) {
    loaded.push(await readPageContent({ blobStore: input.blobStore, page }));
  }
  return loaded;
}

export async function findPageByPath(input: {
  storage: SqliteStorageBackend;
  blobStore: BlobStore;
  documentIds: string[];
  filePath: string;
}): Promise<{ document: StoredDocument; page: LoadedDocumentPage } | null> {
  const normalized = resolve(input.filePath).toLowerCase();
  const documents = input.storage.listDocumentsByIds(input.documentIds, false);
  for (const document of documents) {
    const pages = input.storage.listDocumentPages(document.id);
    for (const page of pages) {
      const localPath = await input.blobStore.materialize({ objectKey: page.object_key });
      if (resolve(localPath).toLowerCase() === normalized) {
        return {
          document,
          page: await readPageContent({ blobStore: input.blobStore, page }),
        };
      }
    }
  }
  return null;
}

/*
Reference: legacy/python/src/fs_explorer/page_store.py
*/

import type { ParsedDocument, ParsedUnit } from "../types/parsing.js";
import type { BlobStore, StoredPage } from "../types/library.js";

const INVALID_STORAGE_FILENAME_CHARS = new Set('\\/:*?"<>|'.split(""));

export function validateStorageFilename(filename: string): string {
  const base = filename.split(/[\\/]/).pop() ?? "";
  if (!base || base === "." || base === "..") {
    throw new Error("Uploaded file must include a valid filename.");
  }
  if ([...base].some((char) => INVALID_STORAGE_FILENAME_CHARS.has(char))) {
    throw new Error(
      "Filename contains characters that cannot be used as an exact storage directory name.",
    );
  }
  return base;
}

export function buildDocumentPrefix(filename: string): string {
  return `documents/${validateStorageFilename(filename)}`;
}

export function buildDocumentSourceKey(filename: string): string {
  const validated = validateStorageFilename(filename);
  return `${buildDocumentPrefix(validated)}/source/${validated}`;
}

export function buildDocumentPagesPrefix(filename: string): string {
  return `${buildDocumentPrefix(filename)}/pages`;
}

export function buildDocumentPageKey(filename: string, pageNo: number): string {
  return `${buildDocumentPagesPrefix(filename)}/page-${String(Math.trunc(pageNo)).padStart(4, "0")}.md`;
}

export function renderPageMarkdown(input: {
  documentId: string;
  originalFilename: string;
  pageNo: number;
  pageLabel: string;
  contentType?: string | null;
  sourceLocator?: string | null;
  heading?: string | null;
  body: string;
}): string {
  void input.documentId;
  void input.originalFilename;
  void input.pageNo;
  void input.pageLabel;
  void input.contentType;
  void input.sourceLocator;
  void input.heading;
  const normalizedBody = String(input.body || "").trim();
  return normalizedBody ? `${normalizedBody}\n` : "";
}

export function stripPageFrontMatter(markdown: string): string {
  const text = String(markdown || "");
  if (!text.startsWith("---\n")) {
    return text;
  }
  const marker = text.indexOf("\n---\n", 4);
  if (marker === -1) {
    return text;
  }
  return text.slice(marker + 5).replace(/^\n+/, "");
}

export function parsePageFrontMatter(markdown: string): [Record<string, string>, string] {
  const text = String(markdown || "");
  if (!text.startsWith("---\n")) {
    return [{}, text];
  }
  const marker = text.indexOf("\n---\n", 4);
  if (marker === -1) {
    return [{}, text];
  }
  const header: Record<string, string> = {};
  for (const line of text.slice(4, marker).split("\n")) {
    if (!line.includes(":")) {
      continue;
    }
    const [rawKey, rawValue] = line.split(":", 2);
    header[rawKey.trim()] = rawValue.trim().replace(/^"|"$/g, "");
  }
  return [header, text.slice(marker + 5).replace(/^\n+/, "")];
}

async function persistPage(input: {
  blobStore: BlobStore;
  documentId: string;
  originalFilename: string;
  contentType?: string | null;
  unit: ParsedUnit;
  pageNo: number;
  syntheticPages: boolean;
}): Promise<StoredPage> {
  const objectKey = buildDocumentPageKey(input.originalFilename, input.pageNo);
  const pageLabel = input.syntheticPages ? `synthetic-${input.pageNo}` : String(input.pageNo);
  const payload = renderPageMarkdown({
    documentId: input.documentId,
    originalFilename: input.originalFilename,
    pageNo: input.pageNo,
    pageLabel,
    contentType: input.contentType ?? null,
    sourceLocator: input.unit.source_locator ?? null,
    heading: input.unit.heading ?? null,
    body: input.unit.markdown,
  });
  await input.blobStore.put({
    objectKey,
    data: Buffer.from(payload, "utf8"),
  });
  return {
    pageNo: input.pageNo,
    objectKey,
    heading: input.unit.heading ?? null,
    sourceLocator: input.unit.source_locator ?? null,
    contentHash: input.unit.content_hash,
    charCount: input.unit.markdown.length,
    isSyntheticPage: input.syntheticPages,
  };
}

export async function persistDocumentPages(input: {
  blobStore: BlobStore;
  documentId: string;
  originalFilename: string;
  contentType?: string | null;
  parsedDocument: ParsedDocument;
  syntheticPages: boolean;
}): Promise<StoredPage[]> {
  const pagesPrefix = buildDocumentPagesPrefix(input.originalFilename);
  await input.blobStore.deletePrefix({ prefix: pagesPrefix });
  const storedPages: StoredPage[] = [];
  const orderedUnits = [...input.parsedDocument.units].sort((left, right) => left.unit_no - right.unit_no);
  for (const [index, unit] of orderedUnits.entries()) {
    storedPages.push(
      await persistPage({
        blobStore: input.blobStore,
        documentId: input.documentId,
        originalFilename: input.originalFilename,
        contentType: input.contentType ?? null,
        unit,
        pageNo: index + 1,
        syntheticPages: input.syntheticPages,
      }),
    );
  }
  return storedPages;
}


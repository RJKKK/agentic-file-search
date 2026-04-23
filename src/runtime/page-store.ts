/*
Reference: legacy/python/src/fs_explorer/page_store.py
*/

import type { ParsedDocument, ParsedUnit } from "../types/parsing.js";
import type { BlobStore, StoredPage } from "../types/library.js";

export function validateStorageFilename(filename: string): string {
  const base = filename.split(/[\\/]/).pop() ?? "";
  if (!base || base === "." || base === "..") {
    throw new Error("Uploaded file must include a valid filename.");
  }
  return base;
}

function validateStorageKey(storageKey: string): string {
  const normalized = String(storageKey || "").trim().replace(/^[\\/]+|[\\/]+$/g, "");
  if (!normalized || normalized === "." || normalized === ".." || /[\\/]/.test(normalized)) {
    throw new Error("A valid storage key is required.");
  }
  return normalized;
}

export function buildDocumentPrefix(storageKey: string): string {
  return `documents/${validateStorageKey(storageKey)}`;
}

export function buildDocumentSourceKey(storageKey: string, filename = "source.bin"): string {
  const base = validateStorageFilename(filename);
  return `${buildDocumentPrefix(storageKey)}/source/${encodeURIComponent(base)}`;
}

export function buildDocumentPagesPrefix(storageKey: string): string {
  return `${buildDocumentPrefix(storageKey)}/pages`;
}

export function buildDocumentPageKey(storageKey: string, pageNo: number): string {
  return `${buildDocumentPagesPrefix(storageKey)}/page-${String(Math.trunc(pageNo)).padStart(4, "0")}.md`;
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
  const normalizedBody = sanitizeStoredPageMarkdown(String(input.body || "")).trim();
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

function splitMarkdownBlocks(markdown: string): string[] {
  return String(markdown || "")
    .replace(/\r\n/g, "\n")
    .trim()
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
}

export function sanitizeStoredPageMarkdown(markdown: string): string {
  return String(markdown || "")
    .split("\n")
    .map((line) => {
      if (!line.includes("|")) {
        return line;
      }
      return line.replace(/(?<=\|)\s*Col(?:\[\d+\]|\d+)\s*(?=\|)/gi, "");
    })
    .join("\n");
}

export function extractBoundaryBlocks(markdown: string): {
  leadingBlockMarkdown: string | null;
  trailingBlockMarkdown: string | null;
} {
  const blocks = splitMarkdownBlocks(markdown);
  if (blocks.length === 0) {
    return {
      leadingBlockMarkdown: null,
      trailingBlockMarkdown: null,
    };
  }
  return {
    leadingBlockMarkdown: blocks[0] ? sanitizeStoredPageMarkdown(blocks[0]) : null,
    trailingBlockMarkdown: blocks.at(-1) ? sanitizeStoredPageMarkdown(blocks.at(-1)!) : null,
  };
}

async function persistPage(input: {
  blobStore: BlobStore;
  storageKey: string;
  documentId: string;
  originalFilename: string;
  contentType?: string | null;
  unit: ParsedUnit;
  pageNo: number;
  syntheticPages: boolean;
}): Promise<StoredPage> {
  const objectKey = buildDocumentPageKey(input.storageKey, input.pageNo);
  const pageLabel = input.syntheticPages ? `synthetic-${input.pageNo}` : String(input.pageNo);
  const sanitizedMarkdown = sanitizeStoredPageMarkdown(input.unit.markdown);
  const payload = renderPageMarkdown({
    documentId: input.documentId,
    originalFilename: input.originalFilename,
    pageNo: input.pageNo,
    pageLabel,
    contentType: input.contentType ?? null,
    sourceLocator: input.unit.source_locator ?? null,
    heading: input.unit.heading ?? null,
    body: sanitizedMarkdown,
  });
  const boundaryBlocks = extractBoundaryBlocks(sanitizedMarkdown);
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
    charCount: sanitizedMarkdown.length,
    isSyntheticPage: input.syntheticPages,
    leadingBlockMarkdown: boundaryBlocks.leadingBlockMarkdown,
    trailingBlockMarkdown: boundaryBlocks.trailingBlockMarkdown,
  };
}

export async function persistDocumentPages(input: {
  blobStore: BlobStore;
  storageKey: string;
  documentId: string;
  originalFilename: string;
  contentType?: string | null;
  parsedDocument: ParsedDocument;
  syntheticPages: boolean;
}): Promise<StoredPage[]> {
  const pagesPrefix = buildDocumentPagesPrefix(input.storageKey);
  await input.blobStore.deletePrefix({ prefix: pagesPrefix });
  const storedPages: StoredPage[] = [];
  const orderedUnits = [...input.parsedDocument.units].sort((left, right) => left.unit_no - right.unit_no);
  for (const [index, unit] of orderedUnits.entries()) {
    storedPages.push(
      await persistPage({
        blobStore: input.blobStore,
        storageKey: input.storageKey,
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

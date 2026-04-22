/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/document_pages.py
Reference: legacy/python/src/fs_explorer/document_library.py
*/

import { resolve } from "node:path";

import type { BlobStore, DocumentScope, LoadedDocumentPage } from "../types/library.js";
import type {
  IndexedDocumentReadResult,
  IndexedSearchHit,
  IndexedSearchInput,
  IndexedSearchResult,
  IndexSearchServiceContract,
  IndexStatusResult,
  LazyIndexingStats,
  PageBatchReadGroup,
  PageBatchReadInput,
  PageScopeSummary,
  PageSearchForTargetResult,
  PageSearchHit,
  ResolvedPageScope,
  RuntimeEventEmitter,
} from "../types/search.js";
import type { PublicCollectionRecord, SqliteStorageBackend, StoredDocument } from "../types/storage.js";
import { LIBRARY_CORPUS_ROOT } from "./document-library.js";
import { findPageByPath, loadDocumentPages, resolvePagesDirectory } from "./document-pages.js";

const EMPTY_LAZY_INDEXING: LazyIndexingStats = {
  triggered: false,
  indexed_documents: 0,
  chunks_written: 0,
  embeddings_written: 0,
};

function cleanExcerpt(text: string, maxChars = 320): string {
  const squashed = text.replace(/\s+/g, " ").trim();
  if (squashed.length <= maxChars) {
    return squashed;
  }
  return `${squashed.slice(0, maxChars)}...`;
}

export function searchTerms(query: string): string[] {
  const normalized = String(query || "").split(/\s+/).join(" ").trim();
  if (!normalized) {
    return [];
  }
  const terms = [normalized.toLowerCase()];
  for (const match of normalized.toLowerCase().matchAll(/[\w\u4e00-\u9fff]{2,}/g)) {
    const token = match[0];
    if (!terms.includes(token)) {
      terms.push(token);
    }
    if (terms.length >= 8) {
      break;
    }
  }
  return terms;
}

export function buildSearchSnippet(input: {
  markdown: string;
  matchStart: number | null;
  window?: number;
}): string {
  const body = String(input.markdown || "").trim();
  const window = input.window ?? 180;
  if (!body) {
    return "";
  }
  if (input.matchStart == null) {
    return body.slice(0, window * 2).replace(/\s+/g, " ").trim();
  }
  const start = Math.max(input.matchStart - window, 0);
  const end = Math.min(input.matchStart + window, body.length);
  const prefix = start > 0 ? "..." : "";
  const suffix = end < body.length ? "..." : "";
  return `${prefix}${body.slice(start, end)}${suffix}`.replace(/\s+/g, " ").trim();
}

export function pageMatchesQuery(input: {
  query: string;
  markdown: string;
  heading?: string | null;
}): { score: number; matchCount: number; matchStart: number | null } {
  const terms = searchTerms(input.query);
  if (terms.length === 0) {
    return { score: 0, matchCount: 0, matchStart: null };
  }
  const body = String(input.markdown || "");
  const bodyLower = body.toLowerCase();
  const headingLower = String(input.heading || "").toLowerCase();
  let score = 0;
  let matchCount = 0;
  let firstMatchStart: number | null = null;
  terms.forEach((term, index) => {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const bodyMatches = [...bodyLower.matchAll(new RegExp(escaped, "g"))];
    const headingMatches = [...headingLower.matchAll(new RegExp(escaped, "g"))];
    if (bodyMatches.length === 0 && headingMatches.length === 0) {
      return;
    }
    matchCount += bodyMatches.length + headingMatches.length;
    const weight = index === 0 ? 12 : 3;
    score += bodyMatches.length * weight;
    score += headingMatches.length * (weight + 2);
    if (firstMatchStart == null && bodyMatches[0]?.index != null) {
      firstMatchStart = bodyMatches[0].index;
    }
  });
  return { score, matchCount, matchStart: firstMatchStart };
}

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

function normalizeTarget(value: string): string {
  return resolve(String(value || "")).toLowerCase();
}

function formatImageSemanticCache(input: {
  storage: SqliteStorageBackend;
  documentId: string;
}): string {
  const imageRows = input.storage.listImageSemanticsForDocument(input.documentId);
  if (imageRows.length === 0) {
    return "";
  }
  const pageHeadings = Object.fromEntries(
    input.storage
      .listDocumentPages(input.documentId)
      .map((page) => [page.page_no, String(page.heading || "").trim()]),
  );
  const grouped = new Map<number, typeof imageRows>();
  for (const image of imageRows) {
    const pageNo = image.source_page_no;
    grouped.set(pageNo, [...(grouped.get(pageNo) ?? []), image]);
  }
  const lines: string[] = [];
  for (const pageNo of [...grouped.keys()].sort((left, right) => left - right)) {
    const entries = (grouped.get(pageNo) ?? [])
      .filter((image) => image.semantic_text)
      .map((image) => `- image ${image.source_image_index}: ${image.semantic_text}`);
    if (entries.length === 0) {
      continue;
    }
    const heading = pageHeadings[pageNo];
    lines.push(`Page ${pageNo}${heading ? ` (${heading})` : ""}:`);
    lines.push(...entries);
  }
  return lines.length ? `\n\nImage semantics cache:\n${lines.join("\n")}` : "";
}

export interface IndexSearchServiceOptions {
  storage: SqliteStorageBackend;
  blobStore: BlobStore;
  rootPath?: string | null;
  documentIds?: string[] | null;
  collectionId?: string | null;
  collectionIds?: string[] | null;
  scopeLabel?: string | null;
  emitRuntimeEvent?: RuntimeEventEmitter | null;
}

export class IndexSearchService implements IndexSearchServiceContract {
  private readonly normalizedDocumentIds: string[];

  constructor(private readonly options: IndexSearchServiceOptions) {
    this.normalizedDocumentIds = [
      ...new Set((options.documentIds ?? []).map((item) => String(item).trim()).filter(Boolean)),
    ];
  }

  scopeLabel(): string | null {
    if (this.options.scopeLabel) {
      return this.options.scopeLabel;
    }
    const scope = this.resolveScope();
    if (scope.collections.length > 0) {
      return scope.collections.length === 1
        ? scope.collections[0]!.name
        : `${scope.collections.length} selected collections`;
    }
    if (scope.documentIds.length > 0) {
      return `${scope.documentIds.length} selected documents`;
    }
    return null;
  }

  emit(eventType: string, data: Record<string, unknown>): void {
    this.options.emitRuntimeEvent?.(eventType, data);
  }

  getIndexStatus(): IndexStatusResult {
    const corpusRoot = this.options.rootPath || LIBRARY_CORPUS_ROOT;
    const corpusId = this.options.storage.getCorpusId(corpusRoot);
    if (!corpusId) {
      return { indexed: false };
    }
    const documents = this.options.storage.listDocuments(corpusId, false);
    return {
      indexed: true,
      corpus_id: corpusId,
      document_count: documents.length,
      schema_name: null,
      has_metadata: false,
      has_embeddings: false,
      schema_fields: [],
    };
  }

  resolveScope(): DocumentScope {
    const collectionIds = [
      ...new Set(
        [
          ...(this.options.collectionIds ?? []),
          ...(this.options.collectionId ? [this.options.collectionId] : []),
        ]
          .map((item) => String(item).trim())
          .filter(Boolean),
      ),
    ];
    if (collectionIds.length > 0) {
      const collections: PublicCollectionRecord[] = [];
      const resolvedIds: string[] = [];
      for (const collectionId of collectionIds) {
        const collection = this.options.storage.getCollection(collectionId);
        if (!collection || collection.is_deleted) {
          throw new Error("Collection not found.");
        }
        collections.push(collection);
        for (const document of this.options.storage.listCollectionDocuments(collectionId, false)) {
          if (!resolvedIds.includes(document.id)) {
            resolvedIds.push(document.id);
          }
        }
      }
      for (const docId of this.normalizedDocumentIds) {
        if (!resolvedIds.includes(docId)) {
          resolvedIds.push(docId);
        }
      }
      const documents = this.options.storage.listDocumentsByIds(resolvedIds, false);
      return scopeResult(
        documents[0]?.corpus_id ?? "",
        documents.map((document) => document.id),
        documents,
        collections,
      );
    }

    if (this.normalizedDocumentIds.length > 0) {
      const documents = this.options.storage.listDocumentsByIds(this.normalizedDocumentIds, false);
      return scopeResult(
        documents[0]?.corpus_id ?? "",
        documents.map((document) => document.id),
        documents,
        [],
      );
    }

    const corpusRoot = this.options.rootPath || LIBRARY_CORPUS_ROOT;
    const corpusId = this.options.storage.getCorpusId(corpusRoot) ?? "";
    if (!corpusId) {
      return scopeResult("", [], [], []);
    }
    const documents = this.options.storage.listDocuments(corpusId, false);
    return scopeResult(corpusId, documents.map((document) => document.id), documents, []);
  }

  listIndexedDocuments(): string {
    const scope = this.resolveScope();
    if (scope.isEmpty) {
      return "No indexed documents found for the active corpus.";
    }
    const lines = ["=== INDEXED DOCUMENTS ==="];
    const label = this.scopeLabel();
    if (label) {
      lines.push(`Scope: ${label}`);
    }
    scope.documents.forEach((document, index) => {
      const pagesDir = resolvePagesDirectory({
        blobStore: this.options.blobStore,
        pagesPrefix: document.pages_prefix || "",
      });
      lines.push(
        `[${index + 1}] doc_id=${document.id} source=${document.absolute_path} ` +
          `pages_dir=${pagesDir} page_count=${document.page_count || 0} ` +
          `name=${document.original_filename || document.relative_path}`,
      );
    });
    lines.push("");
    lines.push("Use glob/grep/read on the pages_dir to answer questions page-by-page.");
    return lines.join("\n");
  }

  async listPageScopes(): Promise<PageScopeSummary[]> {
    const summaries: PageScopeSummary[] = [];
    for (const document of this.resolveScope().documents) {
      const pagesDir = document.pages_prefix
        ? resolvePagesDirectory({
            blobStore: this.options.blobStore,
            pagesPrefix: document.pages_prefix,
          })
        : "";
      let pageRange: PageScopeSummary["page_range"] = null;
      try {
        const pages = await loadDocumentPages({
          storage: this.options.storage,
          blobStore: this.options.blobStore,
          documentId: document.id,
        });
        if (pages.length > 0) {
          const pageNos = pages.map((page) => page.page_no);
          pageRange = { start: Math.min(...pageNos), end: Math.max(...pageNos) };
        }
      } catch {
        pageRange = null;
      }
      summaries.push({
        doc_id: document.id,
        filename: document.original_filename || document.relative_path || document.id,
        source_path: document.absolute_path,
        pages_dir: pagesDir,
        page_count: document.page_count || 0,
        page_range: pageRange,
      });
    }
    summaries.sort((left, right) =>
      left.filename.toLowerCase().localeCompare(right.filename.toLowerCase()) ||
      left.doc_id.localeCompare(right.doc_id),
    );
    return summaries;
  }

  async getDocument(docId: string): Promise<IndexedDocumentReadResult> {
    const document = this.options.storage.getDocument(docId);
    if (!document) {
      return {
        rendered: `No indexed document found for doc_id=${JSON.stringify(docId)}`,
        structured: null,
      };
    }
    const scope = this.resolveScope();
    if (scope.documentIds.length > 0 && !scope.documentIds.includes(docId)) {
      return {
        rendered: `Document ${docId} is outside the currently selected document scope.`,
        structured: null,
      };
    }
    if (document.is_deleted) {
      return {
        rendered: `Document ${docId} is marked as deleted in the index.`,
        structured: null,
      };
    }
    let documentBody = document.content || "";
    if (!documentBody.trim()) {
      const pages = await loadDocumentPages({
        storage: this.options.storage,
        blobStore: this.options.blobStore,
        documentId: docId,
      });
      documentBody = pages
        .map((page) => page.markdown.trim())
        .filter(Boolean)
        .join("\n\n");
    }
    const semanticSection = formatImageSemanticCache({
      storage: this.options.storage,
      documentId: docId,
    });
    const rendered = [
      `=== DOCUMENT ${docId} ===`,
      `Path: ${document.absolute_path}`,
      "",
      `${documentBody}${semanticSection}`,
    ].join("\n");
    return {
      rendered,
      structured: {
        document_id: docId,
        absolute_path: document.absolute_path,
        label: document.original_filename || document.relative_path || docId,
        content: documentBody,
      },
    };
  }

  async search(input: IndexedSearchInput): Promise<IndexedSearchResult> {
    const query = String(input.query || "").trim();
    if (!query) {
      throw new Error("Query must not be empty.");
    }
    if (input.filters && input.filters.trim()) {
      throw new Error("Metadata filters are not implemented in the Node phase-1 index search.");
    }
    const scope = this.resolveScope();
    if (scope.isEmpty) {
      throw new Error("At least one document or one collection must be selected.");
    }
    const hits: IndexedSearchHit[] = [];
    for (const document of scope.documents) {
      const originalFilename = document.original_filename || document.relative_path || document.id;
      const pages = await loadDocumentPages({
        storage: this.options.storage,
        blobStore: this.options.blobStore,
        documentId: document.id,
      });
      for (const page of pages) {
        const match = pageMatchesQuery({
          query,
          markdown: page.markdown,
          heading: page.heading,
        });
        if (match.score <= 0) {
          continue;
        }
        hits.push({
          doc_id: document.id,
          original_filename: originalFilename,
          relative_path: document.relative_path || originalFilename,
          absolute_path: page.file_path || document.absolute_path || "",
          position: page.page_no,
          source_unit_no: page.page_no,
          text: buildSearchSnippet({
            markdown: page.markdown,
            matchStart: match.matchStart,
          }),
          semantic_score: Number(match.score.toFixed(4)),
          metadata_score: 0,
          score: Number(match.score.toFixed(4)),
          matched_by: "page_content",
          heading: page.heading,
          match_count: match.matchCount,
        });
      }
    }
    hits.sort(
      (left, right) =>
        right.score - left.score ||
        right.match_count - left.match_count ||
        left.original_filename.toLowerCase().localeCompare(right.original_filename.toLowerCase()) ||
        left.source_unit_no - right.source_unit_no,
    );
    const limit = Math.max(Number(input.limit ?? 5), 1);
    return {
      query,
      document_ids: [...scope.documentIds],
      collection_id: scope.collections[0]?.id ?? this.options.collectionId ?? null,
      collection_ids: scope.collections.map((collection) => collection.id),
      lazy_indexing: { ...EMPTY_LAZY_INDEXING },
      hits: hits.slice(0, limit),
    };
  }

  renderSearchResult(result: IndexedSearchResult): string {
    if (result.hits.length === 0) {
      return `No indexed matches found for query: ${JSON.stringify(result.query)}`;
    }
    const lines = ["=== INDEXED SEARCH RESULTS ===", `Query: ${result.query}`, ""];
    result.hits.forEach((hit, index) => {
      lines.push(`[${index + 1}] doc_id: ${hit.doc_id}`);
      lines.push(`    path: ${hit.absolute_path}`);
      lines.push(`    match: ${hit.matched_by}`);
      lines.push(`    chunk_position: ${hit.position}`);
      lines.push(`    source_unit_no: ${hit.source_unit_no}`);
      lines.push(`    semantic_score: ${hit.semantic_score}`);
      lines.push(`    metadata_score: ${hit.metadata_score}`);
      lines.push(`    score: ${hit.score.toFixed(2)}`);
      lines.push(`    excerpt: ${cleanExcerpt(hit.text)}`);
      lines.push("");
    });
    lines.push("Use get_document(doc_id=...) to read full content for the most relevant documents.");
    return lines.join("\n");
  }

  resolveDocumentPageScope(target: string): ResolvedPageScope | null {
    const normalizedTarget = normalizeTarget(target);
    for (const document of this.resolveScope().documents) {
      if (!document.pages_prefix) {
        continue;
      }
      const pagesDir = resolvePagesDirectory({
        blobStore: this.options.blobStore,
        pagesPrefix: document.pages_prefix,
      });
      const sourcePath = normalizeTarget(document.absolute_path);
      if (normalizedTarget === sourcePath || normalizedTarget === normalizeTarget(pagesDir)) {
        return { document, pagesDir };
      }
    }
    return null;
  }

  async searchPagesForTarget(
    target: string,
    pattern: string,
  ): Promise<PageSearchForTargetResult | null> {
    const resolved = this.resolveDocumentPageScope(target);
    if (!resolved) {
      return null;
    }
    let regex: RegExp;
    try {
      regex = new RegExp(pattern, "gim");
    } catch (error) {
      throw new Error(`Invalid regex pattern ${pattern}: ${String(error)}`);
    }
    const pages = await loadDocumentPages({
      storage: this.options.storage,
      blobStore: this.options.blobStore,
      documentId: resolved.document.id,
    });
    const hits: PageSearchHit[] = [];
    for (const page of pages) {
      const matches = [...page.markdown.matchAll(regex)];
      if (matches.length === 0) {
        continue;
      }
      const matchIndex = matches[0]?.index ?? 0;
      const start = Math.max(0, matchIndex - 40);
      const end = Math.min(page.markdown.length, matchIndex + 180);
      hits.push({
        doc_id: resolved.document.id,
        absolute_path: page.file_path,
        source_unit_no: page.page_no,
        score: matches.length,
        text: page.markdown.slice(start, end),
        match_count: matches.length,
      });
    }
    hits.sort((left, right) => right.score - left.score || left.source_unit_no - right.source_unit_no);
    return {
      ...resolved,
      hits: hits.slice(0, 8),
      pages,
    };
  }

  async searchPagesAcrossScope(
    query: string,
    options: { maxHitsPerDocument?: number; maxTotalHits?: number; regex?: boolean } = {},
  ): Promise<PageSearchForTargetResult> {
    const trimmedQuery = String(query || "").trim();
    if (!trimmedQuery) {
      throw new Error("Query must not be empty.");
    }
    const scope = this.resolveScope();
    if (scope.isEmpty) {
      throw new Error("At least one document or one collection must be selected.");
    }
    const maxHitsPerDocument = Math.max(Number(options.maxHitsPerDocument ?? 5), 1);
    const maxTotalHits = Math.max(Number(options.maxTotalHits ?? 24), 1);
    let regex: RegExp | null = null;
    if (options.regex) {
      try {
        regex = new RegExp(trimmedQuery, "gim");
      } catch (error) {
        throw new Error(`Invalid regex pattern ${trimmedQuery}: ${String(error)}`);
      }
    }

    const hits: PageSearchHit[] = [];
    const allPages: LoadedDocumentPage[] = [];
    for (const document of scope.documents) {
      const pages = await loadDocumentPages({
        storage: this.options.storage,
        blobStore: this.options.blobStore,
        documentId: document.id,
      });
      allPages.push(...pages);
      const docHits: PageSearchHit[] = [];
      for (const page of pages) {
        let score = 0;
        let matchCount = 0;
        let matchStart: number | null = null;
        if (regex) {
          const matches = [...page.markdown.matchAll(regex)];
          matchCount = matches.length;
          score = matches.length;
          matchStart = matches[0]?.index ?? null;
        } else {
          const match = pageMatchesQuery({
            query: trimmedQuery,
            markdown: page.markdown,
            heading: page.heading,
          });
          score = match.score;
          matchCount = match.matchCount;
          matchStart = match.matchStart;
        }
        if (score <= 0) {
          continue;
        }
        docHits.push({
          doc_id: document.id,
          absolute_path: page.file_path,
          source_unit_no: page.page_no,
          score,
          text: buildSearchSnippet({
            markdown: page.markdown,
            matchStart,
          }),
          match_count: matchCount,
        });
      }
      docHits.sort(
        (left, right) =>
          right.score - left.score ||
          right.match_count - left.match_count ||
          left.source_unit_no - right.source_unit_no,
      );
      hits.push(...docHits.slice(0, maxHitsPerDocument));
    }

    hits.sort(
      (left, right) =>
        right.score - left.score ||
        right.match_count - left.match_count ||
        left.doc_id.localeCompare(right.doc_id) ||
        left.source_unit_no - right.source_unit_no,
    );
    return {
      hits: hits.slice(0, maxTotalHits),
      pages: allPages,
    };
  }

  async resolvePageBatch(input: PageBatchReadInput): Promise<PageBatchReadGroup[]> {
    const maxPages = Math.max(Number(input.maxPages ?? 5), 1);
    const maxChars = Math.max(Number(input.maxChars ?? 8_000), 1);
    const groups = new Map<string, { document: StoredDocument; pages: LoadedDocumentPage[] }>();

    const addPage = (document: StoredDocument, page: LoadedDocumentPage): void => {
      const existing = groups.get(document.id) ?? { document, pages: [] };
      if (!existing.pages.some((item) => item.page_no === page.page_no)) {
        existing.pages.push(page);
      }
      groups.set(document.id, existing);
    };

    const filePaths = [...new Set((input.filePaths ?? []).map((item) => String(item).trim()).filter(Boolean))];
    for (const filePath of filePaths) {
      const resolvedPage = await this.findPageByPath(filePath);
      if (resolvedPage) {
        addPage(resolvedPage.document, resolvedPage.page);
      }
    }

    const documentId = String(input.documentId ?? "").trim();
    if (documentId) {
      const scope = this.resolveScope();
      if (!scope.documentIds.includes(documentId)) {
        throw new Error(`Document ${documentId} is outside the currently selected document scope.`);
      }
      const document = scope.documents.find((item) => item.id === documentId);
      if (!document) {
        throw new Error(`Document ${documentId} was not found.`);
      }
      const pages = await loadDocumentPages({
        storage: this.options.storage,
        blobStore: this.options.blobStore,
        documentId,
      });
      const startPage = Math.max(Number(input.startPage ?? input.endPage ?? 1), 1);
      const endPage = Math.max(Number(input.endPage ?? startPage), startPage);
      for (const page of pages) {
        if (page.page_no >= startPage && page.page_no <= endPage) {
          addPage(document, page);
        }
      }
    }

    const orderedGroups = [...groups.values()].sort((left, right) =>
      (left.document.original_filename || left.document.relative_path || left.document.id)
        .toLowerCase()
        .localeCompare((right.document.original_filename || right.document.relative_path || right.document.id).toLowerCase()) ||
      left.document.id.localeCompare(right.document.id),
    );

    const limitedGroups: PageBatchReadGroup[] = [];
    let pagesUsed = 0;
    let charsUsed = 0;
    for (const group of orderedGroups) {
      const sortedPages = [...group.pages].sort((left, right) => left.page_no - right.page_no);
      const accepted: LoadedDocumentPage[] = [];
      const omittedPages: number[] = [];
      for (const page of sortedPages) {
        const nextChars = charsUsed + page.markdown.length;
        if (pagesUsed >= maxPages || (accepted.length > 0 && nextChars > maxChars)) {
          omittedPages.push(page.page_no);
          continue;
        }
        if (nextChars > maxChars && pagesUsed > 0) {
          omittedPages.push(page.page_no);
          continue;
        }
        accepted.push(page);
        pagesUsed += 1;
        charsUsed = nextChars;
      }
      if (accepted.length > 0 || omittedPages.length > 0) {
        limitedGroups.push({
          document: group.document,
          pages: accepted,
          truncated: omittedPages.length > 0,
          omittedPages,
        });
      }
    }
    return limitedGroups;
  }

  async findPageByPath(
    filePath: string,
  ): Promise<{ document: StoredDocument; page: LoadedDocumentPage } | null> {
    const scope = this.resolveScope();
    return findPageByPath({
      storage: this.options.storage,
      blobStore: this.options.blobStore,
      documentIds: scope.documentIds,
      filePath,
    });
  }
}

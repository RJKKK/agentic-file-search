/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
*/

import { createHash } from "node:crypto";

import {
  extractChineseFtsQueryTokens,
  extractPlainQueryTokens,
} from "./chinese-fts.js";
import { resolveEmbeddingConfig } from "./model-config.js";
import {
  protectImageLinks,
  restoreProtectedImageLinks,
} from "./image-semantic.js";
import type {
  TraditionalRagCerContextRef,
  TraditionalRetrievalMode,
} from "../types/rag.js";
import type {
  StoredFixedRetrievalChunk,
  StoredRetrievalChunk,
} from "../types/storage.js";

export const DEFAULT_SMALL_CHUNK_MAX_CHARS = 399;
export const DEFAULT_NORMAL_CHUNK_MAX_CHARS = 2000;
export const DEFAULT_EVIDENCE_CHAR_BUDGET = 12000;
export const CER_CONTEXT_EXCERPT_TARGET = 600;
export const IMAGE_PROMPT_VERSION = "v3";
export const DEFAULT_ZHIPU_EMBEDDING_3_MAX_TOKENS = 3072;
export const DEFAULT_OPENAI_EMBEDDING_MAX_TOKENS = 8191;
export const DEFAULT_EMBEDDING_CHAR_BUDGET_RATIO = 2;
export const DEFAULT_EMBEDDING_IMAGE_NEIGHBOR_MAX_CHARS = 1600;

export interface ChunkThresholds {
  smallMaxChars: number;
  normalMaxChars: number;
}

export interface SourceChunkRecord {
  id: string;
  documentId: string;
  pageNo: number;
  documentIndex: number;
  pageIndex: number;
  blockType: string;
  bboxJson: string;
  contentMd: string;
  sizeClass: "small" | "normal" | "oversized";
  summaryText: string | null;
  mergedPageNosJson: string;
  mergedBboxesJson: string;
  previousContextMd: string | null;
}

export interface ImageChunkRendering {
  dropped: boolean;
  markdown: string;
}

export interface RankedChunkHit {
  retrievalUnitId: string;
  documentId: string;
  sourceDocumentChunkId: string;
  referenceRetrievalChunkId: string | null;
  ordinal: number;
  score: number;
  unitText: string;
  isSplitFromOversized: boolean;
}

export type TraditionalRetrievalChunkRecord =
  | StoredRetrievalChunk
  | StoredFixedRetrievalChunk;

export interface GroupedEvidence {
  citationNo: number;
  referenceId: string;
  referenceKind: "retrieval_chunk" | "document_chunk";
  documentId: string;
  documentName: string;
  sourceLocator: string | null;
  hasOversizedSplitChild: boolean;
  documentChunkIds: string[];
  retrievalUnitIds: string[];
  pageNos: number[];
  bboxes: Array<[number, number, number, number]>;
  score: number;
  summaryText: string | null;
  sourceLink: string;
  mergedExcerpt: string;
  baseContent: string;
  enrichedContent: string;
  content: string;
  cerApplied: boolean;
  cerContextRefs: TraditionalRagCerContextRef[];
  cerSegments: EvidenceSegment[];
  compressionApplied: boolean;
}

export interface EvidenceSegment {
  citationNo: number;
  referenceId: string;
  referenceKind: "retrieval_chunk" | "document_chunk";
  summaryText: string | null;
  mergedExcerpt: string;
  sourceLink: string;
  sourceLocator: string | null;
  pageNos: number[];
  sortOrder: number;
  blockType: string | null;
  isOversized: boolean;
  isCenter: boolean;
}

export interface PreparedEvidenceSet {
  mode: TraditionalRetrievalMode;
  question: string;
  warnings: string[];
  chunks: GroupedEvidence[];
}

export function stableId(prefix: string, value: string): string {
  return `${prefix}_${createHash("sha1").update(value, "utf8").digest("hex")}`;
}

export function stableUuid(value: string): string {
  const hex = createHash("sha1")
    .update(value, "utf8")
    .digest("hex")
    .slice(0, 32)
    .split("");
  hex[12] = "5";
  hex[16] = ((Number.parseInt(hex[16] || "0", 16) & 0x3) | 0x8).toString(16);
  return `${hex.slice(0, 8).join("")}-${hex.slice(8, 12).join("")}-${hex.slice(12, 16).join("")}-${hex.slice(16, 20).join("")}-${hex.slice(20, 32).join("")}`;
}

export function qdrantPointIdForRetrievalUnit(retrievalUnitId: string): string {
  return stableUuid(`qdrant:${retrievalUnitId}`);
}

export function parsePositiveIntegerEnv(name: string, fallback: number): number {
  const raw = String(process.env[name] ?? "").trim();
  if (!raw) {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    console.warn(
      `[traditional-rag] Invalid ${name}=${JSON.stringify(raw)}; using ${fallback}.`,
    );
    return fallback;
  }
  return parsed;
}

export function getChunkThresholds(): ChunkThresholds {
  const smallMaxChars = parsePositiveIntegerEnv(
    "TRADITIONAL_RAG_SMALL_CHUNK_MAX_CHARS",
    DEFAULT_SMALL_CHUNK_MAX_CHARS,
  );
  const normalMaxChars = parsePositiveIntegerEnv(
    "TRADITIONAL_RAG_NORMAL_CHUNK_MAX_CHARS",
    DEFAULT_NORMAL_CHUNK_MAX_CHARS,
  );
  if (normalMaxChars <= smallMaxChars) {
    console.warn(
      `[traditional-rag] Invalid thresholds: normal (${normalMaxChars}) must be greater than small (${smallMaxChars}); using defaults.`,
    );
    return {
      smallMaxChars: DEFAULT_SMALL_CHUNK_MAX_CHARS,
      normalMaxChars: DEFAULT_NORMAL_CHUNK_MAX_CHARS,
    };
  }
  return { smallMaxChars, normalMaxChars };
}

export function getEvidenceCharBudget(): number {
  return parsePositiveIntegerEnv(
    "TRADITIONAL_RAG_EVIDENCE_CHAR_BUDGET",
    DEFAULT_EVIDENCE_CHAR_BUDGET,
  );
}

export function classifySize(text: string): "small" | "normal" | "oversized" {
  const thresholds = getChunkThresholds();
  const length = String(text || "").length;
  if (length <= thresholds.smallMaxChars) {
    return "small";
  }
  if (length <= thresholds.normalMaxChars) {
    return "normal";
  }
  return "oversized";
}

export function normalizeWhitespace(value: string): string {
  return String(value || "")
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function mergeContinuedBlockMarkdown(
  left: string,
  right: string,
  blockType: string,
): string {
  if (blockType !== "table") {
    return normalizeWhitespace(`${left}\n\n${right}`);
  }
  const normalizedLeft = String(left || "").replace(/\r\n/g, "\n");
  const normalizedRight = String(right || "").replace(/\r\n/g, "\n");
  const leftTrimmed = normalizedLeft.replace(/(?:\n[ \t]*)+$/g, "");
  const rightTrimmed = normalizedRight.replace(/^(?:[ \t]*\n)+/g, "");
  if (!leftTrimmed.trim()) {
    return rightTrimmed.trim();
  }
  if (!rightTrimmed.trim()) {
    return leftTrimmed.trim();
  }
  return `${leftTrimmed}\n${rightTrimmed}`.trim();
}

export function toFtsQuery(value: string): string {
  const tokens = extractChineseFtsQueryTokens(value);
  if (!tokens?.length) {
    return String(value || "")
      .replace(/["']/g, " ")
      .trim();
  }
  return [...new Set(tokens)].map((token) => `"${token}"`).join(" OR ");
}

export function parseJsonArray<T>(raw: string, fallback: T[] = []): T[] {
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as T[]) : fallback;
  } catch {
    return fallback;
  }
}

export function bboxKey(bbox: [number, number, number, number]): string {
  return JSON.stringify(bbox.map((value) => Number(value.toFixed(3))));
}

export function extractQueryTokens(value: string): string[] {
  return extractPlainQueryTokens(value);
}

export function parsePositiveFloatEnv(name: string, fallback: number): number {
  const raw = Number.parseFloat(String(process.env[name] ?? "").trim());
  return Number.isFinite(raw) && raw > 0 ? raw : fallback;
}

export function embeddingTokenBudget(
  modelName: string = resolveEmbeddingConfig().modelName,
): number {
  const envBudget = parsePositiveIntegerEnv("EMBEDDING_MODEL_MAX_TOKENS", 0);
  if (envBudget > 0) {
    return envBudget;
  }
  const normalized = String(modelName || "").trim().toLowerCase();
  if (normalized === "embedding-3" || normalized.startsWith("embedding-3-")) {
    return DEFAULT_ZHIPU_EMBEDDING_3_MAX_TOKENS;
  }
  if (
    normalized === "text-embedding-3-small" ||
    normalized === "text-embedding-3-large" ||
    normalized === "text-embedding-ada-002"
  ) {
    return DEFAULT_OPENAI_EMBEDDING_MAX_TOKENS;
  }
  return DEFAULT_ZHIPU_EMBEDDING_3_MAX_TOKENS;
}

export function embeddingCharBudget(
  modelName: string = resolveEmbeddingConfig().modelName,
): number {
  const ratio = parsePositiveFloatEnv(
    "EMBEDDING_MODEL_CHAR_BUDGET_RATIO",
    DEFAULT_EMBEDDING_CHAR_BUDGET_RATIO,
  );
  return Math.max(Math.floor(embeddingTokenBudget(modelName) * ratio), 256);
}

export function normalizeBaseUrl(baseUrl: string | null): string {
  return (baseUrl?.trim() || "https://api.openai.com/v1").replace(/\/+$/, "");
}

export function resolveServerBaseUrl(): string {
  const configured = String(process.env.FS_EXPLORER_SERVER_BASE_URL ?? "").trim();
  if (configured) {
    return configured.replace(/\/+$/, "");
  }
  const port = parsePositiveIntegerEnv("FS_EXPLORER_PORT", 8000);
  return `http://localhost:${port}`;
}

export function toAbsoluteServerUrl(pathOrUrl: string): string {
  const value = String(pathOrUrl || "").trim();
  if (!value) {
    return value;
  }
  if (/^https?:\/\//i.test(value)) {
    return value;
  }
  return `${resolveServerBaseUrl()}${value.startsWith("/") ? value : `/${value}`}`;
}

export function resolveTextRequestTimeoutMs(): number {
  const raw = Number.parseInt(process.env.FS_EXPLORER_TEXT_TIMEOUT_MS || "", 10);
  if (Number.isFinite(raw) && raw > 0) {
    return raw;
  }
  return 60_000;
}

export function restorePresentProtectedImageLinks(
  value: string,
  protectedBlocks: Array<{ placeholder: string; text: string }>,
): string {
  let restored = String(value || "");
  for (const block of protectedBlocks) {
    if (!restored.includes(block.placeholder)) {
      continue;
    }
    restored = restored.split(block.placeholder).join(block.text);
  }
  return normalizeWhitespace(restored);
}

export function adjustSplitEndForProtectedImageLinks(
  text: string,
  proposedEnd: number,
): number {
  const placeholderPattern = /@@IMG_LINK_\d+@@/g;
  for (const match of text.matchAll(placeholderPattern)) {
    const start = match.index ?? -1;
    if (start < 0) {
      continue;
    }
    const end = start + match[0].length;
    if (proposedEnd <= start || proposedEnd >= end) {
      continue;
    }
    return start === 0 ? end : start;
  }
  return proposedEnd;
}

export function compressTextPreservingImageLinks(
  text: string,
  compressor: (value: string) => string,
): string {
  const normalized = normalizeWhitespace(text);
  if (!normalized) {
    return normalized;
  }
  const protectedResult = protectImageLinks(normalized);
  if (protectedResult.protectedBlocks.length === 0) {
    return normalizeWhitespace(compressor(normalized));
  }
  return restoreProtectedImageLinks(
    compressor(protectedResult.placeholderText),
    protectedResult.protectedBlocks,
  );
}

export function truncatePlainTextSmart(value: string, maxChars: number): string {
  const normalized = normalizeWhitespace(value);
  if (!normalized || normalized.length <= maxChars) {
    return normalized;
  }
  const paragraphs = normalized
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (paragraphs.length > 1) {
    const kept: string[] = [];
    let length = 0;
    for (const paragraph of paragraphs) {
      const candidateLength = length === 0 ? paragraph.length : length + 2 + paragraph.length;
      if (candidateLength > maxChars) {
        break;
      }
      kept.push(paragraph);
      length = candidateLength;
    }
    if (kept.length > 0) {
      return kept.join("\n\n");
    }
  }
  const lines = normalized
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length > 1) {
    const kept: string[] = [];
    let length = 0;
    for (const line of lines) {
      const candidateLength = length === 0 ? line.length : length + 1 + line.length;
      if (candidateLength > maxChars) {
        break;
      }
      kept.push(line);
      length = candidateLength;
    }
    if (kept.length > 0) {
      return kept.join("\n");
    }
  }
  const slice = normalized.slice(0, maxChars);
  const breakAt = Math.max(
    slice.lastIndexOf("\n\n"),
    slice.lastIndexOf("\n"),
    slice.lastIndexOf("。"),
    slice.lastIndexOf("？"),
    slice.lastIndexOf("！"),
    slice.lastIndexOf(". "),
    slice.lastIndexOf(" "),
  );
  const end = breakAt >= Math.floor(maxChars * 0.55) ? breakAt : maxChars;
  return slice.slice(0, end).trim();
}

export function truncateTextPreservingImageLinks(
  text: string,
  maxChars: number,
): string {
  const normalized = normalizeWhitespace(text);
  if (!normalized || normalized.length <= maxChars) {
    return normalized;
  }
  const protectedResult = protectImageLinks(normalized);
  if (protectedResult.protectedBlocks.length === 0) {
    return truncatePlainTextSmart(normalized, maxChars);
  }
  const linkText = normalizeWhitespace(
    protectedResult.protectedBlocks.map((block) => block.text).join("\n"),
  );
  const remainingText = normalizeWhitespace(
    protectedResult.placeholderText.replace(/@@IMG_LINK_\d+@@/g, ""),
  );
  if (!remainingText) {
    return linkText;
  }
  const separatorLength = linkText ? 2 : 0;
  const remainingBudget = Math.max(maxChars - linkText.length - separatorLength, 0);
  const truncatedRemainder =
    remainingBudget > 0 ? truncatePlainTextSmart(remainingText, remainingBudget) : "";
  return normalizeWhitespace([linkText, truncatedRemainder].filter(Boolean).join("\n\n"));
}

export function imageSemanticCacheVersion(contextText: string | null): string {
  const contextHash = contextText
    ? createHash("sha1")
        .update(normalizeWhitespace(contextText))
        .digest("hex")
        .slice(0, 12)
    : "noctx";
  return `${IMAGE_PROMPT_VERSION}:${contextHash}`;
}

export function normalizeEmbeddingNeighborText(
  value: string,
  maxChars: number,
): string {
  return truncatePlainTextSmart(value, Math.max(maxChars, 0));
}

export function uniqueOrdered<T>(values: T[]): T[] {
  return [...new Set(values)];
}

export function sourceLocatorFromPages(pageNos: number[]): string | null {
  const normalized = uniqueOrdered(
    pageNos
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value > 0),
  );
  if (normalized.length === 0) {
    return null;
  }
  if (normalized.length === 1) {
    return `page-${normalized[0]}`;
  }
  return `pages-${normalized.join("-")}`;
}

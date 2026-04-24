/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
*/

import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { extname } from "node:path";

import { resolveDocumentScope } from "./document-library.js";
import { resolveEmbeddingConfig, resolveTextConfig, resolveVisionConfig } from "./model-config.js";
import { PythonDocumentAssetBridge } from "./python-document-assets.js";
import type { BlobStore } from "../types/library.js";
import type { ParsedBlock, ParsedDocument, ParsedImage, ParsedUnit } from "../types/parsing.js";
import type { TraditionalRagQueryResult, TraditionalRetrievalMode } from "../types/rag.js";
import type {
  SqliteStorageBackend,
  StorageDocumentChunkRecord,
  StorageImageSemanticRecord,
  StorageRetrievalChunkRecord,
  StoredDocument,
  StoredDocumentChunk,
  StoredImageSemantic,
  StoredRetrievalChunk,
} from "../types/storage.js";

const DEFAULT_SMALL_CHUNK_MAX_CHARS = 399;
const DEFAULT_NORMAL_CHUNK_MAX_CHARS = 2000;
const DEFAULT_EVIDENCE_CHAR_BUDGET = 12000;
const MAX_IMAGE_BYTES_FOR_VISION = 4 * 1024 * 1024;
const IMAGE_PROMPT_VERSION = "v1";

interface ChunkThresholds {
  smallMaxChars: number;
  normalMaxChars: number;
}

interface ImageSemanticPayload {
  recognizable: boolean;
  image_kind?: string | null;
  contains_text?: boolean | null;
  visible_text?: string | null;
  summary?: string | null;
  entities?: string[];
  keywords?: string[];
  qa_hints?: string[];
  drop_reason?: string | null;
}

export interface ImageChunkRendering {
  dropped: boolean;
  markdown: string;
}

interface RankedChunkHit {
  retrievalChunkId: string;
  documentId: string;
  sourceDocumentChunkId: string;
  ordinal: number;
  score: number;
  chunkText: string;
  isSplitFromOversized: boolean;
}

interface GroupedEvidence {
  documentChunkId: string;
  retrievalChunkIds: string[];
  pageNos: number[];
  bboxes: Array<[number, number, number, number]>;
  score: number;
  summaryText: string | null;
  sourceLink: string;
  mergedExcerpt: string;
  content: string;
  compressionApplied: boolean;
}

function stableId(prefix: string, value: string): string {
  return `${prefix}_${createHash("sha1").update(value, "utf8").digest("hex")}`;
}

function stableUuid(value: string): string {
  const hex = createHash("sha1").update(value, "utf8").digest("hex").slice(0, 32).split("");
  hex[12] = "5";
  hex[16] = ((Number.parseInt(hex[16] || "0", 16) & 0x3) | 0x8).toString(16);
  return `${hex.slice(0, 8).join("")}-${hex.slice(8, 12).join("")}-${hex.slice(12, 16).join("")}-${hex.slice(16, 20).join("")}-${hex.slice(20, 32).join("")}`;
}

function qdrantPointIdForRetrievalChunk(retrievalChunkId: string): string {
  return stableUuid(`qdrant:${retrievalChunkId}`);
}

function parsePositiveIntegerEnv(name: string, fallback: number): number {
  const raw = String(process.env[name] ?? "").trim();
  if (!raw) {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    console.warn(`[traditional-rag] Invalid ${name}=${JSON.stringify(raw)}; using ${fallback}.`);
    return fallback;
  }
  return parsed;
}

function getChunkThresholds(): ChunkThresholds {
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

function getEvidenceCharBudget(): number {
  return parsePositiveIntegerEnv(
    "TRADITIONAL_RAG_EVIDENCE_CHAR_BUDGET",
    DEFAULT_EVIDENCE_CHAR_BUDGET,
  );
}

function classifySize(text: string): "small" | "normal" | "oversized" {
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

function normalizeWhitespace(value: string): string {
  return String(value || "").replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

function toFtsQuery(value: string): string {
  const tokens = extractQueryTokens(value);
  if (!tokens?.length) {
    return String(value || "").replace(/["']/g, " ").trim();
  }
  return [...new Set(tokens)].map((token) => `"${token}"`).join(" OR ");
}

function parseJsonArray<T>(raw: string, fallback: T[] = []): T[] {
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as T[]) : fallback;
  } catch {
    return fallback;
  }
}

function bboxKey(bbox: [number, number, number, number]): string {
  return JSON.stringify(bbox.map((value) => Number(value.toFixed(3))));
}

function extractQueryTokens(value: string): string[] {
  return [...new Set(String(value || "").toLowerCase().match(/[\u4e00-\u9fff]{1,}|[a-z0-9_]{2,}/g) ?? [])];
}

async function fetchJsonWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs = 60_000,
): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(`HTTP ${response.status} ${body}`.trim());
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

function qdrantConfig(): { url: string | null; apiKey: string | null } {
  return {
    url: process.env.QDRANT_URL?.trim() || process.env.FS_EXPLORER_QDRANT_URL?.trim() || null,
    apiKey: process.env.QDRANT_API_KEY?.trim() || process.env.FS_EXPLORER_QDRANT_API_KEY?.trim() || null,
  };
}

async function createEmbedding(text: string): Promise<number[] | null> {
  const config = resolveEmbeddingConfig();
  if (!config.apiKey) {
    return null;
  }
  const baseUrl = (config.baseUrl?.trim() || "https://api.openai.com/v1").replace(/\/+$/, "");
  const payload = await fetchJsonWithTimeout(
    `${baseUrl}/embeddings`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify({
        model: config.modelName,
        input: text,
      }),
    },
    60_000,
  );
  const vector = (payload as { data?: Array<{ embedding?: number[] }> }).data?.[0]?.embedding;
  if (!Array.isArray(vector) || vector.length === 0) {
    return null;
  }
  const normalized = vector.map((value) => Number(value));
  if (normalized.some((value) => !Number.isFinite(value))) {
    console.warn("[traditional-rag] Embedding provider returned non-finite values; discarding vector.");
    return null;
  }
  if (!normalized.some((value) => Math.abs(value) > 1e-12)) {
    console.warn("[traditional-rag] Embedding provider returned an all-zero vector; discarding vector.");
    return null;
  }
  return normalized;
}

async function describeImageWithVision(input: {
  bytes: Uint8Array;
  mimeType: string;
}): Promise<ImageSemanticPayload | null> {
  const config = resolveVisionConfig();
  if (!config.apiKey) {
    return null;
  }
  const baseUrl = (config.baseUrl?.trim() || "https://api.openai.com/v1").replace(/\/+$/, "");
  const dataUrl = `data:${input.mimeType};base64,${Buffer.from(input.bytes).toString("base64")}`;
  const payload = await fetchJsonWithTimeout(
    `${baseUrl}/chat/completions`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify({
        model: config.modelName,
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [
          {
            role: "system",
            content:
              "Return strict JSON with fields recognizable, image_kind, contains_text, visible_text, summary, entities, keywords, qa_hints, drop_reason.",
          },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Analyze this document image for retrieval. Keep summary concise and factual.",
              },
              {
                type: "image_url",
                image_url: { url: dataUrl },
              },
            ],
          },
        ],
      }),
    },
    90_000,
  );
  const text =
    (payload as { choices?: Array<{ message?: { content?: string | Array<{ text?: string }> } }> }).choices?.[0]
      ?.message?.content ?? "";
  const raw = Array.isArray(text) ? text.map((item) => item.text ?? "").join("") : String(text || "");
  try {
    return JSON.parse(raw) as ImageSemanticPayload;
  } catch {
    return null;
  }
}

async function summarizeAnswer(input: {
  question: string;
  hits: Array<{ documentChunkId: string; score: number; content: string }>;
}): Promise<{ answer: string; used_chunks: Array<{ document_chunk_id: string; score: number }> } | null> {
  const config = resolveTextConfig();
  if (!config.apiKey) {
    return null;
  }
  const baseUrl = (config.baseUrl?.trim() || "https://api.openai.com/v1").replace(/\/+$/, "");
  const evidence = input.hits
    .map(
      (hit, index) =>
        `[${index + 1}] chunk=${hit.documentChunkId} score=${hit.score.toFixed(4)}\n${hit.content}`,
    )
    .join("\n\n");
  const payload = await fetchJsonWithTimeout(
    `${baseUrl}/chat/completions`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify({
        model: config.modelName,
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [
          {
            role: "system",
            content:
              "Answer only from evidence. Return JSON with answer and used_chunks. used_chunks must be an array of objects with document_chunk_id and score.",
          },
          {
            role: "user",
            content: `Question:\n${input.question}\n\nEvidence:\n${evidence}`,
          },
        ],
      }),
    },
    90_000,
  );
  const text =
    (payload as { choices?: Array<{ message?: { content?: string | Array<{ text?: string }> } }> }).choices?.[0]
      ?.message?.content ?? "";
  const raw = Array.isArray(text) ? text.map((item) => item.text ?? "").join("") : String(text || "");
  try {
    const parsed = JSON.parse(raw) as {
      answer?: string;
      used_chunks?: Array<{ document_chunk_id?: string; score?: number }>;
    };
    return {
      answer: String(parsed.answer || ""),
      used_chunks: (parsed.used_chunks ?? [])
        .filter((item) => item.document_chunk_id)
        .map((item) => ({
          document_chunk_id: String(item.document_chunk_id),
          score: Number(item.score ?? 0),
        })),
    };
  } catch {
    return null;
  }
}

async function ensureQdrantCollection(documentId: string, vectorSize: number): Promise<boolean> {
  const config = qdrantConfig();
  if (!config.url) {
    return false;
  }
  const response = await fetch(`${config.url.replace(/\/+$/, "")}/collections/doc_${documentId}`, {
    method: "PUT",
    headers: {
      "content-type": "application/json",
      ...(config.apiKey ? { "api-key": config.apiKey } : {}),
    },
    body: JSON.stringify({
      vectors: {
        size: vectorSize,
        distance: "Cosine",
      },
    }),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Failed to create Qdrant collection doc_${documentId}: HTTP ${response.status} ${body}`.trim());
  }
  return true;
}

async function deleteQdrantCollection(documentId: string): Promise<void> {
  const config = qdrantConfig();
  if (!config.url) {
    return;
  }
  await fetch(`${config.url.replace(/\/+$/, "")}/collections/doc_${documentId}`, {
    method: "DELETE",
    headers: {
      ...(config.apiKey ? { "api-key": config.apiKey } : {}),
    },
  }).catch(() => undefined);
}

async function upsertQdrantPoints(
  documentId: string,
  points: Array<{ retrievalChunkId: string; vector: number[] }>,
): Promise<number> {
  const config = qdrantConfig();
  if (!config.url || points.length === 0) {
    return 0;
  }
  await fetchJsonWithTimeout(
    `${config.url.replace(/\/+$/, "")}/collections/doc_${documentId}/points?wait=true`,
    {
      method: "PUT",
      headers: {
        "content-type": "application/json",
        ...(config.apiKey ? { "api-key": config.apiKey } : {}),
      },
      body: JSON.stringify({
        points: points.map((point) => ({
          id: qdrantPointIdForRetrievalChunk(point.retrievalChunkId),
          vector: point.vector,
          payload: { retrieval_chunk_id: point.retrievalChunkId },
        })),
      }),
    },
    60_000,
  );
  return points.length;
}

async function rebuildQdrantCollection(
  documentId: string,
  chunks: Array<{ id: string; text: string }>,
): Promise<number> {
  const points: Array<{ retrievalChunkId: string; vector: number[] }> = [];
  for (const chunk of chunks) {
    const vector = await createEmbedding(chunk.text);
    if (!vector) {
      continue;
    }
    points.push({ retrievalChunkId: chunk.id, vector });
  }
  await deleteQdrantCollection(documentId);
  if (points.length > 0) {
    await ensureQdrantCollection(documentId, points[0]!.vector.length);
    return upsertQdrantPoints(documentId, points);
  }
  return 0;
}

async function searchQdrant(documentId: string, vector: number[], limit: number): Promise<Array<{ id: string; score: number }>> {
  const config = qdrantConfig();
  if (!config.url) {
    return [];
  }
  const payload = await fetchJsonWithTimeout(
    `${config.url.replace(/\/+$/, "")}/collections/doc_${documentId}/points/search`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(config.apiKey ? { "api-key": config.apiKey } : {}),
      },
      body: JSON.stringify({
        vector,
        limit,
        with_payload: true,
      }),
    },
    60_000,
  ).catch(() => null);
  const result = (payload as { result?: Array<{ id?: string; score?: number; payload?: { retrieval_chunk_id?: string } }> } | null)
    ?.result;
  return (result ?? []).map((item) => ({
    id: String(item.payload?.retrieval_chunk_id ?? item.id ?? ""),
    score: Number(item.score ?? 0),
  })).filter((item) => item.id);
}

function pageDominatedByType(unit: ParsedUnit, blockType: string): boolean {
  if (!unit.blocks.length) {
    return false;
  }
  const normalized = unit.blocks.filter((block) => normalizeWhitespace(block.markdown));
  if (!normalized.length) {
    return false;
  }
  const matching = normalized.filter((block) => block.block_type === blockType);
  if (!matching.length) {
    return false;
  }
  const matchedChars = matching.reduce((sum, block) => sum + block.markdown.length, 0);
  const totalChars = normalized.reduce((sum, block) => sum + block.markdown.length, 0);
  return matchedChars / Math.max(totalChars, 1) >= 0.7;
}

function ruleSummary(content: string, blockType: string): string {
  const text = normalizeWhitespace(content);
  if (!text) {
    return "";
  }
  if (blockType === "table") {
    const lines = text.split("\n").filter(Boolean).slice(0, 12);
    return lines.join("\n").slice(0, 500);
  }
  const sentences = text.split(/(?<=[。！？.!?])\s+/).filter(Boolean);
  if (sentences.length <= 3) {
    return text.slice(0, 500);
  }
  const wordScores = new Map<string, number>();
  for (const token of text.toLowerCase().match(/[\u4e00-\u9fff]{1,}|[a-z0-9_]{2,}/g) ?? []) {
    wordScores.set(token, (wordScores.get(token) ?? 0) + 1);
  }
  const middle = sentences.slice(1, -1)
    .map((sentence) => ({
      sentence,
      score: (sentence.toLowerCase().match(/[\u4e00-\u9fff]{1,}|[a-z0-9_]{2,}/g) ?? []).reduce(
        (sum, token) => sum + (wordScores.get(token) ?? 0),
        0,
      ),
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 3)
    .map((item) => item.sentence);
  const ordered = [sentences[0], ...middle, sentences[sentences.length - 1]].join(" ");
  return ordered.slice(0, 500);
}

function splitOversized(content: string, blockType: string): string[] {
  const thresholds = getChunkThresholds();
  const text = normalizeWhitespace(content);
  if (text.length <= thresholds.normalMaxChars) {
    return [text];
  }
  const units =
    blockType === "table"
      ? text.split("\n").filter(Boolean)
      : text.split(/\n{2,}|(?<=[。！？.!?])\s+/).filter(Boolean);
  const parts: string[] = [];
  let current = "";
  for (const piece of units) {
    const candidate = current ? `${current}\n\n${piece}` : piece;
    if (candidate.length <= thresholds.normalMaxChars) {
      current = candidate;
      continue;
    }
    if (current) {
      parts.push(current);
    }
    if (piece.length <= thresholds.normalMaxChars) {
      current = piece;
      continue;
    }
    for (let index = 0; index < piece.length; index += thresholds.normalMaxChars) {
      parts.push(piece.slice(index, index + thresholds.normalMaxChars));
    }
    current = "";
  }
  if (current) {
    parts.push(current);
  }
  return parts;
}

function uniqueOrdered<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function composeEvidenceContent(input: {
  summaryText: string | null;
  mergedExcerpt: string;
  sourceLink: string;
}): string {
  return [input.summaryText, input.mergedExcerpt, `Source: ${input.sourceLink}`]
    .filter((value) => String(value || "").trim())
    .join("\n\n");
}

function fallbackCompressExcerpt(text: string, targetLength: number): string {
  const normalized = normalizeWhitespace(text);
  if (normalized.length <= targetLength) {
    return normalized;
  }
  if (targetLength <= 12) {
    return normalized.slice(0, Math.max(targetLength, 0));
  }
  const head = Math.max(Math.floor(targetLength * 0.35), 4);
  const tail = Math.max(targetLength - head - 3, 4);
  return `${normalized.slice(0, head)}...${normalized.slice(Math.max(normalized.length - tail, head))}`;
}

function compressExcerptAroundKeywords(text: string, queryTokens: string[], targetLength: number): string {
  const normalized = normalizeWhitespace(text);
  if (normalized.length <= targetLength) {
    return normalized;
  }
  if (targetLength <= 24) {
    return fallbackCompressExcerpt(normalized, targetLength);
  }
  const lowered = normalized.toLowerCase();
  const anchorIndexes = queryTokens
    .map((token) => lowered.indexOf(token))
    .filter((index) => index >= 0)
    .sort((left, right) => left - right);
  if (anchorIndexes.length === 0) {
    return fallbackCompressExcerpt(normalized, targetLength);
  }
  const center = Math.round(anchorIndexes.reduce((sum, index) => sum + index, 0) / anchorIndexes.length);
  const focusLength = Math.max(Math.floor(targetLength * 0.55), Math.min(240, targetLength));
  const focusStart = Math.max(0, Math.min(center - Math.floor(focusLength / 2), normalized.length - focusLength));
  const focusEnd = Math.min(normalized.length, focusStart + focusLength);
  const focus = normalized.slice(focusStart, focusEnd).trim();
  let remaining = targetLength - focus.length;
  if (remaining <= 3) {
    return focus.slice(0, targetLength);
  }
  const leftRoom = Math.max(Math.floor(remaining * 0.45), 0);
  const rightRoom = Math.max(remaining - leftRoom - 6, 0);
  const leftPart = focusStart > 0 ? fallbackCompressExcerpt(normalized.slice(0, focusStart), leftRoom).trim() : "";
  const rightPart =
    focusEnd < normalized.length
      ? fallbackCompressExcerpt(normalized.slice(focusEnd), rightRoom).trim()
      : "";
  const parts = [
    leftPart ? `${leftPart}...` : "",
    focus,
    rightPart ? `...${rightPart}` : "",
  ].filter(Boolean);
  const result = parts.join("\n");
  return result.length <= targetLength ? result : fallbackCompressExcerpt(result, targetLength);
}

function compressEvidenceItems(
  items: GroupedEvidence[],
  question: string,
  budget: number,
): GroupedEvidence[] {
  const queryTokens = extractQueryTokens(question);
  const materialized = items.map((item) => ({
    ...item,
    content: composeEvidenceContent(item),
  }));
  const totalLength = materialized.reduce((sum, item) => sum + item.content.length, 0);
  if (totalLength <= budget) {
    return materialized;
  }
  const lockedLength = materialized.reduce(
    (sum, item) => sum + composeEvidenceContent({ summaryText: item.summaryText, mergedExcerpt: "", sourceLink: item.sourceLink }).length,
    0,
  );
  const availableForExcerpts = Math.max(budget - lockedLength, 0);
  const totalExcerptLength = materialized.reduce((sum, item) => sum + item.mergedExcerpt.length, 0);
  const compressed = materialized.map((item) => {
    if (!item.mergedExcerpt) {
      return item;
    }
    const share = totalExcerptLength > 0 ? item.mergedExcerpt.length / totalExcerptLength : 1 / Math.max(materialized.length, 1);
    const targetLength = Math.max(Math.floor(availableForExcerpts * share), 80);
    if (item.mergedExcerpt.length <= targetLength) {
      return item;
    }
    const mergedExcerpt = compressExcerptAroundKeywords(item.mergedExcerpt, queryTokens, targetLength);
    return {
      ...item,
      mergedExcerpt,
      compressionApplied: true,
      content: composeEvidenceContent({
        summaryText: item.summaryText,
        mergedExcerpt,
        sourceLink: item.sourceLink,
      }),
    };
  });
  let running = compressed.reduce((sum, item) => sum + item.content.length, 0);
  if (running <= budget) {
    return compressed;
  }
  const sortedByScore = [...compressed].sort((left, right) => left.score - right.score);
  const dropped = new Set<string>();
  for (const item of sortedByScore) {
    if (running <= budget) {
      break;
    }
    dropped.add(item.documentChunkId);
    running -= item.content.length;
  }
  return compressed.filter((item) => !dropped.has(item.documentChunkId));
}

export class TraditionalRagService {
  private readonly pythonAssets = new PythonDocumentAssetBridge();

  constructor(
    private readonly storage: SqliteStorageBackend,
    private readonly blobStore: BlobStore,
  ) {}

  async indexDocument(input: {
    documentId: string;
    filePath: string;
    originalFilename: string;
    parsedDocument: ParsedDocument;
    enableEmbedding?: boolean;
    enableImageSemantic?: boolean;
  }): Promise<{ pageChunkCounts: Map<number, number>; chunksWritten: number; retrievalChunksWritten: number; embeddingsWritten: number }> {
    const imageRenderMap = await this.processImages({
      ...input,
      enableImageSemantic: input.enableImageSemantic !== false,
    });
    const documentChunks = this.buildDocumentChunks({
      documentId: input.documentId,
      parsedDocument: input.parsedDocument,
      imageRenderMap,
    });
    const retrievalChunks = this.buildRetrievalChunks(input.documentId, documentChunks);
    this.storage.replaceDocumentChunks(input.documentId, documentChunks);
    this.storage.replaceRetrievalChunks(input.documentId, retrievalChunks);

    let embeddingsWritten = 0;
    if (input.enableEmbedding !== false) {
      embeddingsWritten = await rebuildQdrantCollection(
        input.documentId,
        retrievalChunks.map((chunk) => ({ id: chunk.id, text: chunk.chunkText })),
      );
    } else {
      await deleteQdrantCollection(input.documentId);
    }

    const pageChunkCounts = new Map<number, number>();
    for (const chunk of documentChunks) {
      pageChunkCounts.set(chunk.pageNo, (pageChunkCounts.get(chunk.pageNo) ?? 0) + 1);
    }
    return {
      pageChunkCounts,
      chunksWritten: documentChunks.length,
      retrievalChunksWritten: retrievalChunks.length,
      embeddingsWritten,
    };
  }

  async buildEmbeddingsForDocument(documentId: string): Promise<number> {
    const retrievalChunks = this.storage.listRetrievalChunks(documentId);
    return this.buildEmbeddingsForChunks(
      documentId,
      retrievalChunks.map((chunk) => ({
        id: chunk.id,
        documentId: chunk.document_id,
        sourceDocumentChunkId: chunk.source_document_chunk_id,
        ordinal: chunk.ordinal,
        chunkText: chunk.chunk_text,
        sizeClass: chunk.size_class,
        isSplitFromOversized: chunk.is_split_from_oversized,
      })),
    );
  }

  async buildEmbeddingsForChunks(
    documentId: string,
    retrievalChunks: Array<{
      id: string;
      chunkText: string;
    }>,
  ): Promise<number> {
    return rebuildQdrantCollection(
      documentId,
      retrievalChunks.map((chunk) => ({ id: chunk.id, text: chunk.chunkText })),
    );
  }

  async renderImages(input: {
    documentId: string;
    filePath: string;
    originalFilename: string;
    parsedDocument: ParsedDocument;
    enableImageSemantic?: boolean;
  }): Promise<Map<string, ImageChunkRendering>> {
    return this.processImages({
      ...input,
      enableImageSemantic: input.enableImageSemantic !== false,
    });
  }

  createDocumentChunks(input: {
    documentId: string;
    parsedDocument: ParsedDocument;
    imageRenderMap: Map<string, ImageChunkRendering>;
  }): StorageDocumentChunkRecord[] {
    return this.buildDocumentChunks(input);
  }

  createRetrievalChunks(
    documentId: string,
    documentChunks: StorageDocumentChunkRecord[],
  ): StorageRetrievalChunkRecord[] {
    return this.buildRetrievalChunks(documentId, documentChunks);
  }

  async deleteDocumentIndex(documentId: string): Promise<void> {
    await deleteQdrantCollection(documentId);
  }

  async query(input: {
    question: string;
    mode: TraditionalRetrievalMode;
    documentIds?: string[] | null;
    collectionIds?: string[] | null;
    keywordWeight?: number;
    semanticWeight?: number;
  }): Promise<TraditionalRagQueryResult> {
    const scope = resolveDocumentScope({
      storage: this.storage,
      documentIds: input.documentIds ?? null,
      collectionIds: input.collectionIds ?? null,
    });
    if (scope.documentIds.length === 0) {
      throw new Error("At least one document or collection must be selected.");
    }
    const question = String(input.question || "").trim();
    if (!question) {
      throw new Error("Question must not be empty.");
    }
    const warnings: string[] = [];
    const documentsWithEmbeddings = scope.documents.filter((document) => document.has_embeddings);
    const documentIdsWithEmbeddings = new Set(documentsWithEmbeddings.map((document) => document.id));
    const missingEmbeddingDocuments = scope.documents.filter((document) => !document.has_embeddings);
    if (input.mode === "semantic" && documentsWithEmbeddings.length === 0) {
      throw new Error("Selected documents do not have embeddings yet.");
    }
    if (missingEmbeddingDocuments.length > 0 && input.mode !== "keyword") {
      warnings.push(
        `Skipped semantic retrieval for ${missingEmbeddingDocuments.length} document(s) without embeddings.`,
      );
    }

    const keywordHits =
      input.mode === "semantic"
        ? []
        : this.storage.keywordSearchRetrievalChunks({
            query: toFtsQuery(question),
            documentIds: scope.documentIds,
            limit: 24,
          }).map((item) => ({
            retrievalChunkId: item.retrieval_chunk_id,
            documentId: item.document_id,
            sourceDocumentChunkId: item.source_document_chunk_id,
            ordinal: item.ordinal,
            score: item.score,
            chunkText: item.chunk_text,
            isSplitFromOversized: item.is_split_from_oversized,
          }));

    const semanticHits: RankedChunkHit[] = [];
    if (input.mode !== "keyword" && documentsWithEmbeddings.length > 0) {
      const vector = await createEmbedding(question);
      if (vector) {
        for (const documentId of scope.documentIds) {
          if (!documentIdsWithEmbeddings.has(documentId)) {
            continue;
          }
          const matches = await searchQdrant(documentId, vector, 8);
          const retrievalMap = new Map(
            this.storage.listRetrievalChunks(documentId).map((chunk) => [chunk.id, chunk] as const),
          );
          for (const match of matches) {
            const chunk = retrievalMap.get(match.id);
            if (!chunk) {
              continue;
            }
            semanticHits.push({
              retrievalChunkId: chunk.id,
              documentId: chunk.document_id,
              sourceDocumentChunkId: chunk.source_document_chunk_id,
              ordinal: chunk.ordinal,
              score: match.score,
              chunkText: chunk.chunk_text,
              isSplitFromOversized: chunk.is_split_from_oversized,
            });
          }
        }
      }
    }

    const combined = this.combineHits({
      keywordHits,
      semanticHits,
      mode: input.mode,
      keywordWeight: input.keywordWeight ?? 0.5,
      semanticWeight: input.semanticWeight ?? 0.5,
    });

    const groupedEvidence = compressEvidenceItems(
      this.groupHitsBySource(combined).slice(0, 8),
      question,
      getEvidenceCharBudget(),
    );
    const summarizerInput = groupedEvidence.map((item) => ({
      documentChunkId: item.documentChunkId,
      score: item.score,
      content: item.content,
    }));
    const modelAnswer = await summarizeAnswer({
      question,
      hits: summarizerInput,
    });

    const usedIds = new Set(
      (modelAnswer?.used_chunks ?? summarizerInput).map((item) =>
        "document_chunk_id" in item ? item.document_chunk_id : item.documentChunkId,
      ),
    );
    const usedChunks = groupedEvidence
      .filter((item) => usedIds.has(item.documentChunkId))
      .map((item) => ({
        document_chunk_id: item.documentChunkId,
        retrieval_chunk_ids: item.retrievalChunkIds,
        page_nos: item.pageNos,
        bboxes: item.bboxes,
        score: item.score,
        content: item.content,
        summary_text: item.summaryText,
        source_link: item.sourceLink,
        compression_applied: item.compressionApplied,
      }));

    const answer =
      modelAnswer?.answer?.trim() ||
      usedChunks.map((chunk) => chunk.content).filter(Boolean).join("\n\n").slice(0, 4000);

    return {
      mode: input.mode,
      question,
      answer,
      used_chunks: usedChunks,
      warnings: warnings.length ? warnings : undefined,
    };
  }

  private combineHits(input: {
    keywordHits: RankedChunkHit[];
    semanticHits: RankedChunkHit[];
    mode: TraditionalRetrievalMode;
    keywordWeight: number;
    semanticWeight: number;
  }): RankedChunkHit[] {
    const scoreMap = new Map<string, RankedChunkHit>();
    const keywordMax = Math.max(...input.keywordHits.map((item) => item.score), 1);
    const semanticMax = Math.max(...input.semanticHits.map((item) => item.score), 1);
    const accumulate = (items: RankedChunkHit[], weight: number, max: number) => {
      for (const item of items) {
        const key = item.retrievalChunkId;
        const existing = scoreMap.get(key);
        const weightedScore = weight * (item.score / Math.max(max, 1));
        if (!existing) {
          scoreMap.set(key, { ...item, score: weightedScore });
          continue;
        }
        existing.score += weightedScore;
      }
    };
    if (input.mode !== "semantic") {
      accumulate(input.keywordHits, input.mode === "keyword" ? 1 : input.keywordWeight, keywordMax);
    }
    if (input.mode !== "keyword") {
      accumulate(input.semanticHits, input.mode === "semantic" ? 1 : input.semanticWeight, semanticMax);
    }

    return [...scoreMap.values()].sort((left, right) => right.score - left.score).slice(0, 24);
  }

  private groupHitsBySource(hits: RankedChunkHit[]): GroupedEvidence[] {
    const grouped = new Map<string, RankedChunkHit[]>();
    for (const hit of hits) {
      const items = grouped.get(hit.sourceDocumentChunkId) ?? [];
      items.push(hit);
      grouped.set(hit.sourceDocumentChunkId, items);
    }
    const mapped = [...grouped.entries()].map(([sourceDocumentChunkId, items]) => {
        const source = this.storage.getDocumentChunk(sourceDocumentChunkId);
        if (!source) {
          return null;
        }
        const orderedHits = [...items].sort((left, right) => left.ordinal - right.ordinal);
        const mergedExcerpt = normalizeWhitespace(orderedHits.map((item) => item.chunkText).join("\n\n"));
        const sourceLink = `/api/document-chunks/${source.id}/content`;
        return {
          documentChunkId: source.id,
          retrievalChunkIds: orderedHits.map((item) => item.retrievalChunkId),
          pageNos: parseJsonArray<number>(source.merged_page_nos_json, [source.page_no]),
          bboxes: parseJsonArray<[number, number, number, number]>(source.merged_bboxes_json, [
            JSON.parse(source.bbox_json) as [number, number, number, number],
          ]),
          score: Math.max(...orderedHits.map((item) => item.score)),
          summaryText: source.summary_text,
          sourceLink,
          mergedExcerpt,
          content: composeEvidenceContent({
            summaryText: source.summary_text,
            mergedExcerpt,
            sourceLink,
          }),
          compressionApplied: false as boolean,
        } as GroupedEvidence;
      });
    return mapped
      .filter((item): item is GroupedEvidence => item !== null)
      .sort((left, right) => right.score - left.score);
  }

  private async processImages(input: {
    documentId: string;
    filePath: string;
    originalFilename: string;
    parsedDocument: ParsedDocument;
    enableImageSemantic: boolean;
  }): Promise<Map<string, ImageChunkRendering>> {
    const renderMap = new Map<string, ImageChunkRendering>();
    const allParsedImages = input.parsedDocument.units.flatMap((unit) => unit.images.map((image) => ({ unit, image })));
    if (allParsedImages.length === 0 || extname(input.originalFilename).toLowerCase() !== ".pdf") {
      return renderMap;
    }
    const extractedImages = await this.pythonAssets.extractPdfImages(
      input.filePath,
      [...new Set(allParsedImages.map(({ image }) => image.page_no))],
    );
    const extractedByHash = new Map(extractedImages.map((image) => [image.image_hash, image] as const));
    const perPageBudget = new Map<number, number>();
    let docBudget = 0;
    const rows: StorageImageSemanticRecord[] = [];

    for (const { image } of allParsedImages) {
      const asset = extractedByHash.get(image.image_hash);
      if (!asset) {
        continue;
      }
      let interferenceScore: number | null = null;
      let hasText: boolean | null = null;
      let compressedBase64 = asset.bytes_base64;
      let compressedMime = asset.mime_type ?? "image/png";
      if (input.enableImageSemantic) {
        const inspection = await this.pythonAssets.inspectImage(asset.bytes_base64, asset.mime_type);
        interferenceScore = Number(inspection?.interference_score ?? 0);
        hasText = Boolean(inspection?.has_text);
        const shouldDrop = !hasText && interferenceScore >= 0.65;
        if (shouldDrop) {
          renderMap.set(image.image_hash, { dropped: true, markdown: "" });
          rows.push({
            imageHash: image.image_hash,
            sourceDocumentId: input.documentId,
            sourcePageNo: image.page_no,
            sourceImageIndex: image.image_index,
            mimeType: asset.mime_type,
            width: asset.width,
            height: asset.height,
            bboxJson: image.bbox ? JSON.stringify(image.bbox) : null,
            hasText,
            interferenceScore,
            isDropped: true,
            recognizable: false,
          });
          continue;
        }
        compressedBase64 = inspection?.compressed_bytes_base64 ?? asset.bytes_base64;
        compressedMime = inspection?.output_mime_type ?? asset.mime_type ?? "image/png";
      }

      const compressedBytes = Buffer.from(compressedBase64, "base64");
      const extension = compressedMime.split("/")[1] || "bin";
      const objectKey = `documents/${input.originalFilename}/images/${image.image_hash}.${extension}`;
      const head = await this.blobStore.put({ objectKey, data: compressedBytes });
      const accessibleUrl = `/api/assets/images/${image.image_hash}`;

      let semanticPayload: ImageSemanticPayload | null = null;
      const pageBudget = perPageBudget.get(image.page_no) ?? 0;
      if (
        input.enableImageSemantic &&
        compressedBytes.length <= MAX_IMAGE_BYTES_FOR_VISION &&
        pageBudget < Number(process.env.VISION_MAX_SEMANTIC_PER_PAGE || 3) &&
        docBudget < Number(process.env.VISION_MAX_SEMANTIC_PER_DOC || 50)
      ) {
        const cached = this.storage.getImageSemanticCache(image.image_hash, IMAGE_PROMPT_VERSION);
        if (cached) {
          semanticPayload = {
            recognizable: cached.recognizable,
            image_kind: cached.image_kind,
            contains_text: cached.contains_text,
            visible_text: cached.visible_text,
            summary: cached.summary,
            entities: parseJsonArray<string>(cached.entities_json, []),
            keywords: parseJsonArray<string>(cached.keywords_json, []),
            qa_hints: parseJsonArray<string>(cached.qa_hints_json, []),
            drop_reason: cached.drop_reason,
          };
        } else {
          semanticPayload = await describeImageWithVision({
            bytes: compressedBytes,
            mimeType: compressedMime,
          });
          if (semanticPayload) {
            this.storage.upsertImageSemanticCache({
              imageHash: image.image_hash,
              promptVersion: IMAGE_PROMPT_VERSION,
              recognizable: Boolean(semanticPayload.recognizable),
              imageKind: semanticPayload.image_kind ?? null,
              containsText: semanticPayload.contains_text ?? null,
              visibleText: semanticPayload.visible_text ?? null,
              summary: semanticPayload.summary ?? null,
              entitiesJson: JSON.stringify(semanticPayload.entities ?? []),
              keywordsJson: JSON.stringify(semanticPayload.keywords ?? []),
              qaHintsJson: JSON.stringify(semanticPayload.qa_hints ?? []),
              dropReason: semanticPayload.drop_reason ?? null,
              semanticModel: resolveVisionConfig().modelName,
            });
          }
        }
        perPageBudget.set(image.page_no, pageBudget + 1);
        docBudget += 1;
      }

      if (semanticPayload && semanticPayload.recognizable === false) {
        await this.blobStore.delete({ objectKey }).catch(() => undefined);
        renderMap.set(image.image_hash, { dropped: true, markdown: "" });
        rows.push({
          imageHash: image.image_hash,
          sourceDocumentId: input.documentId,
          sourcePageNo: image.page_no,
          sourceImageIndex: image.image_index,
          mimeType: compressedMime,
          width: asset.width,
          height: asset.height,
          bboxJson: image.bbox ? JSON.stringify(image.bbox) : null,
          hasText,
          interferenceScore,
          isDropped: true,
          recognizable: false,
        });
        continue;
      }

      const semanticText = semanticPayload
        ? [semanticPayload.summary, semanticPayload.visible_text, ...(semanticPayload.keywords ?? [])]
            .filter(Boolean)
            .join("\n")
            .trim()
        : null;
      const markdown = semanticText
        ? `![image](${accessibleUrl})\n\n${semanticText}`
        : `![image](${accessibleUrl})`;
      renderMap.set(image.image_hash, { dropped: false, markdown });
      rows.push({
        imageHash: image.image_hash,
        sourceDocumentId: input.documentId,
        sourcePageNo: image.page_no,
        sourceImageIndex: image.image_index,
        mimeType: compressedMime,
        width: asset.width,
        height: asset.height,
        bboxJson: image.bbox ? JSON.stringify(image.bbox) : null,
        objectKey,
        storageUri: head.storageUri,
        hasText,
        interferenceScore,
        isDropped: false,
        recognizable: semanticPayload ? semanticPayload.recognizable !== false : null,
        accessibleUrl,
        semanticText: semanticText || null,
        semanticModel: semanticPayload && input.enableImageSemantic ? resolveVisionConfig().modelName : null,
      });
    }
    this.storage.upsertImageSemantics(rows);
    return renderMap;
  }

  private buildDocumentChunks(input: {
    documentId: string;
    parsedDocument: ParsedDocument;
    imageRenderMap: Map<string, ImageChunkRendering>;
  }): StorageDocumentChunkRecord[] {
    const perPage = input.parsedDocument.units.map((unit) => ({
      unitNo: unit.unit_no,
      blocks: this.renderUnitBlocks(unit, input.imageRenderMap),
    }));
    const records: StorageDocumentChunkRecord[] = [];
    let documentIndex = 0;
    for (let pageIndex = 0; pageIndex < perPage.length; pageIndex += 1) {
      const page = perPage[pageIndex]!;
      for (let blockIndex = 0; blockIndex < page.blocks.length; blockIndex += 1) {
        const block = page.blocks[blockIndex]!;
        if (!normalizeWhitespace(block.markdown)) {
          continue;
        }
        let mergedContent = block.markdown;
        const mergedPageNos = [page.unitNo];
        const mergedBboxes = [block.bbox];
        let scanPageIndex = pageIndex;
        while (
          ["table", "text"].includes(block.block_type) &&
          scanPageIndex + 1 < perPage.length &&
          blockIndex === page.blocks.length - 1
        ) {
          const nextPage = perPage[scanPageIndex + 1]!;
          const nextHead = nextPage.blocks[0];
          const nextUnit = input.parsedDocument.units[scanPageIndex + 1]!;
          if (!nextHead || nextHead.block_type !== block.block_type || !pageDominatedByType(nextUnit, block.block_type)) {
            break;
          }
          mergedContent = normalizeWhitespace(`${mergedContent}\n\n${nextHead.markdown}`);
          mergedPageNos.push(nextPage.unitNo);
          mergedBboxes.push(nextHead.bbox);
          nextPage.blocks = nextPage.blocks.slice(1);
          scanPageIndex += 1;
        }
        const sizeClass = classifySize(mergedContent);
        const record: StorageDocumentChunkRecord = {
          id: stableId("dchunk", `${input.documentId}:${documentIndex}:${page.unitNo}:${block.index}:${mergedContent}`),
          documentId: input.documentId,
          pageNo: page.unitNo,
          documentIndex,
          pageIndex: block.index,
          blockType: block.block_type,
          bboxJson: JSON.stringify(block.bbox),
          contentMd: mergedContent,
          sizeClass,
          summaryText: sizeClass === "oversized" ? ruleSummary(mergedContent, block.block_type) : null,
          mergedPageNosJson: JSON.stringify(mergedPageNos),
          mergedBboxesJson: JSON.stringify(mergedBboxes.map((bbox) => [...bbox])),
        };
        records.push(record);
        documentIndex += 1;
      }
    }
    return records;
  }

  private renderUnitBlocks(
    unit: ParsedUnit,
    imageRenderMap: Map<string, ImageChunkRendering>,
  ): ParsedBlock[] {
    const blocks = unit.blocks.length
      ? [...unit.blocks]
      : [
          {
            index: 0,
            block_type: "text",
            bbox: [0, 0, 0, 0],
            markdown: unit.markdown,
            char_count: unit.markdown.length,
            image_hash: null,
            source_image_index: null,
          },
        ];
    return blocks
      .map((block) => {
        if (block.block_type !== "picture" || !block.image_hash) {
          return block;
        }
        const rendered = imageRenderMap.get(block.image_hash);
        return {
          ...block,
          markdown: rendered?.dropped ? "" : rendered?.markdown || block.markdown,
          char_count: (rendered?.markdown || block.markdown).length,
        };
      })
      .filter((block) => normalizeWhitespace(block.markdown));
  }

  private buildRetrievalChunks(
    documentId: string,
    documentChunks: StorageDocumentChunkRecord[],
  ): StorageRetrievalChunkRecord[] {
    const thresholds = getChunkThresholds();
    const records: StorageRetrievalChunkRecord[] = [];
    let ordinal = 0;
    let pendingSmall: StorageDocumentChunkRecord[] = [];
    const flushPending = () => {
      if (pendingSmall.length === 0) {
        return;
      }
      const text = normalizeWhitespace(pendingSmall.map((item) => item.contentMd).join("\n\n"));
      records.push({
        id: stableId("rchunk", `${documentId}:${pendingSmall[0]!.id}:${ordinal}:${text}`),
        documentId,
        sourceDocumentChunkId: pendingSmall[0]!.id,
        ordinal,
        chunkText: text,
        sizeClass: classifySize(text),
        isSplitFromOversized: false,
      });
      ordinal += 1;
      pendingSmall = [];
    };

    for (const chunk of documentChunks) {
      if (chunk.sizeClass === "small") {
        const candidate = normalizeWhitespace(
          [...pendingSmall.map((item) => item.contentMd), chunk.contentMd].join("\n\n"),
        );
        if (candidate.length > thresholds.normalMaxChars) {
          flushPending();
          pendingSmall = [chunk];
        } else {
          pendingSmall.push(chunk);
        }
        continue;
      }
      flushPending();
      if (chunk.sizeClass === "normal") {
        records.push({
          id: stableId("rchunk", `${documentId}:${chunk.id}:${ordinal}`),
          documentId,
          sourceDocumentChunkId: chunk.id,
          ordinal,
          chunkText: chunk.contentMd,
          sizeClass: "normal",
          isSplitFromOversized: false,
        });
        ordinal += 1;
        continue;
      }
      for (const part of splitOversized(chunk.contentMd, chunk.blockType)) {
        records.push({
          id: stableId("rchunk", `${documentId}:${chunk.id}:${ordinal}:${part}`),
          documentId,
          sourceDocumentChunkId: chunk.id,
          ordinal,
          chunkText: part,
          sizeClass: part.length <= thresholds.smallMaxChars ? "small" : "normal",
          isSplitFromOversized: true,
        });
        ordinal += 1;
      }
    }
    flushPending();
    return records;
  }
}

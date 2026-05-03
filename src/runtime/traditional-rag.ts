/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
*/

import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { extname } from "node:path";

import { resolveDocumentScope } from "./document-library.js";
import { evaluateImageSemanticCandidateInspection } from "./image-semantic-screening.js";
import {
  buildVisionPromptMessages,
  renderImageSemantic,
  type ImageSemanticPayload,
} from "./image-semantic.js";
import { resolveEmbeddingConfig, resolveTextConfig, resolveVisionConfig } from "./model-config.js";
import { PythonDocumentAssetBridge } from "./python-document-assets.js";
import type { BlobStore } from "../types/library.js";
import type { ParsedBlock, ParsedDocument, ParsedImage, ParsedUnit } from "../types/parsing.js";
import type {
  TraditionalRagCerContextRef,
  TraditionalRagChunkReference,
  TraditionalRagDetailChunk,
  TraditionalRagPromptMessage,
  TraditionalRagPromptPreviewResult,
  TraditionalRagQueryResult,
  TraditionalRagRetrieveResult,
  TraditionalRetrievalMode,
} from "../types/rag.js";
import type {
  SqliteStorageBackend,
  StorageDocumentChunkRecord,
  StorageFixedRetrievalChunkRecord,
  StorageImageSemanticRecord,
  StorageRetrievalChunkRecord,
  StoredDocument,
  StoredDocumentChunk,
  StoredFixedRetrievalChunk,
  StoredImageSemantic,
  StoredRetrievalChunk,
} from "../types/storage.js";

const DEFAULT_SMALL_CHUNK_MAX_CHARS = 399;
const DEFAULT_NORMAL_CHUNK_MAX_CHARS = 2000;
const DEFAULT_EVIDENCE_CHAR_BUDGET = 12000;
const CER_CONTEXT_EXCERPT_TARGET = 600;
const MAX_IMAGE_BYTES_FOR_VISION = 4 * 1024 * 1024;
const IMAGE_PROMPT_VERSION = "v2";

interface ChunkThresholds {
  smallMaxChars: number;
  normalMaxChars: number;
}

interface SourceChunkRecord {
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
}

interface RetrievalChunkPlan {
  id: string;
  documentId: string;
  ordinal: number;
  contentMd: string;
  sizeClass: "small" | "normal" | "oversized";
  summaryText: string | null;
  sourceChunkIds: string[];
  pageNosJson: string;
  sourceLocator: string | null;
  bboxesJson: string;
}

interface FixedChunkPiece {
  sourceChunkId: string;
  pageNo: number;
  contentMd: string;
  blockType: string;
  mergedPageNosJson: string;
  mergedBboxesJson: string;
  bboxJson: string;
}

export interface ImageChunkRendering {
  dropped: boolean;
  markdown: string;
}

interface RankedChunkHit {
  retrievalUnitId: string;
  documentId: string;
  sourceDocumentChunkId: string;
  referenceRetrievalChunkId: string | null;
  ordinal: number;
  score: number;
  unitText: string;
  isSplitFromOversized: boolean;
}

type TraditionalRetrievalChunkRecord = StoredRetrievalChunk | StoredFixedRetrievalChunk;

interface GroupedEvidence {
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

interface EvidenceSegment {
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

interface PreparedEvidenceSet {
  mode: TraditionalRetrievalMode;
  question: string;
  warnings: string[];
  chunks: GroupedEvidence[];
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

function qdrantPointIdForRetrievalUnit(retrievalUnitId: string): string {
  return stableUuid(`qdrant:${retrievalUnitId}`);
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

function mergeContinuedBlockMarkdown(left: string, right: string, blockType: string): string {
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

function normalizeBaseUrl(baseUrl: string | null): string {
  return (baseUrl?.trim() || "https://api.openai.com/v1").replace(/\/+$/, "");
}

function resolveServerBaseUrl(): string {
  const configured = String(process.env.FS_EXPLORER_SERVER_BASE_URL ?? "").trim();
  if (configured) {
    return configured.replace(/\/+$/, "");
  }
  const port = parsePositiveIntegerEnv("FS_EXPLORER_PORT", 8000);
  return `http://localhost:${port}`;
}

function toAbsoluteServerUrl(pathOrUrl: string): string {
  const value = String(pathOrUrl || "").trim();
  if (!value) {
    return value;
  }
  if (/^https?:\/\//i.test(value)) {
    return value;
  }
  return `${resolveServerBaseUrl()}${value.startsWith("/") ? value : `/${value}`}`;
}

function resolveTextRequestTimeoutMs(): number {
  const raw = Number.parseInt(process.env.FS_EXPLORER_TEXT_TIMEOUT_MS || "", 10);
  if (Number.isFinite(raw) && raw > 0) {
    return raw;
  }
  return 60_000;
}

async function fetchWithTimeout(
  input: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Text model request timed out after ${timeoutMs}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function extractContentText(
  content: string | Array<{ type?: string; text?: string }> | undefined,
  trim: boolean,
): string {
  if (typeof content === "string") {
    return trim ? content.trim() : content;
  }
  if (Array.isArray(content)) {
    const text = content
      .map((item) => (item.type === "text" || item.text ? String(item.text ?? "") : ""))
      .filter(Boolean)
      .join("\n");
    return trim ? text.trim() : text;
  }
  return "";
}

function extractChoiceText(response: {
  choices?: Array<{ message?: { content?: string | Array<{ type?: string; text?: string }> } }>;
}): string {
  return extractContentText(response.choices?.[0]?.message?.content, true);
}

function extractDeltaText(response: {
  choices?: Array<{
    delta?: { content?: string | Array<{ type?: string; text?: string }> };
    message?: { content?: string | Array<{ type?: string; text?: string }> };
  }>;
}): string {
  return extractContentText(response.choices?.[0]?.delta?.content ?? response.choices?.[0]?.message?.content, false);
}

async function* iterSseDataLines(stream: ReadableStream<Uint8Array>): AsyncIterable<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex >= 0) {
        const block = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const dataLines = block
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart());
        if (dataLines.length > 0) {
          yield dataLines.join("\n");
        }
        separatorIndex = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

async function fetchJsonWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs = 60_000,
): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    let response: Response;
    try {
      response = await fetch(url, { ...init, signal: controller.signal });
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw new Error(`Request timed out after ${timeoutMs}ms.`);
      }
      throw error;
    }
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
  const [vector] = await createEmbeddings([text]);
  return vector ?? null;
}

function normalizeEmbeddingVector(vector: unknown): number[] | null {
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

function embeddingBatchSize(): number {
  return parsePositiveIntegerEnv("EMBEDDING_BATCH_SIZE", 50);
}

function embeddingConcurrency(): number {
  return parsePositiveIntegerEnv("EMBEDDING_CONCURRENCY", 3);
}

function qdrantUpsertBatchSize(): number {
  return parsePositiveIntegerEnv("QDRANT_UPSERT_BATCH_SIZE", 128);
}

function formatTraditionalRagError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function createEmbeddings(
  texts: string[],
  input: { batchIndex?: number; batchCount?: number } = {},
): Promise<Array<number[] | null>> {
  if (texts.length === 0) {
    return [];
  }
  const config = resolveEmbeddingConfig();
  if (!config.apiKey) {
    return texts.map(() => null);
  }
  const baseUrl = (config.baseUrl?.trim() || "https://api.openai.com/v1").replace(/\/+$/, "");
  const endpoint = `${baseUrl}/embeddings`;
  const batchLabel =
    input.batchIndex != null && input.batchCount != null
      ? `batch ${input.batchIndex + 1}/${input.batchCount}`
      : "single request";
  let payload: unknown;
  try {
    payload = await fetchJsonWithTimeout(
      endpoint,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify({
          model: config.modelName,
          input: texts,
        }),
      },
      60_000,
    );
  } catch (error) {
    throw new Error(
      `[traditional-rag] Embedding request failed for ${batchLabel} (${texts.length} texts, ${endpoint}): ${formatTraditionalRagError(error)}`,
    );
  }
  const rows = (payload as { data?: Array<{ embedding?: number[]; index?: number }> }).data ?? [];
  if (rows.length !== texts.length) {
    console.warn(
      `[traditional-rag] Embedding provider returned ${rows.length} vectors for ${texts.length} inputs; unmatched entries will be discarded.`,
    );
  }
  const indexedRows = Array.from({ length: texts.length }, () => null as { embedding?: number[] } | null);
  for (let rawIndex = 0; rawIndex < rows.length; rawIndex += 1) {
    const row = rows[rawIndex];
    const targetIndex =
      Number.isInteger(row?.index) && Number(row.index) >= 0 && Number(row.index) < texts.length
        ? Number(row.index)
        : rawIndex;
    indexedRows[targetIndex] = row ?? null;
  }
  return texts.map((_, index) => normalizeEmbeddingVector(indexedRows[index]?.embedding));
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
  const prompts = buildVisionPromptMessages();
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
            content: prompts.systemPrompt,
          },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: prompts.userPrompt,
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

function buildAnswerEvidenceText(input: {
  question: string;
  hits: GroupedEvidence[];
}): string {
  const evidence = input.hits
    .map(
      (hit) =>
        `[${hit.citationNo}] document=${hit.documentName} locator=${hit.sourceLocator ?? "-"} ` +
        `pages=${hit.pageNos.join(", ") || "-"} score=${hit.score.toFixed(4)} ` +
        `full_content_via_source=${hit.cerSegments.some((segment) => segment.isOversized) ? "yes" : "no"}\n${hit.content}`,
    )
    .join("\n\n");
  return `Question:\n${input.question}\n\nEvidence:\n${evidence}`;
}

function buildAnswerSystemPrompt(variant: "json" | "stream"): string {
  if (variant === "json") {
    return (
      "Answer only from the numbered evidence. Do not reproduce long evidence excerpts. " +
      "Evidence blocks may contain direct links to the full content. For oversized evidence, the excerpt may be compressed and the link can be included in your answer when users should inspect the full content." +
      "Return strict JSON with keys answer and citations. citations must be an array of evidence numbers such as [1, 3]. " +
      "Use inline references like [1] in the answer when helpful."
    );
  }
  return (
    "Answer only from the numbered evidence. Cite supporting evidence using [n]. " +
    "Do not quote or restate long evidence excerpts. If the evidence is insufficient, say so briefly. " +
    "Evidence blocks may contain direct links to the full content. For oversized evidence, the excerpt may be compressed and the link can be included in your answer when users should inspect the full content."
  );
}

function buildAnswerMessages(input: {
  question: string;
  hits: GroupedEvidence[];
  variant: "json" | "stream";
}): TraditionalRagPromptMessage[] {
  return [
    {
      role: "system",
      content: buildAnswerSystemPrompt(input.variant),
    },
    {
      role: "user",
      content: buildAnswerEvidenceText(input),
    },
  ];
}

function buildAnswerRequestPayload(input: {
  question: string;
  hits: GroupedEvidence[];
  variant: "json" | "stream";
}): {
  model: string | null;
  temperature: number;
  response_format?: { type: "json_object" };
  stream?: true;
  messages: TraditionalRagPromptMessage[];
} {
  const config = resolveTextConfig();
  const payload: {
    model: string | null;
    temperature: number;
    response_format?: { type: "json_object" };
    stream?: true;
    messages: TraditionalRagPromptMessage[];
  } = {
    model: config.modelName,
    temperature: 0,
    messages: buildAnswerMessages(input),
  };
  if (input.variant === "json") {
    payload.response_format = { type: "json_object" };
  } else {
    payload.stream = true;
  }
  return payload;
}

function buildFallbackAnswer(hits: GroupedEvidence[]): string {
  if (hits.length === 0) {
    return "未召回到可用证据。";
  }
  return `已召回 ${hits.length} 个相关片段，请查看下方“详情内容如下”。`;
}

async function answerWithCitations(input: {
  question: string;
  hits: GroupedEvidence[];
}): Promise<{ answer: string; citations: number[] } | null> {
  const config = resolveTextConfig();
  if (!config.apiKey) {
    return null;
  }
  const baseUrl = normalizeBaseUrl(config.baseUrl);
  const payload = await fetchJsonWithTimeout(
    `${baseUrl}/chat/completions`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify(buildAnswerRequestPayload({ ...input, variant: "json" })),
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
      citations?: unknown;
    };
    return {
      answer: String(parsed.answer || ""),
      citations: Array.isArray(parsed.citations)
        ? uniqueOrdered(
            parsed.citations
              .map((item) => Number.parseInt(String(item ?? ""), 10))
              .filter((item) => Number.isFinite(item) && item > 0),
          )
        : [],
    };
  } catch {
    return null;
  }
}

async function* streamAnswerFromModel(input: {
  question: string;
  hits: GroupedEvidence[];
}): AsyncIterable<string> {
  const config = resolveTextConfig();
  if (!config.apiKey) {
    const fallback = buildFallbackAnswer(input.hits);
    if (fallback) {
      yield fallback;
    }
    return;
  }
  const endpoint = `${normalizeBaseUrl(config.baseUrl)}/chat/completions`;
  const timeoutMs = resolveTextRequestTimeoutMs();
  const requestPayload = buildAnswerRequestPayload({ ...input, variant: "stream" });
  let response = await fetchWithTimeout(
    endpoint,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify(requestPayload),
    },
    timeoutMs,
  );

  if (!response.ok || !response.body) {
    const fallbackPayload = {
      ...requestPayload,
      stream: undefined,
    };
    response = await fetchWithTimeout(
      endpoint,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify(fallbackPayload),
      },
      timeoutMs,
    );
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(`Text model request failed: HTTP ${response.status} ${body}`.trim());
    }
    const fullText = extractChoiceText((await response.json()) as { choices?: Array<{ message?: { content?: string } }> });
    if (fullText) {
      yield fullText;
    }
    return;
  }

  for await (const dataLine of iterSseDataLines(response.body)) {
    if (!dataLine || dataLine === "[DONE]") {
      if (dataLine === "[DONE]") {
        break;
      }
      continue;
    }
    const text = extractDeltaText(
      JSON.parse(dataLine) as {
        choices?: Array<{
          delta?: { content?: string | Array<{ type?: string; text?: string }> };
          message?: { content?: string | Array<{ type?: string; text?: string }> };
        }>;
      },
    );
    if (text) {
      yield text;
    }
  }
}

async function ensureQdrantCollection(documentId: string, vectorSize: number): Promise<boolean> {
  const config = qdrantConfig();
  if (!config.url) {
    return false;
  }
  const endpoint = `${config.url.replace(/\/+$/, "")}/collections/doc_${documentId}`;
  let response: Response;
  try {
    response = await fetch(endpoint, {
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
  } catch (error) {
    throw new Error(
      `[traditional-rag] Qdrant collection creation failed for doc_${documentId} (${endpoint}): ${formatTraditionalRagError(error)}`,
    );
  }
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
  points: Array<{ retrievalUnitId: string; vector: number[] }>,
): Promise<number> {
  const config = qdrantConfig();
  if (!config.url || points.length === 0) {
    return 0;
  }
  const endpoint = `${config.url.replace(/\/+$/, "")}/collections/doc_${documentId}/points?wait=true`;
  const batchSize = qdrantUpsertBatchSize();
  const batchCount = Math.ceil(points.length / batchSize);
  for (let batchIndex = 0; batchIndex < batchCount; batchIndex += 1) {
    const batch = points.slice(batchIndex * batchSize, (batchIndex + 1) * batchSize);
    try {
      await fetchJsonWithTimeout(
        endpoint,
        {
          method: "PUT",
          headers: {
            "content-type": "application/json",
            ...(config.apiKey ? { "api-key": config.apiKey } : {}),
          },
          body: JSON.stringify({
            points: batch.map((point) => ({
              id: qdrantPointIdForRetrievalUnit(point.retrievalUnitId),
              vector: point.vector,
              payload: { retrieval_unit_id: point.retrievalUnitId },
            })),
          }),
        },
        60_000,
      );
    } catch (error) {
      throw new Error(
        `[traditional-rag] Qdrant points upsert failed for doc_${documentId} (batch ${batchIndex + 1}/${batchCount}, ${batch.length} points, ${endpoint}): ${formatTraditionalRagError(error)}`,
      );
    }
  }
  return points.length;
}

async function rebuildQdrantCollection(
  documentId: string,
  chunks: Array<{ id: string; text: string }>,
): Promise<number> {
  const batchSize = embeddingBatchSize();
  const batches: Array<Array<{ id: string; text: string }>> = [];
  for (let index = 0; index < chunks.length; index += batchSize) {
    batches.push(chunks.slice(index, index + batchSize));
  }
  const pointsByBatch: Array<Array<{ retrievalUnitId: string; vector: number[] }>> = Array.from(
    { length: batches.length },
    () => [],
  );
  let nextBatchIndex = 0;
  const workerCount = Math.min(embeddingConcurrency(), Math.max(batches.length, 1));
  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      while (true) {
        const batchIndex = nextBatchIndex;
        nextBatchIndex += 1;
        if (batchIndex >= batches.length) {
          return;
        }
        const batch = batches[batchIndex]!;
        const vectors = await createEmbeddings(
          batch.map((chunk) => chunk.text),
          { batchIndex, batchCount: batches.length },
        );
        const batchPoints: Array<{ retrievalUnitId: string; vector: number[] }> = [];
        for (let itemIndex = 0; itemIndex < batch.length; itemIndex += 1) {
          const vector = vectors[itemIndex];
          if (!vector) {
            continue;
          }
          batchPoints.push({ retrievalUnitId: batch[itemIndex]!.id, vector });
        }
        pointsByBatch[batchIndex] = batchPoints;
      }
    }),
  );
  const points = pointsByBatch.flat();
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
  const result = (payload as { result?: Array<{ id?: string; score?: number; payload?: { retrieval_unit_id?: string } }> } | null)
    ?.result;
  return (result ?? []).map((item) => ({
    id: String(item.payload?.retrieval_unit_id ?? item.id ?? ""),
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

function splitFixedChunkText(content: string, maxChars: number): string[] {
  const text = String(content || "");
  if (text.length <= maxChars) {
    return [text];
  }
  const parts: string[] = [];
  let remaining = text;
  while (remaining.length > maxChars) {
    const slice = remaining.slice(0, maxChars);
    const bestBreakAt = Math.max(
      slice.lastIndexOf("\n\n"),
      slice.lastIndexOf("。"),
      slice.lastIndexOf("！"),
      slice.lastIndexOf("？"),
      slice.lastIndexOf(". "),
      slice.lastIndexOf(" "),
    );
    const breakAt = bestBreakAt >= 0 ? bestBreakAt : maxChars;
    const end = breakAt > Math.floor(maxChars * 0.5) ? breakAt : maxChars;
    parts.push(remaining.slice(0, end).trim());
    remaining = remaining.slice(end).trimStart();
  }
  if (remaining.trim()) {
    parts.push(remaining.trim());
  }
  return parts.filter((part) => part.length > 0);
}

function splitFixedChunkTextSmart(content: string, maxChars: number): string[] {
  const text = normalizeWhitespace(String(content || ""));
  if (text.length <= maxChars) {
    return [text];
  }
  const parts: string[] = [];
  let remaining = text;
  while (remaining.length > maxChars) {
    const slice = remaining.slice(0, maxChars);
    const bestBreakAt = Math.max(
      ...[
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n- ",
        "\n1. ",
        "\n2. ",
        "\n3. ",
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        ". ",
        "; ",
        " ",
      ].map((marker) => slice.lastIndexOf(marker)),
    );
    const breakAt = bestBreakAt >= 0 ? bestBreakAt : maxChars;
    const end = breakAt > Math.floor(maxChars * 0.5) ? breakAt : maxChars;
    parts.push(slice.slice(0, end).trim());
    remaining = remaining.slice(end).trimStart();
  }
  if (remaining.trim()) {
    parts.push(remaining.trim());
  }
  return parts.filter((part) => part.length > 0);
}

function isHeadingLikeShortCenterSegment(segment: EvidenceSegment): boolean {
  if (!segment.isCenter) {
    return false;
  }
  if (segment.blockType && segment.blockType !== "text") {
    return false;
  }
  const text = normalizeWhitespace(segment.mergedExcerpt);
  if (!text || text.length > 160) {
    return false;
  }
  const lines = String(segment.mergedExcerpt || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    return false;
  }
  return lines.some((line) => {
    const normalizedLine = line.replace(/\*+/g, "").trim();
    return (
      /^#{1,6}\s*/.test(line) ||
      /^(section|chapter|appendix)\b/i.test(normalizedLine) ||
      /(利润表|资产负债表|现金流量表|所有者权益变动表|附注|注释)$/.test(normalizedLine)
    );
  });
}

function uniqueOrdered<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function sourceLocatorFromPages(pageNos: number[]): string | null {
  const normalized = uniqueOrdered(
    pageNos.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0),
  );
  if (normalized.length === 0) {
    return null;
  }
  if (normalized.length === 1) {
    return `page-${normalized[0]}`;
  }
  return `pages-${normalized.join("-")}`;
}

function buildCerSegmentLabels(citationNo: number, segments: EvidenceSegment[]): string[] {
  if (segments.length === 0) {
    return [];
  }
  const labels = new Array<string>(segments.length).fill(`chunk[${citationNo}]`);
  const centerIndex = segments.findIndex((segment) => segment.isCenter);
  if (centerIndex < 0) {
    return labels;
  }
  labels[centerIndex] = `chunk[${citationNo}]`;
  let previousCount = 0;
  for (let index = centerIndex - 1; index >= 0; index -= 1) {
    previousCount += 1;
    labels[index] = `chunk[${citationNo}-p-${previousCount}]`;
  }
  let nextCount = 0;
  for (let index = centerIndex + 1; index < segments.length; index += 1) {
    nextCount += 1;
    labels[index] = `chunk[${citationNo}-n-${nextCount}]`;
  }
  return labels;
}

function composeSegmentContent(segment: EvidenceSegment, chunkLabel: string): string {
  return composeEvidenceContent({
    chunkLabel,
    summaryText: segment.summaryText,
    mergedExcerpt: segment.mergedExcerpt,
    sourceLink: segment.sourceLink,
    includeSource: segment.isOversized,
  });
}

function sentenceSplit(text: string): string[] {
  const normalized = normalizeWhitespace(text);
  if (!normalized) {
    return [];
  }
  const sentences = normalized
    .split(/(?<=[\n。！？!?])\s+/)
    .map((value) => value.trim())
    .filter(Boolean);
  return sentences.length ? sentences : [normalized];
}

function isTableSegment(segment: EvidenceSegment): boolean {
  return segment.blockType === "table";
}

function extractRelevantTextSegment(text: string, queryTokens: string[]): string {
  const sentences = sentenceSplit(text);
  if (sentences.length <= 2 || queryTokens.length === 0) {
    return normalizeWhitespace(text);
  }
  const lowered = sentences.map((sentence) => sentence.toLowerCase());
  const hits = lowered
    .map((sentence, index) => ({
      index,
      matched: queryTokens.some((token) => sentence.includes(token)),
    }))
    .filter((item) => item.matched)
    .map((item) => item.index);
  if (hits.length === 0) {
    return compressExcerptAroundKeywords(text, queryTokens, Math.min(Math.max(text.length, 160), CER_CONTEXT_EXCERPT_TARGET));
  }
  const keep = new Set<number>();
  for (const index of hits) {
    keep.add(index);
    if (index > 0) {
      keep.add(index - 1);
    }
    if (index + 1 < sentences.length) {
      keep.add(index + 1);
    }
  }
  return [...keep]
    .sort((left, right) => left - right)
    .map((index) => sentences[index] ?? "")
    .filter(Boolean)
    .join("\n");
}

function extractRelevantTableSegment(text: string, queryTokens: string[]): string {
  const lines = normalizeWhitespace(text).split("\n").map((line) => line.trim()).filter(Boolean);
  if (lines.length <= 3 || queryTokens.length === 0) {
    return normalizeWhitespace(text);
  }
  const keep = new Set<number>();
  keep.add(0);
  for (let index = 0; index < lines.length; index += 1) {
    const lowered = lines[index]!.toLowerCase();
    if (!queryTokens.some((token) => lowered.includes(token))) {
      continue;
    }
    keep.add(index);
    if (index > 0) {
      keep.add(index - 1);
    }
    if (index + 1 < lines.length) {
      keep.add(index + 1);
    }
  }
  if (keep.size <= 1) {
    return compressExcerptAroundKeywords(text, queryTokens, Math.min(Math.max(text.length, 160), CER_CONTEXT_EXCERPT_TARGET));
  }
  return [...keep]
    .sort((left, right) => left - right)
    .map((index) => lines[index] ?? "")
    .filter(Boolean)
    .join("\n");
}

function extractRelevantSegment(segment: EvidenceSegment, queryTokens: string[]): string {
  if (!segment.mergedExcerpt) {
    return segment.mergedExcerpt;
  }
  return isTableSegment(segment)
    ? extractRelevantTableSegment(segment.mergedExcerpt, queryTokens)
    : extractRelevantTextSegment(segment.mergedExcerpt, queryTokens);
}

function materializeEvidenceItem(item: GroupedEvidence): GroupedEvidence {
  const cerSegments = item.cerSegments.map((segment) =>
    segment.isCenter
      ? {
          ...segment,
          summaryText: item.summaryText,
          mergedExcerpt: item.mergedExcerpt,
          sourceLink: item.sourceLink,
          sourceLocator: item.sourceLocator,
          pageNos: item.pageNos,
          isOversized: item.hasOversizedSplitChild,
        }
      : segment,
  );
  const baseContent = composeEvidenceContent({
    chunkLabel: `chunk[${item.citationNo}]`,
    summaryText: item.summaryText,
    mergedExcerpt: item.mergedExcerpt,
    sourceLink: item.sourceLink,
    includeSource: item.hasOversizedSplitChild,
  });
  const cerLabels = buildCerSegmentLabels(item.citationNo, cerSegments);
  const enrichedContent = cerSegments
    .map((segment, segmentIndex) => composeSegmentContent(segment, cerLabels[segmentIndex] || `chunk[${item.citationNo}]`))
    .filter((value) => String(value || "").trim())
    .join("\n\n");
  return {
    ...item,
    cerSegments,
    baseContent,
    enrichedContent,
    content: enrichedContent || baseContent,
    cerApplied: cerSegments.length > 1,
    cerContextRefs: cerSegments
      .filter((segment) => !segment.isCenter)
      .map((segment) => ({
        reference_id: segment.referenceId,
        reference_kind: segment.referenceKind,
        role: "context" as const,
        source_locator: segment.sourceLocator,
        source_link: segment.sourceLink,
        page_nos: segment.pageNos,
      })),
  };
}

function toPublicChunkReference(
  item: GroupedEvidence,
  options: { includeCerDebug?: boolean } = {},
): TraditionalRagChunkReference {
  const base = {
    citation_no: item.citationNo,
    reference_id: item.referenceId,
    reference_kind: item.referenceKind,
    document_id: item.documentId,
    document_name: item.documentName,
    source_locator: item.sourceLocator,
    show_full_chunk_detail: false,
    document_chunk_ids: item.documentChunkIds,
    retrieval_unit_ids: item.retrievalUnitIds,
    page_nos: item.pageNos,
    bboxes: item.bboxes,
    score: item.score,
    source_link: item.sourceLink,
    compression_applied: item.compressionApplied,
  };
  if (!options.includeCerDebug) {
    return base;
  }
  return {
    ...base,
    debug_cer_applied: item.cerApplied,
    debug_cer_context_refs: item.cerContextRefs,
    debug_enriched_content: item.content,
  };
}

function toPublicChunkReferences(
  items: GroupedEvidence[],
  options: { includeCerDebug?: boolean } = {},
): TraditionalRagChunkReference[] {
  const detailIds = new Set(selectHighScoreOversizedChunks(items).map((item) => item.referenceId));
  return items.map((item) => ({
    ...toPublicChunkReference(item, options),
    show_full_chunk_detail: detailIds.has(item.referenceId),
  }));
}

function toPublicDetailChunks(items: GroupedEvidence[]): TraditionalRagDetailChunk[] {
  const detailMap = new Map<string, TraditionalRagDetailChunk>();
  for (const item of items) {
    const add = (chunk: TraditionalRagDetailChunk) => {
      const key = `${chunk.reference_kind}:${chunk.reference_id}`;
      if (!detailMap.has(key)) {
        detailMap.set(key, chunk);
      }
    };
    if (item.referenceKind === "retrieval_chunk" && item.hasOversizedSplitChild) {
      add({
        ...toPublicChunkReference(item),
        show_full_chunk_detail: true,
      });
    }
    if (item.referenceKind === "document_chunk" && item.hasOversizedSplitChild) {
      add({
        ...toPublicChunkReference(item),
        show_full_chunk_detail: true,
      });
    }
    for (const segment of item.cerSegments) {
      if (!segment.isOversized || !segment.sourceLink) {
        continue;
      }
      add({
        citation_no: item.citationNo,
        reference_id: segment.referenceId,
        reference_kind: segment.referenceKind,
        document_id: item.documentId,
        document_name: item.documentName,
        source_locator: segment.sourceLocator,
        show_full_chunk_detail: true,
        document_chunk_ids:
          segment.referenceKind === "retrieval_chunk" && segment.referenceId === item.referenceId
            ? item.documentChunkIds
            : segment.referenceKind === "document_chunk"
              ? [segment.referenceId]
              : [],
        retrieval_unit_ids: item.retrievalUnitIds,
        page_nos: segment.pageNos,
        bboxes: [],
        score: item.score,
        source_link: segment.sourceLink,
        compression_applied: item.compressionApplied,
      });
    }
  }
  return [...detailMap.values()];
}

function answerContainsAnyDetailLink(answer: string, detailChunks: TraditionalRagDetailChunk[]): boolean {
  const normalized = String(answer || "");
  return detailChunks.some((item) => item.source_link && normalized.includes(item.source_link));
}

function parseCitationNumbers(value: string): number[] {
  const citations: number[] = [];
  for (const match of String(value || "").matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
    const raw = String(match[1] ?? "");
    for (const piece of raw.split(",")) {
      const parsed = Number.parseInt(piece.trim(), 10);
      if (Number.isFinite(parsed) && parsed > 0) {
        citations.push(parsed);
      }
    }
  }
  return uniqueOrdered(citations);
}

function selectHighScoreOversizedChunks(chunks: GroupedEvidence[]): GroupedEvidence[] {
  const oversized = chunks.filter((item) => item.hasOversizedSplitChild);
  if (oversized.length === 0) {
    return [];
  }
  const maxScore = Math.max(...oversized.map((item) => item.score), 0);
  const threshold = maxScore * 0.85;
  return oversized.filter((item) => item.score >= threshold);
}

function appendOversizedChunkLinks(answer: string, chunks: GroupedEvidence[]): string {
  const selected = selectHighScoreOversizedChunks(chunks).filter((item) => item.sourceLink);
  if (selected.length === 0) {
    return answer;
  }
  const links = uniqueOrdered(
    selected.map(
      (item) =>
        `- [引用 ${item.citationNo} 对应完整块](${item.sourceLink})`,
    ),
  );
  const suffix = `详情内容如下：\n${links.join("\n")}`;
  const trimmed = String(answer || "").trim();
  if (!trimmed) {
    return suffix;
  }
  if (links.every((link) => trimmed.includes(link))) {
    return trimmed;
  }
  return `${trimmed}\n\n${suffix}`;
}

function composeEvidenceContent(input: {
  chunkLabel: string;
  summaryText: string | null;
  mergedExcerpt: string;
  sourceLink: string;
  includeSource?: boolean;
}): string {
  const summaryBlock = input.summaryText ? `${input.chunkLabel} summary:\n${input.summaryText}` : null;
  const excerptBlock = `${input.chunkLabel} compressed excerpt:\n${input.mergedExcerpt}`;
  const oversizedNote =
    input.includeSource && input.sourceLink
      ? (
          `Note: the ${input.chunkLabel} is shown as a compressed excerpt. ` +
          `Full content is available at ${input.sourceLink}, and you may provide this link directly to the user when it is helpful.`
        )
      : null;
  return [summaryBlock, excerptBlock, oversizedNote]
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
  const materialized = items.map((item) => materializeEvidenceItem(item));
  const totalLength = materialized.reduce((sum, item) => sum + item.content.length, 0);
  if (totalLength <= budget) {
    return materialized;
  }
  const contextCompressed = materialized.map((item) => {
    let changed = false;
    const cerSegments = item.cerSegments.map((segment) => {
      if (segment.isCenter || !segment.mergedExcerpt) {
        return segment;
      }
      const maxLength = segment.referenceKind === "document_chunk" ? CER_CONTEXT_EXCERPT_TARGET : 240;
      if (segment.mergedExcerpt.length <= maxLength) {
        return segment;
      }
      changed = true;
      return {
        ...segment,
        mergedExcerpt: compressExcerptAroundKeywords(segment.mergedExcerpt, queryTokens, maxLength),
      };
    });
    return changed
      ? materializeEvidenceItem({
          ...item,
          cerSegments,
          compressionApplied: true,
        })
      : item;
  });
  const contextCompressedLength = contextCompressed.reduce((sum, item) => sum + item.content.length, 0);
  if (contextCompressedLength <= budget) {
    return contextCompressed;
  }
  const lockedLength = contextCompressed.reduce(
    (sum, item) => sum + Math.max(item.content.length - item.mergedExcerpt.length, 0),
    0,
  );
  const availableForExcerpts = Math.max(budget - lockedLength, 0);
  const totalExcerptLength = contextCompressed.reduce((sum, item) => sum + item.mergedExcerpt.length, 0);
  const compressed = contextCompressed.map((item) => {
    if (!item.mergedExcerpt) {
      return item;
    }
    const share =
      totalExcerptLength > 0
        ? item.mergedExcerpt.length / totalExcerptLength
        : 1 / Math.max(contextCompressed.length, 1);
    const targetLength = Math.max(Math.floor(availableForExcerpts * share), 80);
    if (item.mergedExcerpt.length <= targetLength) {
      return item;
    }
    const mergedExcerpt = compressExcerptAroundKeywords(item.mergedExcerpt, queryTokens, targetLength);
    return materializeEvidenceItem({
      ...item,
      mergedExcerpt,
      compressionApplied: true,
    });
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
    dropped.add(item.referenceId);
    running -= item.content.length;
  }
  return compressed.filter((item) => !dropped.has(item.referenceId));
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
    const sourceChunks = this.buildSourceChunks({
      documentId: input.documentId,
      parsedDocument: input.parsedDocument,
      imageRenderMap,
    });
    const retrievalChunks = this.buildRetrievalChunks(input.documentId, sourceChunks);
    const documentChunks = this.buildIndexedDocumentChunks(input.documentId, sourceChunks, retrievalChunks);
    this.storage.replaceDocumentChunks(input.documentId, documentChunks);
    this.storage.replaceRetrievalChunks(input.documentId, retrievalChunks);

    let embeddingsWritten = 0;
    if (input.enableEmbedding !== false) {
      embeddingsWritten = await rebuildQdrantCollection(
        input.documentId,
        documentChunks.map((chunk) => ({ id: chunk.id, text: chunk.contentMd })),
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
    const document = this.storage.getDocument(documentId);
    const retrievalUnits =
      document?.retrieval_chunking_strategy === "fixed"
        ? this.storage.listFixedRetrievalChunks(documentId)
        : this.storage.listDocumentChunks(documentId);
    return this.buildEmbeddingsForChunks(
      documentId,
      retrievalUnits.map((chunk) => ({
        id: chunk.id,
        unitText: chunk.content_md,
      })),
    );
  }

  async buildEmbeddingsForChunks(
    documentId: string,
    retrievalUnits: Array<{
      id: string;
      unitText: string;
    }>,
  ): Promise<number> {
    return rebuildQdrantCollection(
      documentId,
      retrievalUnits.map((chunk) => ({ id: chunk.id, text: chunk.unitText })),
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
  }): SourceChunkRecord[] {
    return this.buildSourceChunks(input);
  }

  createRetrievalChunks(
    documentId: string,
    documentChunks: SourceChunkRecord[],
  ): StorageRetrievalChunkRecord[] {
    return this.buildRetrievalChunks(documentId, documentChunks);
  }

  createFixedRetrievalChunks(input: {
    documentId: string;
    sourceChunks: SourceChunkRecord[];
    fixedChunkChars: number;
  }): StorageFixedRetrievalChunkRecord[] {
    return this.buildFixedRetrievalChunks(input);
  }

  createIndexedDocumentChunks(
    documentId: string,
    sourceChunks: SourceChunkRecord[],
    retrievalChunks: StorageRetrievalChunkRecord[],
    fixedRetrievalChunks: StorageFixedRetrievalChunkRecord[] = [],
  ): StorageDocumentChunkRecord[] {
    return this.buildIndexedDocumentChunks(documentId, sourceChunks, retrievalChunks, fixedRetrievalChunks);
  }

  async deleteDocumentIndex(documentId: string): Promise<void> {
    await deleteQdrantCollection(documentId);
  }

  private listTraditionalRetrievalChunks(documentId: string): TraditionalRetrievalChunkRecord[] {
    const document = this.storage.getDocument(documentId);
    if (document?.retrieval_chunking_strategy === "fixed") {
      return this.storage.listFixedRetrievalChunks(documentId);
    }
    return this.storage.listRetrievalChunks(documentId);
  }

  private getTraditionalRetrievalChunk(
    chunkId: string,
    input: { documentId?: string | null } = {},
  ): TraditionalRetrievalChunkRecord | null {
    if (input.documentId) {
      const document = this.storage.getDocument(input.documentId);
      if (document?.retrieval_chunking_strategy === "fixed") {
        return this.storage.getFixedRetrievalChunk(chunkId);
      }
      return this.storage.getRetrievalChunk(chunkId);
    }
    return this.storage.getRetrievalChunk(chunkId) ?? this.storage.getFixedRetrievalChunk(chunkId);
  }

  async query(input: {
    question: string;
    mode: TraditionalRetrievalMode;
    documentIds?: string[] | null;
    collectionIds?: string[] | null;
    keywordWeight?: number;
    semanticWeight?: number;
  }): Promise<TraditionalRagQueryResult> {
    const prepared = await this.prepareEvidence(input);
    const answered = await answerWithCitations({
      question: prepared.question,
      hits: prepared.chunks,
    });
    const selectedCitationNos =
      answered?.citations?.length
        ? answered.citations
        : answered?.answer
          ? parseCitationNumbers(answered.answer)
          : [];
    const usedChunks = this.selectChunksByCitationNos(prepared.chunks, selectedCitationNos);
    const detailChunks = toPublicDetailChunks(usedChunks);
    const rawAnswer = answered?.answer?.trim() || buildFallbackAnswer(usedChunks);
    const answer =
      !answered?.answer?.trim() || !answerContainsAnyDetailLink(rawAnswer, detailChunks)
        ? appendOversizedChunkLinks(rawAnswer, usedChunks)
        : rawAnswer;

    return {
      mode: prepared.mode,
      question: prepared.question,
      answer,
      used_chunks: toPublicChunkReferences(usedChunks),
      detail_chunks: detailChunks,
      warnings: prepared.warnings.length ? prepared.warnings : undefined,
    };
  }

  async retrieve(input: {
    question: string;
    mode: TraditionalRetrievalMode;
    documentIds?: string[] | null;
    collectionIds?: string[] | null;
    keywordWeight?: number;
    semanticWeight?: number;
  }): Promise<TraditionalRagRetrieveResult> {
    const prepared = await this.prepareEvidence(input);
    return {
      mode: prepared.mode,
      question: prepared.question,
      retrieved_chunks: toPublicChunkReferences(prepared.chunks, { includeCerDebug: true }),
      warnings: prepared.warnings.length ? prepared.warnings : undefined,
    };
  }

  async previewPrompt(input: {
    question: string;
    mode: TraditionalRetrievalMode;
    documentIds?: string[] | null;
    collectionIds?: string[] | null;
    keywordWeight?: number;
    semanticWeight?: number;
  }): Promise<TraditionalRagPromptPreviewResult> {
    const prepared = await this.prepareEvidence(input);
    const requestPayload = buildAnswerRequestPayload({
      question: prepared.question,
      hits: prepared.chunks,
      variant: "stream",
    });
    const messages = requestPayload.messages;
    return {
      mode: prepared.mode,
      question: prepared.question,
      prompt_variant: "stream",
      model: requestPayload.model,
      temperature: requestPayload.temperature,
      messages,
      system_prompt: messages.find((message) => message.role === "system")?.content ?? "",
      user_prompt: messages.find((message) => message.role === "user")?.content ?? "",
      request_body_json: JSON.stringify(requestPayload, null, 2),
      warnings: prepared.warnings.length ? prepared.warnings : undefined,
    };
  }

  async *streamQuery(input: {
    question: string;
    mode: TraditionalRetrievalMode;
    documentIds?: string[] | null;
    collectionIds?: string[] | null;
    keywordWeight?: number;
    semanticWeight?: number;
  }): AsyncIterable<
    | { type: "start"; question: string; mode: TraditionalRetrievalMode; retrieved_count: number }
    | { type: "answer_delta"; delta_text: string }
    | {
        type: "complete";
        answer: string;
        used_chunks: TraditionalRagChunkReference[];
        detail_chunks: TraditionalRagDetailChunk[];
        warnings?: string[];
      }
  > {
    const prepared = await this.prepareEvidence(input);
    yield {
      type: "start",
      question: prepared.question,
      mode: prepared.mode,
      retrieved_count: prepared.chunks.length,
    };

    let answer = "";
    for await (const delta of streamAnswerFromModel({
      question: prepared.question,
      hits: prepared.chunks,
    })) {
      if (!delta) {
        continue;
      }
      answer += delta;
      yield {
        type: "answer_delta",
        delta_text: delta,
      };
    }

    const trimmedAnswer = answer.trim() || buildFallbackAnswer(prepared.chunks);
    const usedChunks = this.selectChunksByCitationNos(prepared.chunks, parseCitationNumbers(trimmedAnswer));
    const detailChunks = toPublicDetailChunks(usedChunks);
    const finalAnswer =
      !answer.trim() || !answerContainsAnyDetailLink(trimmedAnswer, detailChunks)
        ? appendOversizedChunkLinks(trimmedAnswer, usedChunks)
        : trimmedAnswer;
    const appendedSuffix = finalAnswer.slice(trimmedAnswer.length);
    if (appendedSuffix) {
      yield {
        type: "answer_delta",
        delta_text: appendedSuffix,
      };
    }
    yield {
      type: "complete",
      answer: finalAnswer,
      used_chunks: toPublicChunkReferences(usedChunks),
      detail_chunks: detailChunks,
      warnings: prepared.warnings.length ? prepared.warnings : undefined,
    };
  }

  private async prepareEvidence(input: {
    question: string;
    mode: TraditionalRetrievalMode;
    documentIds?: string[] | null;
    collectionIds?: string[] | null;
    keywordWeight?: number;
    semanticWeight?: number;
  }): Promise<PreparedEvidenceSet> {
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
    const fixedDocuments = scope.documents.filter((document) => document.retrieval_chunking_strategy === "fixed");
    const fixedDocumentIds = fixedDocuments.map((document) => document.id);
    const smallToBigDocumentIds = scope.documents
      .filter((document) => document.retrieval_chunking_strategy !== "fixed")
      .map((document) => document.id);
    if (input.mode === "semantic" && documentsWithEmbeddings.length === 0) {
      throw new Error("Selected documents do not have embeddings yet.");
    }
    if (missingEmbeddingDocuments.length > 0 && input.mode !== "keyword") {
      warnings.push(
        `Skipped semantic retrieval for ${missingEmbeddingDocuments.length} document(s) without embeddings.`,
      );
    }

    const keywordHits: RankedChunkHit[] = [];
    if (input.mode !== "semantic") {
      if (smallToBigDocumentIds.length > 0) {
        keywordHits.push(
          ...this.storage.keywordSearchDocumentChunks({
            query: toFtsQuery(question),
            documentIds: smallToBigDocumentIds,
            limit: 24,
          }).map((item) => ({
            retrievalUnitId: item.document_chunk_id,
            documentId: item.document_id,
            sourceDocumentChunkId: item.document_chunk_id,
            referenceRetrievalChunkId: item.reference_retrieval_chunk_id,
            ordinal: item.ordinal,
            score: item.score,
            unitText: item.content_md,
            isSplitFromOversized: item.is_split_from_oversized,
          })),
        );
      }
      if (fixedDocumentIds.length > 0) {
        keywordHits.push(
          ...this.storage.keywordSearchFixedRetrievalChunks({
            query: toFtsQuery(question),
            documentIds: fixedDocumentIds,
            limit: 24,
          }).map((item) => ({
            retrievalUnitId: item.retrieval_chunk_id,
            documentId: item.document_id,
            sourceDocumentChunkId:
              parseJsonArray<string>(item.source_document_chunk_ids_json, [item.retrieval_chunk_id])[0] ??
              item.retrieval_chunk_id,
            referenceRetrievalChunkId: item.retrieval_chunk_id,
            ordinal: item.ordinal,
            score: item.score,
            unitText: item.content_md,
            isSplitFromOversized: false,
          })),
        );
      }
    }

    const semanticHits: RankedChunkHit[] = [];
    if (input.mode !== "keyword" && documentsWithEmbeddings.length > 0) {
      const vector = await createEmbedding(question);
      if (vector) {
        for (const document of scope.documents) {
          if (!documentIdsWithEmbeddings.has(document.id)) {
            continue;
          }
          const matches = await searchQdrant(document.id, vector, 8);
          const retrievalMap =
            document.retrieval_chunking_strategy === "fixed"
              ? new Map(this.storage.listFixedRetrievalChunks(document.id).map((chunk) => [chunk.id, chunk] as const))
              : new Map(this.storage.listDocumentChunks(document.id).map((chunk) => [chunk.id, chunk] as const));
          for (const match of matches) {
            const chunk = retrievalMap.get(match.id);
            if (!chunk) {
              continue;
            }
            semanticHits.push({
              retrievalUnitId: chunk.id,
              documentId: chunk.document_id,
              sourceDocumentChunkId:
                "reference_retrieval_chunk_id" in chunk
                  ? chunk.id
                  : parseJsonArray<string>(chunk.source_document_chunk_ids_json, [chunk.id])[0] ?? chunk.id,
              referenceRetrievalChunkId:
                "reference_retrieval_chunk_id" in chunk
                  ? chunk.reference_retrieval_chunk_id
                  : chunk.id,
              ordinal: chunk.ordinal,
              score: match.score,
              unitText: chunk.content_md,
              isSplitFromOversized:
                "is_split_from_oversized" in chunk ? chunk.is_split_from_oversized : false,
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
      this.applyRelevantSegmentExtraction(
        this.applyContextEnrichedRetrieval(this.groupHitsBySource(combined).slice(0, 8), question),
        question,
      ),
      question,
      getEvidenceCharBudget(),
    ).map((item, index) => {
      const citationNo = index + 1;
      const cerSegments = item.cerSegments.map((segment) => ({
        ...segment,
        citationNo,
      }));
      const baseContent = composeEvidenceContent({
        chunkLabel: `chunk[${citationNo}]`,
        summaryText: item.summaryText,
        mergedExcerpt: item.mergedExcerpt,
        sourceLink: item.sourceLink,
        includeSource: item.hasOversizedSplitChild,
      });
      const cerLabels = buildCerSegmentLabels(citationNo, cerSegments);
      const enrichedContent = cerSegments
        .map((segment, segmentIndex) => composeSegmentContent(segment, cerLabels[segmentIndex] || `chunk[${citationNo}]`))
        .filter((value) => String(value || "").trim())
        .join("\n\n");
      return {
        ...item,
        citationNo,
        cerSegments,
        baseContent,
        enrichedContent,
        content: enrichedContent || baseContent,
        sourceLocator: item.sourceLocator || sourceLocatorFromPages(item.pageNos),
      };
    });

    return {
      mode: input.mode,
      question,
      warnings,
      chunks: groupedEvidence,
    };
  }

  private selectChunksByCitationNos(
    chunks: GroupedEvidence[],
    citationNos: number[],
  ): GroupedEvidence[] {
    if (citationNos.length === 0) {
      return chunks;
    }
    const byCitation = new Map(chunks.map((item) => [item.citationNo, item] as const));
    const selected = citationNos.map((citationNo) => byCitation.get(citationNo)).filter(Boolean) as GroupedEvidence[];
    return selected.length > 0 ? selected : chunks;
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
        const key = item.retrievalUnitId;
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

  private applyRelevantSegmentExtraction(items: GroupedEvidence[], question: string): GroupedEvidence[] {
    const queryTokens = extractQueryTokens(question);
    if (queryTokens.length === 0) {
      return items;
    }
    return items.map((item) => {
      let changed = false;
      const cerSegments = item.cerSegments.map((segment) => {
        if (segment.isCenter || !segment.mergedExcerpt.trim()) {
          return segment;
        }
        const mergedExcerpt = extractRelevantSegment(segment, queryTokens);
        if (mergedExcerpt === segment.mergedExcerpt) {
          return segment;
        }
        changed = true;
        return {
          ...segment,
          mergedExcerpt,
        };
      });
      if (!changed) {
        return item;
      }
      return materializeEvidenceItem({
        ...item,
        cerSegments,
        compressionApplied: true,
      });
    });
  }

  private applyContextEnrichedRetrieval(
    items: GroupedEvidence[],
    _question: string,
  ): GroupedEvidence[] {
    const documentIds = uniqueOrdered(items.map((item) => item.documentId));
    const documentChunksByDocument = new Map<string, StoredDocumentChunk[]>();
    const documentChunkById = new Map<string, StoredDocumentChunk>();
    const retrievalChunksByDocument = new Map<string, TraditionalRetrievalChunkRecord[]>();
    const retrievalChunkById = new Map<string, TraditionalRetrievalChunkRecord>();
    const retrievalRangesById = new Map<string, { start: number; end: number }>();
    const retrievalStrategyByDocument = new Map<string, "small_to_big" | "fixed">();

    for (const documentId of documentIds) {
      const documentChunks = this.storage
        .listDocumentChunks(documentId)
        .sort((left, right) => left.document_index - right.document_index);
      documentChunksByDocument.set(documentId, documentChunks);
      for (const chunk of documentChunks) {
        documentChunkById.set(chunk.id, chunk);
      }

      const retrievalChunks = this.listTraditionalRetrievalChunks(documentId).sort(
        (left, right) => left.ordinal - right.ordinal,
      );
      const document = this.storage.getDocument(documentId);
      retrievalStrategyByDocument.set(documentId, document?.retrieval_chunking_strategy === "fixed" ? "fixed" : "small_to_big");
      retrievalChunksByDocument.set(documentId, retrievalChunks);
      for (const chunk of retrievalChunks) {
        retrievalChunkById.set(chunk.id, chunk);
        const range = this.resolveRetrievalChunkRange(chunk, documentChunkById);
        if (range) {
          retrievalRangesById.set(chunk.id, range);
        }
      }
    }

    return items.map((item) => {
      const segments = new Map<string, EvidenceSegment>();
      const addSegment = (segment: EvidenceSegment | null) => {
        if (!segment) {
          return;
        }
        const key = `${segment.referenceKind}:${segment.referenceId}`;
        if (!segments.has(key)) {
          segments.set(key, segment);
        }
      };

      addSegment(this.createCenterEvidenceSegment(item, documentChunkById, retrievalRangesById));

      if (item.referenceKind === "retrieval_chunk") {
        const retrievalChunk = retrievalChunkById.get(item.referenceId);
        const retrievalChunks = retrievalChunksByDocument.get(item.documentId) ?? [];
        if (retrievalChunk) {
          const currentIndex = retrievalChunks.findIndex((chunk) => chunk.id === retrievalChunk.id);
          if (retrievalStrategyByDocument.get(item.documentId) === "fixed") {
            for (let index = Math.max(0, currentIndex - 2); index <= Math.min(retrievalChunks.length - 1, currentIndex + 2); index += 1) {
              if (index === currentIndex) {
                continue;
              }
              addSegment(this.createRetrievalChunkContextSegment(retrievalChunks[index] ?? null, documentChunkById));
            }
          } else {
            const centerSegment = [...segments.values()].find((segment) => segment.isCenter) ?? null;
            const onlyKeepNext = centerSegment ? isHeadingLikeShortCenterSegment(centerSegment) : false;
            if (!onlyKeepNext) {
              addSegment(this.selectRetrievalChunkCerSide(retrievalChunks, currentIndex, "left", documentChunkById));
            }
            addSegment(this.selectRetrievalChunkCerSide(retrievalChunks, currentIndex, "right", documentChunkById));
          }
        }
      } else {
        const sourceDocumentChunk = documentChunkById.get(item.referenceId);
        const retrievalChunks = retrievalChunksByDocument.get(item.documentId) ?? [];
        if (sourceDocumentChunk) {
          addSegment(
            this.findNearestRetrievalChunkSegment(
              retrievalChunks,
              sourceDocumentChunk.document_index,
              "left",
              documentChunkById,
            ),
          );
          addSegment(
            this.findNearestRetrievalChunkSegment(
              retrievalChunks,
              sourceDocumentChunk.document_index,
              "right",
              documentChunkById,
            ),
          );
        }
      }

      const cerSegments = [...segments.values()].sort(
        (left, right) =>
          left.sortOrder - right.sortOrder ||
          Number(right.isCenter) - Number(left.isCenter) ||
          left.referenceId.localeCompare(right.referenceId),
      );
      return materializeEvidenceItem({
        ...item,
        cerSegments,
        compressionApplied: item.compressionApplied,
      });
    });
  }

  private createCenterEvidenceSegment(
    item: GroupedEvidence,
    documentChunkById: Map<string, StoredDocumentChunk>,
    retrievalRangesById: Map<string, { start: number; end: number }>,
  ): EvidenceSegment {
    const sortOrder =
      item.referenceKind === "document_chunk"
        ? documentChunkById.get(item.referenceId)?.document_index ?? 0
        : retrievalRangesById.get(item.referenceId)?.start ??
          Math.min(
            ...item.documentChunkIds
              .map((id) => documentChunkById.get(id)?.document_index ?? Number.POSITIVE_INFINITY)
              .filter((value) => Number.isFinite(value)),
          );
    return {
      citationNo: item.citationNo,
      referenceId: item.referenceId,
      referenceKind: item.referenceKind,
      summaryText: item.summaryText,
      mergedExcerpt: item.mergedExcerpt,
      sourceLink: item.sourceLink,
      sourceLocator: item.sourceLocator,
      pageNos: item.pageNos,
      sortOrder: Number.isFinite(sortOrder) ? sortOrder : 0,
      blockType:
        item.referenceKind === "document_chunk"
          ? documentChunkById.get(item.referenceId)?.block_type ?? null
          : null,
      isOversized: item.hasOversizedSplitChild,
      isCenter: true,
    };
  }

  private createRetrievalChunkContextSegment(
    chunk: TraditionalRetrievalChunkRecord | null,
    documentChunkById: Map<string, StoredDocumentChunk>,
  ): EvidenceSegment | null {
    if (!chunk) {
      return null;
    }
    const range = this.resolveRetrievalChunkRange(chunk, documentChunkById);
    return {
      citationNo: 0,
      referenceId: chunk.id,
      referenceKind: "retrieval_chunk",
      summaryText: chunk.summary_text,
      mergedExcerpt: normalizeWhitespace(chunk.content_md),
      sourceLink: toAbsoluteServerUrl(`/api/retrieval-chunks/${chunk.id}/content`),
      sourceLocator: chunk.source_locator || sourceLocatorFromPages(parseJsonArray<number>(chunk.page_nos_json, [])),
      pageNos: parseJsonArray<number>(chunk.page_nos_json, []),
      sortOrder: range?.start ?? chunk.ordinal,
      blockType: this.resolveRetrievalChunkBlockType(chunk, documentChunkById),
      isOversized: chunk.size_class === "oversized",
      isCenter: false,
    };
  }

  private findNearestOversizedRetrievalChunkSegment(
    chunks: TraditionalRetrievalChunkRecord[],
    anchorIndex: number,
    direction: "left" | "right",
    documentChunkById: Map<string, StoredDocumentChunk>,
  ): EvidenceSegment | null {
    if (anchorIndex < 0) {
      return null;
    }
    if (direction === "left") {
      for (let index = anchorIndex - 1; index >= 0; index -= 1) {
        const chunk = chunks[index];
        if (chunk?.size_class === "oversized") {
          return this.createRetrievalChunkContextSegment(chunk, documentChunkById);
        }
      }
      return null;
    }
    for (let index = anchorIndex + 1; index < chunks.length; index += 1) {
      const chunk = chunks[index];
      if (chunk?.size_class === "oversized") {
        return this.createRetrievalChunkContextSegment(chunk, documentChunkById);
      }
    }
    return null;
  }

  private selectRetrievalChunkCerSide(
    chunks: TraditionalRetrievalChunkRecord[],
    anchorIndex: number,
    direction: "left" | "right",
    documentChunkById: Map<string, StoredDocumentChunk>,
  ): EvidenceSegment | null {
    const oversized = this.findNearestOversizedRetrievalChunkSegment(chunks, anchorIndex, direction, documentChunkById);
    if (oversized) {
      return oversized;
    }
    const adjacentIndex = direction === "left" ? anchorIndex - 1 : anchorIndex + 1;
    return this.createRetrievalChunkContextSegment(chunks[adjacentIndex] ?? null, documentChunkById);
  }

  private createAdjacentOversizedDocumentSegment(
    chunk: StoredDocumentChunk | null,
  ): EvidenceSegment | null {
    if (!chunk || chunk.size_class !== "oversized") {
      return null;
    }
    return {
      citationNo: 0,
      referenceId: chunk.id,
      referenceKind: "document_chunk",
      summaryText: chunk.summary_text,
      mergedExcerpt: normalizeWhitespace(chunk.content_md),
      sourceLink: toAbsoluteServerUrl(`/api/document-chunks/${chunk.id}/content`),
      sourceLocator: sourceLocatorFromPages(parseJsonArray<number>(chunk.merged_page_nos_json, [chunk.page_no])),
      pageNos: parseJsonArray<number>(chunk.merged_page_nos_json, [chunk.page_no]),
      sortOrder: chunk.document_index,
      blockType: chunk.block_type,
      isOversized: true,
      isCenter: false,
    };
  }

  private findAdjacentDocumentChunk(
    chunks: StoredDocumentChunk[],
    anchorIndex: number,
    direction: "left" | "right",
  ): StoredDocumentChunk | null {
    if (direction === "left") {
      for (let index = chunks.length - 1; index >= 0; index -= 1) {
        const chunk = chunks[index];
        if (chunk && chunk.document_index < anchorIndex) {
          return chunk;
        }
      }
      return null;
    }
    for (const chunk of chunks) {
      if (chunk.document_index > anchorIndex) {
        return chunk;
      }
    }
    return null;
  }

  private findNearestRetrievalChunkSegment(
    chunks: TraditionalRetrievalChunkRecord[],
    anchorDocumentIndex: number,
    direction: "left" | "right",
    documentChunkById: Map<string, StoredDocumentChunk>,
  ): EvidenceSegment | null {
    const ranges = chunks
      .map((chunk) => ({
        chunk,
        range: this.resolveRetrievalChunkRange(chunk, documentChunkById),
      }))
      .filter((item) => item.range !== null) as Array<{
      chunk: TraditionalRetrievalChunkRecord;
      range: { start: number; end: number };
    }>;
    if (direction === "left") {
      const candidate = ranges
        .filter((item) => item.range.end < anchorDocumentIndex)
        .sort((left, right) => right.range.end - left.range.end)[0];
      return this.createRetrievalChunkContextSegment(candidate?.chunk ?? null, documentChunkById);
    }
    const candidate = ranges
      .filter((item) => item.range.start > anchorDocumentIndex)
      .sort((left, right) => left.range.start - right.range.start)[0];
    return this.createRetrievalChunkContextSegment(candidate?.chunk ?? null, documentChunkById);
  }

  private resolveRetrievalChunkRange(
    chunk: TraditionalRetrievalChunkRecord,
    documentChunkById: Map<string, StoredDocumentChunk>,
  ): { start: number; end: number } | null {
    const indexes = parseJsonArray<string>(chunk.source_document_chunk_ids_json, [])
      .map((id) => documentChunkById.get(id)?.document_index ?? null)
      .filter((value): value is number => Number.isFinite(value));
    if (indexes.length === 0) {
      return null;
    }
    return {
      start: Math.min(...indexes),
      end: Math.max(...indexes),
    };
  }

  private resolveRetrievalChunkBlockType(
    chunk: TraditionalRetrievalChunkRecord,
    documentChunkById: Map<string, StoredDocumentChunk>,
  ): string | null {
    const blockTypes = uniqueOrdered(
      parseJsonArray<string>(chunk.source_document_chunk_ids_json, [])
        .map((id) => documentChunkById.get(id)?.block_type ?? null)
        .filter((value): value is string => Boolean(value)),
    );
    return blockTypes.length === 1 ? blockTypes[0] : blockTypes[0] ?? null;
  }

  private groupHitsBySource(hits: RankedChunkHit[]): GroupedEvidence[] {
    const grouped = new Map<string, RankedChunkHit[]>();
    for (const hit of hits) {
      const key = hit.referenceRetrievalChunkId
        ? `retrieval:${hit.referenceRetrievalChunkId}`
        : hit.isSplitFromOversized
          ? `document:${hit.sourceDocumentChunkId}`
          : "";
      if (!key || key === "retrieval:") {
        continue;
      }
      const items = grouped.get(key) ?? [];
      items.push(hit);
      grouped.set(key, items);
    }
    const mapped = [...grouped.entries()].map(([groupKey, items]) => {
      const orderedHits = [...items].sort((left, right) => left.ordinal - right.ordinal);
      const documentKeyPrefix = "document:";
      if (groupKey.startsWith(documentKeyPrefix)) {
        const sourceDocumentChunkId = groupKey.slice(documentKeyPrefix.length);
        const source = this.storage.getDocumentChunk(sourceDocumentChunkId);
        if (!source) {
          return null;
        }
        const document = this.storage.getDocument(source.document_id);
        const mergedExcerpt = normalizeWhitespace(orderedHits.map((item) => item.unitText).join("\n\n"));
        const sourceLink = toAbsoluteServerUrl(`/api/document-chunks/${source.id}/content`);
        return materializeEvidenceItem({
          citationNo: 0,
          referenceId: source.id,
          referenceKind: "document_chunk",
          documentId: source.document_id,
          documentName:
            document?.original_filename ||
            document?.relative_path ||
            document?.absolute_path ||
            source.document_id,
          sourceLocator: sourceLocatorFromPages(parseJsonArray<number>(source.merged_page_nos_json, [source.page_no])),
          hasOversizedSplitChild: true,
          documentChunkIds: [source.id],
          retrievalUnitIds: orderedHits.map((item) => item.retrievalUnitId),
          pageNos: parseJsonArray<number>(source.merged_page_nos_json, [source.page_no]),
          bboxes: parseJsonArray<[number, number, number, number]>(source.merged_bboxes_json, [
            JSON.parse(source.bbox_json) as [number, number, number, number],
          ]),
          score: Math.max(...orderedHits.map((item) => item.score)),
          summaryText: source.summary_text,
          sourceLink,
          mergedExcerpt,
          baseContent: "",
          enrichedContent: "",
          content: "",
          cerApplied: false,
          cerContextRefs: [],
          cerSegments: [
            {
              citationNo: 0,
              referenceId: source.id,
              referenceKind: "document_chunk",
              summaryText: source.summary_text,
              mergedExcerpt,
              sourceLink,
              sourceLocator: sourceLocatorFromPages(
                parseJsonArray<number>(source.merged_page_nos_json, [source.page_no]),
              ),
              pageNos: parseJsonArray<number>(source.merged_page_nos_json, [source.page_no]),
              sortOrder: source.document_index,
              blockType: source.block_type,
              isOversized: true,
              isCenter: true,
            },
          ],
          compressionApplied: false as boolean,
        } as GroupedEvidence);
      }

      const retrievalChunkId = groupKey.slice("retrieval:".length);
      const reference = this.getTraditionalRetrievalChunk(retrievalChunkId, { documentId: items[0]?.documentId ?? null });
      if (!reference) {
        return null;
      }
      const document = this.storage.getDocument(reference.document_id);
      const pageNos = parseJsonArray<number>(reference.page_nos_json, []);
      const sourceDocumentChunkIds = parseJsonArray<string>(reference.source_document_chunk_ids_json, []);
      const sourceChunkBlockType =
        sourceDocumentChunkIds
          .map((id) => this.storage.getDocumentChunk(id)?.block_type ?? null)
          .find(Boolean) ?? null;
      const mergedExcerpt = normalizeWhitespace(reference.content_md);
      const sourceLink = toAbsoluteServerUrl(`/api/retrieval-chunks/${reference.id}/content`);
      const hasOversizedSplitChild =
        reference.size_class === "oversized" || orderedHits.some((item) => item.isSplitFromOversized);
      return materializeEvidenceItem({
        citationNo: 0,
        referenceId: reference.id,
        referenceKind: "retrieval_chunk",
        documentId: reference.document_id,
        documentName:
          document?.original_filename ||
          document?.relative_path ||
          document?.absolute_path ||
          reference.document_id,
        sourceLocator: reference.source_locator || sourceLocatorFromPages(pageNos),
        hasOversizedSplitChild,
        documentChunkIds: sourceDocumentChunkIds,
        retrievalUnitIds: orderedHits.map((item) => item.retrievalUnitId),
        pageNos,
        bboxes: parseJsonArray<[number, number, number, number]>(reference.bboxes_json, []),
        score: Math.max(...orderedHits.map((item) => item.score)),
        summaryText: reference.summary_text,
        sourceLink,
        mergedExcerpt,
        baseContent: "",
        enrichedContent: "",
        content: "",
        cerApplied: false,
        cerContextRefs: [],
        cerSegments: [
          {
            citationNo: 0,
            referenceId: reference.id,
            referenceKind: "retrieval_chunk",
            summaryText: reference.summary_text,
            mergedExcerpt,
            sourceLink,
            sourceLocator: reference.source_locator || sourceLocatorFromPages(pageNos),
            pageNos,
            sortOrder: reference.ordinal,
            blockType: sourceChunkBlockType,
            isOversized: hasOversizedSplitChild,
            isCenter: true,
          },
        ],
        compressionApplied: false as boolean,
      } as GroupedEvidence);
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
    const sameHashOnPageCounts = new Map<string, number>();
    for (const { image } of allParsedImages) {
      const key = `${image.page_no}:${image.image_hash}`;
      sameHashOnPageCounts.set(key, (sameHashOnPageCounts.get(key) ?? 0) + 1);
    }
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
        const screeningContext = {
          sameHashOnPageCount: sameHashOnPageCounts.get(`${image.page_no}:${image.image_hash}`) ?? 1,
        };
        interferenceScore = Number(inspection?.interference_score ?? 0);
        hasText = Boolean(inspection?.has_text);
        const shouldDrop = inspection
          ? evaluateImageSemanticCandidateInspection(inspection, screeningContext).shouldDrop
          : false;
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
            retrieval_summary: cached.summary,
            detail_markdown: cached.detail_markdown,
            entities: parseJsonArray<string>(cached.entities_json, []),
            keywords: parseJsonArray<string>(cached.keywords_json, []),
            qa_hints: parseJsonArray<string>(cached.qa_hints_json, []),
            detail_truncated: cached.detail_truncated,
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
              summary: semanticPayload.retrieval_summary ?? null,
              detailMarkdown: semanticPayload.detail_markdown ?? null,
              detailTruncated: semanticPayload.detail_truncated ?? null,
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

      const renderedSemantic = renderImageSemantic({
        payload: semanticPayload,
        accessibleUrl,
      });
      renderMap.set(image.image_hash, {
        dropped: false,
        markdown: renderedSemantic.shortMarkdown || `![image](${accessibleUrl})`,
      });
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
        semanticText: renderedSemantic.semanticText,
        semanticDetailText: renderedSemantic.semanticDetailText,
        semanticModel: semanticPayload && input.enableImageSemantic ? resolveVisionConfig().modelName : null,
      });
    }
    this.storage.upsertImageSemantics(rows);
    return renderMap;
  }

  private buildSourceChunks(input: {
    documentId: string;
    parsedDocument: ParsedDocument;
    imageRenderMap: Map<string, ImageChunkRendering>;
  }): SourceChunkRecord[] {
    const perPage = input.parsedDocument.units.map((unit) => ({
      unitNo: unit.unit_no,
      blocks: this.renderUnitBlocks(unit, input.imageRenderMap),
    }));
    const records: SourceChunkRecord[] = [];
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
          mergedContent = mergeContinuedBlockMarkdown(mergedContent, nextHead.markdown, block.block_type);
          mergedPageNos.push(nextPage.unitNo);
          mergedBboxes.push(nextHead.bbox);
          nextPage.blocks = nextPage.blocks.slice(1);
          scanPageIndex += 1;
        }
        const sizeClass = classifySize(mergedContent);
        const record: SourceChunkRecord = {
          id: stableId("schunk", `${input.documentId}:${documentIndex}:${page.unitNo}:${block.index}:${mergedContent}`),
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
    documentChunks: SourceChunkRecord[],
  ): StorageRetrievalChunkRecord[] {
    const thresholds = getChunkThresholds();
    const records: StorageRetrievalChunkRecord[] = [];
    let ordinal = 0;
    let pendingSmall: SourceChunkRecord[] = [];
    const createContextChunk = (
      chunks: SourceChunkRecord[],
      contentMd: string,
      input: { sizeClass?: "small" | "normal" | "oversized"; summaryText?: string | null } = {},
    ): StorageRetrievalChunkRecord => {
      const sourceDocumentChunkIds = chunks.map((item) => item.id);
      const pageNos = uniqueOrdered(
        chunks.flatMap((item) => parseJsonArray<number>(item.mergedPageNosJson ?? "[]", [item.pageNo])),
      );
      const bboxes = chunks.flatMap((item) =>
        parseJsonArray<[number, number, number, number]>(
          item.mergedBboxesJson ?? "[]",
          [JSON.parse(item.bboxJson) as [number, number, number, number]],
        ),
      );
      return {
        id: stableId("rchunk", `${documentId}:${ordinal}:${sourceDocumentChunkIds.join(",")}:${contentMd}`),
        documentId,
        ordinal,
        contentMd,
        sizeClass: input.sizeClass ?? classifySize(contentMd),
        summaryText: input.summaryText ?? null,
        sourceDocumentChunkIdsJson: JSON.stringify(sourceDocumentChunkIds),
        pageNosJson: JSON.stringify(pageNos),
        sourceLocator: sourceLocatorFromPages(pageNos),
        bboxesJson: JSON.stringify(bboxes),
      };
    };
    const flushPending = () => {
      if (pendingSmall.length === 0) {
        return;
      }
      const text = normalizeWhitespace(pendingSmall.map((item) => item.contentMd).join("\n\n"));
      records.push(createContextChunk(pendingSmall, text));
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
        records.push(createContextChunk([chunk], chunk.contentMd));
        ordinal += 1;
        continue;
      }
      if (chunk.sizeClass === "oversized") {
        records.push(
          createContextChunk([chunk], chunk.contentMd, {
            sizeClass: "oversized",
            summaryText: chunk.summaryText,
          }),
        );
        ordinal += 1;
      }
    }
    flushPending();
    return records;
  }

  private buildFixedRetrievalChunks(input: {
    documentId: string;
    sourceChunks: SourceChunkRecord[];
    fixedChunkChars: number;
  }): StorageFixedRetrievalChunkRecord[] {
    const maxChars = Math.max(Number(input.fixedChunkChars || 0), 1);
    const pieces: FixedChunkPiece[] = [];
    for (const chunk of input.sourceChunks) {
      const contentParts = splitFixedChunkTextSmart(chunk.contentMd, maxChars);
      for (const part of contentParts) {
        pieces.push({
          sourceChunkId: chunk.id,
          pageNo: chunk.pageNo,
          contentMd: part,
          blockType: chunk.blockType,
          mergedPageNosJson: chunk.mergedPageNosJson,
          mergedBboxesJson: chunk.mergedBboxesJson,
          bboxJson: chunk.bboxJson,
        });
      }
    }

    const records: StorageFixedRetrievalChunkRecord[] = [];
    let ordinal = 0;
    let pendingPieces: FixedChunkPiece[] = [];
    const flushPending = () => {
      if (pendingPieces.length === 0) {
        return;
      }
      const contentMd = normalizeWhitespace(pendingPieces.map((piece) => piece.contentMd).join("\n\n"));
      const sourceDocumentChunkIds = uniqueOrdered(pendingPieces.map((piece) => piece.sourceChunkId));
      const pageNos = uniqueOrdered(
        pendingPieces.flatMap((piece) => parseJsonArray<number>(piece.mergedPageNosJson, [piece.pageNo])),
      );
      const bboxes = pendingPieces.flatMap((piece) =>
        parseJsonArray<[number, number, number, number]>(
          piece.mergedBboxesJson,
          [JSON.parse(piece.bboxJson) as [number, number, number, number]],
        ),
      );
      records.push({
        id: stableId("frchunk", `${input.documentId}:${ordinal}:${sourceDocumentChunkIds.join(",")}:${contentMd}`),
        documentId: input.documentId,
        ordinal,
        contentMd,
        sizeClass: classifySize(contentMd),
        summaryText: null,
        sourceDocumentChunkIdsJson: JSON.stringify(sourceDocumentChunkIds),
        pageNosJson: JSON.stringify(pageNos),
        sourceLocator: sourceLocatorFromPages(pageNos),
        bboxesJson: JSON.stringify(bboxes),
      });
      ordinal += 1;
      pendingPieces = [];
    };

    for (const piece of pieces) {
      const pieceText = normalizeWhitespace(piece.contentMd);
      if (!pieceText) {
        continue;
      }
      const currentText = normalizeWhitespace(pendingPieces.map((item) => item.contentMd).join("\n\n"));
      const candidate = normalizeWhitespace([currentText, pieceText].filter(Boolean).join("\n\n"));
      if (candidate.length > maxChars && pendingPieces.length > 0) {
        flushPending();
      }
      pendingPieces.push(piece);
      const updatedText = normalizeWhitespace(pendingPieces.map((item) => item.contentMd).join("\n\n"));
      if (updatedText.length >= maxChars) {
        flushPending();
      }
    }
    flushPending();
    return records;
  }

  private buildIndexedDocumentChunks(
    documentId: string,
    sourceChunks: SourceChunkRecord[],
    retrievalChunks: StorageRetrievalChunkRecord[],
    fixedRetrievalChunks: StorageFixedRetrievalChunkRecord[] = [],
  ): StorageDocumentChunkRecord[] {
    const thresholds = getChunkThresholds();
    const referenceBySourceChunkId = new Map<string, string>();
    for (const chunk of retrievalChunks) {
      for (const sourceChunkId of parseJsonArray<string>(chunk.sourceDocumentChunkIdsJson, [])) {
        referenceBySourceChunkId.set(sourceChunkId, chunk.id);
      }
    }

    const records: StorageDocumentChunkRecord[] = [];
    const unitIdsBySourceChunkId = new Map<string, string[]>();
    let ordinal = 0;
    for (const chunk of sourceChunks) {
      const referenceRetrievalChunkId = referenceBySourceChunkId.get(chunk.id) ?? null;
      if (chunk.sizeClass === "oversized") {
        const parts = splitOversized(chunk.contentMd, chunk.blockType);
        for (let splitIndex = 0; splitIndex < parts.length; splitIndex += 1) {
          const part = parts[splitIndex]!;
          records.push({
            id: stableId("dchunk", `${documentId}:${chunk.id}:${splitIndex}:${part}`),
            documentId,
            ordinal,
            referenceRetrievalChunkId,
            pageNo: chunk.pageNo,
            documentIndex: chunk.documentIndex,
            pageIndex: chunk.pageIndex,
            blockType: chunk.blockType,
            bboxJson: chunk.bboxJson,
            contentMd: part,
            sizeClass: part.length <= thresholds.smallMaxChars ? "small" : "normal",
            summaryText: chunk.summaryText,
            isSplitFromOversized: true,
            splitIndex,
            splitCount: parts.length,
            mergedPageNosJson: chunk.mergedPageNosJson,
            mergedBboxesJson: chunk.mergedBboxesJson,
          });
          const unitIds = unitIdsBySourceChunkId.get(chunk.id) ?? [];
          unitIds.push(records[records.length - 1]!.id);
          unitIdsBySourceChunkId.set(chunk.id, unitIds);
          ordinal += 1;
        }
        continue;
      }
      records.push({
        id: stableId("dchunk", `${documentId}:${chunk.id}:${ordinal}:${chunk.contentMd}`),
        documentId,
        referenceRetrievalChunkId,
        ordinal,
        pageNo: chunk.pageNo,
        documentIndex: chunk.documentIndex,
        pageIndex: chunk.pageIndex,
        blockType: chunk.blockType,
        bboxJson: chunk.bboxJson,
        contentMd: chunk.contentMd,
        sizeClass: chunk.sizeClass,
        summaryText: chunk.summaryText,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: chunk.mergedPageNosJson,
        mergedBboxesJson: chunk.mergedBboxesJson,
      });
      const unitIds = unitIdsBySourceChunkId.get(chunk.id) ?? [];
      unitIds.push(records[records.length - 1]!.id);
      unitIdsBySourceChunkId.set(chunk.id, unitIds);
      ordinal += 1;
    }
    const remapChunkSourceIds = (
      chunks: Array<{ sourceDocumentChunkIdsJson: string }>,
    ) => {
      for (const chunk of chunks) {
        const sourceChunkIds = parseJsonArray<string>(chunk.sourceDocumentChunkIdsJson, []);
        const unitIds = sourceChunkIds.flatMap((sourceChunkId) => unitIdsBySourceChunkId.get(sourceChunkId) ?? []);
        chunk.sourceDocumentChunkIdsJson = JSON.stringify(unitIds);
      }
    };
    remapChunkSourceIds(retrievalChunks);
    remapChunkSourceIds(fixedRetrievalChunks);
    return records;
  }
}

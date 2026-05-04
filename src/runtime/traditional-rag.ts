/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
*/

import { Buffer } from "node:buffer";
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
import {
  buildFixedRetrievalChunks,
  buildIndexedDocumentChunks,
  buildRetrievalChunks,
  buildSourceChunks,
} from "./traditional-rag-chunking.js";
import {
  answerContainsAnyDetailLink,
  appendOversizedChunkLinks,
  compressEvidenceItems,
  extractRelevantSegment,
  isHeadingLikeShortCenterSegment,
  materializeEvidenceItem,
  parseCitationNumbers,
  toPublicChunkReferences,
  toPublicDetailChunks,
} from "./traditional-rag-evidence.js";
import {
  DEFAULT_EMBEDDING_IMAGE_NEIGHBOR_MAX_CHARS,
  bboxKey,
  extractQueryTokens,
  embeddingCharBudget,
  getEvidenceCharBudget,
  imageSemanticCacheVersion,
  normalizeBaseUrl,
  normalizeEmbeddingNeighborText,
  normalizeWhitespace,
  parseJsonArray,
  parsePositiveIntegerEnv,
  qdrantPointIdForRetrievalUnit,
  resolveTextRequestTimeoutMs,
  sourceLocatorFromPages,
  toAbsoluteServerUrl,
  toFtsQuery,
  truncateTextPreservingImageLinks,
  uniqueOrdered,
  type EvidenceSegment,
  type GroupedEvidence,
  type ImageChunkRendering,
  type PreparedEvidenceSet,
  type RankedChunkHit,
  type SourceChunkRecord,
  type TraditionalRetrievalChunkRecord,
} from "./traditional-rag-shared.js";
import type { BlobStore } from "../types/library.js";
import type { ParsedDocument, ParsedImage } from "../types/parsing.js";
import type {
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
  StorageImageSemanticRecord,
  StoredDocumentChunk,
  StoredImageSemantic,
} from "../types/storage.js";

const MAX_IMAGE_BYTES_FOR_VISION = 4 * 1024 * 1024;

export type { ImageChunkRendering } from "./traditional-rag-shared.js";

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
  contextText?: string | null;
}): Promise<ImageSemanticPayload | null> {
  const config = resolveVisionConfig();
  if (!config.apiKey) {
    return null;
  }
  const baseUrl = (config.baseUrl?.trim() || "https://api.openai.com/v1").replace(/\/+$/, "");
  const dataUrl = `data:${input.mimeType};base64,${Buffer.from(input.bytes).toString("base64")}`;
  const prompts = buildVisionPromptMessages({ contextText: input.contextText ?? null });
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
      "If you want to output with images, please use Markdown format them."+
      "Evidence blocks may contain direct links to the full content. For oversized evidence, the excerpt may be compressed and the link can be included in your answer when users should inspect the full content." +
      "Return strict JSON with keys answer and citations. citations must be an array of evidence numbers such as [1, 3]. " +
      "Use inline references like [1] in the answer when helpful."
    );
  }
  return (
    "Answer only from the numbered evidence. Cite supporting evidence using [n]. " +
    "If you want to output with images, please use Markdown format them."+
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
    return "No relevant evidence was found.";
  }
  return `Found ${hits.length} relevant evidence chunks.`;
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
    retrievalChunkingStrategy?: "small_to_big" | "fixed";
  }): Promise<{ pageChunkCounts: Map<number, number>; chunksWritten: number; retrievalChunksWritten: number; embeddingsWritten: number }> {
    const imageRenderMap = await this.processImages({
      ...input,
      enableImageSemantic: input.enableImageSemantic !== false,
    });
    const sourceChunks = buildSourceChunks({
      documentId: input.documentId,
      parsedDocument: input.parsedDocument,
      imageRenderMap,
    });
    const retrievalChunks = buildRetrievalChunks(input.documentId, sourceChunks);
    const documentChunks = buildIndexedDocumentChunks(input.documentId, sourceChunks, retrievalChunks);
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
    const document = this.storage?.getDocument?.(documentId) ?? null;
    const strategy = document?.retrieval_chunking_strategy === "fixed" ? "fixed" : "small_to_big";
    const embeddingTexts =
      strategy === "small_to_big"
        ? this.buildEmbeddingTextsForDocumentChunks(documentId, retrievalUnits)
        : retrievalUnits.map((chunk) => ({ id: chunk.id, text: chunk.unitText }));
    return rebuildQdrantCollection(
      documentId,
      embeddingTexts,
    );
  }

  async renderImages(input: {
    documentId: string;
    filePath: string;
    originalFilename: string;
    parsedDocument: ParsedDocument;
    enableImageSemantic?: boolean;
    retrievalChunkingStrategy?: "small_to_big" | "fixed";
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
    return buildSourceChunks(input);
  }

  createRetrievalChunks(
    documentId: string,
    documentChunks: SourceChunkRecord[],
  ) {
    return buildRetrievalChunks(documentId, documentChunks);
  }

  createFixedRetrievalChunks(input: {
    documentId: string;
    sourceChunks: SourceChunkRecord[];
    fixedChunkChars: number;
  }) {
    return buildFixedRetrievalChunks(input);
  }

  createIndexedDocumentChunks(
    documentId: string,
    sourceChunks: SourceChunkRecord[],
    retrievalChunks: ReturnType<typeof buildRetrievalChunks>,
    fixedRetrievalChunks: ReturnType<typeof buildFixedRetrievalChunks> = [],
  ) {
    return buildIndexedDocumentChunks(documentId, sourceChunks, retrievalChunks, fixedRetrievalChunks);
  }

  private buildEmbeddingTextsForDocumentChunks(
    documentId: string,
    retrievalUnits: Array<{
      id: string;
      unitText: string;
    }>,
  ): Array<{ id: string; text: string }> {
    const storedChunks = this.storage?.listDocumentChunks?.(documentId) ?? [];
    if (storedChunks.length === 0) {
      return retrievalUnits.map((chunk) => ({ id: chunk.id, text: chunk.unitText }));
    }
    const chunkById = new Map(storedChunks.map((chunk) => [chunk.id, chunk] as const));
    const orderedChunks = [...storedChunks].sort((left, right) => left.ordinal - right.ordinal);
    const previousNonPictureById = new Map<string, StoredDocumentChunk | null>();
    let previousNonPicture: StoredDocumentChunk | null = null;
    for (const chunk of orderedChunks) {
      previousNonPictureById.set(chunk.id, previousNonPicture);
      if (chunk.block_type !== "picture" && normalizeWhitespace(chunk.content_md)) {
        previousNonPicture = chunk;
      }
    }
    const maxChars = embeddingCharBudget();
    const neighborMaxChars = parsePositiveIntegerEnv(
      "EMBEDDING_IMAGE_NEIGHBOR_MAX_CHARS",
      DEFAULT_EMBEDDING_IMAGE_NEIGHBOR_MAX_CHARS,
    );
    return retrievalUnits.map((chunk) => {
      const storedChunk = chunkById.get(chunk.id);
      if (!storedChunk || storedChunk.block_type !== "picture") {
        return { id: chunk.id, text: chunk.unitText };
      }
      const previousChunk = previousNonPictureById.get(chunk.id) ?? null;
      const previousText = previousChunk
        ? normalizeEmbeddingNeighborText(previousChunk.summary_text || previousChunk.content_md, neighborMaxChars)
        : "";
      const combined = this.composePictureEmbeddingText({
        pictureText: storedChunk.content_md,
        previousChunkText: previousText,
        maxChars,
      });
      return {
        id: chunk.id,
        text: combined || truncateTextPreservingImageLinks(storedChunk.content_md, maxChars),
      };
    });
  }

  private composePictureEmbeddingText(input: {
    pictureText: string;
    previousChunkText?: string | null;
    maxChars: number;
  }): string {
    const pictureText = normalizeWhitespace(input.pictureText);
    if (!pictureText) {
      return pictureText;
    }
    const previousChunkText = normalizeWhitespace(input.previousChunkText ?? "");
    const baseLength = pictureText.length;
    if (!previousChunkText) {
      return truncateTextPreservingImageLinks(pictureText, input.maxChars);
    }
    const separatorLength = 2;
    if (baseLength + separatorLength + previousChunkText.length <= input.maxChars) {
      return normalizeWhitespace([pictureText, previousChunkText].join("\n\n"));
    }
    if (baseLength + separatorLength <= input.maxChars) {
      const remaining = Math.max(input.maxChars - baseLength - separatorLength, 0);
      const trimmedNeighbor = normalizeEmbeddingNeighborText(previousChunkText, remaining);
      return normalizeWhitespace([pictureText, trimmedNeighbor].filter(Boolean).join("\n\n"));
    }
    return truncateTextPreservingImageLinks(pictureText, input.maxChars);
  }

  private findPreviousNonPictureBlockMarkdown(
    parsedDocument: ParsedDocument,
    targetImage: ParsedImage,
  ): string | null {
    let previousNonPicture: string | null = null;
    for (const unit of parsedDocument.units) {
      const blocks = unit.blocks.length
        ? unit.blocks
        : [
            {
              index: 0,
              block_type: "text",
              bbox: [0, 0, 0, 0] as [number, number, number, number],
              markdown: unit.markdown,
              char_count: unit.markdown.length,
              image_hash: null,
              source_image_index: null,
            },
          ];
      for (const block of blocks) {
        const blockMarkdown = normalizeWhitespace(block.markdown);
        if (block.block_type !== "picture" && blockMarkdown) {
          previousNonPicture = blockMarkdown;
        }
        if (
          unit.unit_no === targetImage.page_no &&
          block.block_type === "picture" &&
          block.image_hash === targetImage.image_hash &&
          (block.source_image_index == null || block.source_image_index === targetImage.image_index)
        ) {
          return previousNonPicture;
        }
      }
    }
    return previousNonPicture;
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
      return materializeEvidenceItem({
        ...item,
        citationNo,
        cerSegments: item.cerSegments.map((segment) => ({
          ...segment,
          citationNo,
        })),
        sourceLocator: item.sourceLocator || sourceLocatorFromPages(item.pageNos),
      });
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
    retrievalChunkingStrategy?: "small_to_big" | "fixed";
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
        const contextText =
          input.retrievalChunkingStrategy === "fixed"
            ? null
            : this.findPreviousNonPictureBlockMarkdown(input.parsedDocument, image);
        const cacheVersion = imageSemanticCacheVersion(contextText);
        const cached = this.storage.getImageSemanticCache(image.image_hash, cacheVersion);
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
            contextText,
          });
          if (semanticPayload) {
            this.storage.upsertImageSemanticCache({
              imageHash: image.image_hash,
              promptVersion: cacheVersion,
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

}

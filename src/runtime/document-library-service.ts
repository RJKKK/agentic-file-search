/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
Reference: legacy/python/src/fs_explorer/document_pages.py
*/

import { createHash, randomBytes } from "node:crypto";
import { mkdtemp, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join } from "node:path";

import { LocalBlobStore } from "./blob-store.js";
import type { PythonDocumentParserExecutor } from "./document-parsing.js";
import { PythonDocumentParserBridge } from "./document-parsing.js";
import {
  buildDocumentObjectKey,
  buildDocumentPagesKeyPrefix,
  createLibraryDocumentCatalog,
  ensureLibraryCorpus,
  materializeDocument,
  serializeDocumentParseTask,
  serializeDocumentSummary,
} from "./document-library.js";
import { pageRecordFromManifest } from "./document-pages.js";
import { persistDocumentPages, validateStorageFilename } from "./page-store.js";
import { type ImageChunkRendering, TraditionalRagService } from "./traditional-rag.js";
import type {
  BlobStore,
  DeleteDocumentResult,
  DocumentParseTaskPayload,
  ReparseDocumentResult,
  UploadDocumentInput,
  UploadDocumentResult,
} from "../types/library.js";
import type { ParsedDocument } from "../types/parsing.js";
import type {
  DocumentParseStageTiming,
  DocumentParseTaskType,
  RetrievalChunkingStrategy,
  SqliteStorageBackend,
  StorageDocumentChunkRecord,
  StorageDocumentRecord,
  StorageFixedRetrievalChunkRecord,
  StorageRetrievalChunkRecord,
  StoredDocument,
  StoredDocumentParseTask,
} from "../types/storage.js";

type UploadParseOptions = {
  enable_embedding: boolean;
  enable_image_semantic: boolean;
  chunking_strategy: RetrievalChunkingStrategy;
  fixed_chunk_chars: number | null;
};

type ReparseOptions = UploadParseOptions & {
  force: boolean;
};

type EmbedOnlyOptions = {
  enable_embedding: true;
};

type TaskOptions = UploadParseOptions | ReparseOptions | EmbedOnlyOptions;

const DEFAULT_FIXED_CHUNK_CHARS = 800;

function normalizeChunkingStrategy(value: unknown, fallback: RetrievalChunkingStrategy = "small_to_big"): RetrievalChunkingStrategy {
  return String(value ?? "").trim() === "fixed" ? "fixed" : fallback;
}

function normalizeFixedChunkChars(value: unknown): number | null {
  if (value == null || value === "") {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error("`fixed_chunk_chars` must be a positive integer.");
  }
  return Math.floor(parsed);
}

type QueuedTask =
  | {
      taskId: string;
      taskType: "upload_parse";
      documentId: string;
      filename: string;
      sourceHash: string;
      filePath: string;
      options: UploadParseOptions;
    }
  | {
      taskId: string;
      taskType: "reparse";
      documentId: string;
      options: ReparseOptions;
    }
  | {
      taskId: string;
      taskType: "embed_only";
      documentId: string;
      options: EmbedOnlyOptions;
    };

const TASK_STAGE_SEQUENCES: Record<DocumentParseTaskType, string[]> = {
  upload_parse: [
    "store_source",
    "parse_document",
    "persist_pages",
    "build_document_chunks",
    "process_images",
    "build_retrieval_chunks",
    "build_indexed_document_chunks",
    "build_embeddings",
    "finalize",
  ],
  reparse: [
    "parse_document",
    "persist_pages",
    "build_document_chunks",
    "process_images",
    "build_retrieval_chunks",
    "build_indexed_document_chunks",
    "build_embeddings",
    "finalize",
  ],
  embed_only: ["load_document_chunks", "build_embeddings", "finalize"],
};

const TASK_STAGE_LABELS: Record<string, string> = {
  store_source: "Store source",
  parse_document: "Parse document",
  persist_pages: "Persist pages",
  build_document_chunks: "Build document chunks",
  process_images: "Process images",
  build_retrieval_chunks: "Build retrieval chunks",
  build_indexed_document_chunks: "Build indexed document chunks",
  build_embeddings: "Build embeddings",
  load_document_chunks: "Load document chunks",
  finalize: "Finalize",
};

function randomDocumentId(): string {
  return randomBytes(16).toString("hex");
}

function randomTaskId(): string {
  return `task_${randomBytes(16).toString("hex")}`;
}

function nowIso(): string {
  return new Date().toISOString();
}

function computeBufferSha256(data: Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

function parseTaskOptions(task: StoredDocumentParseTask): TaskOptions {
  try {
    return JSON.parse(task.options_json || "{}") as TaskOptions;
  } catch {
    return {} as TaskOptions;
  }
}

function parseStageTimings(task: StoredDocumentParseTask): DocumentParseStageTiming[] {
  try {
    const parsed = JSON.parse(task.stage_timings_json || "[]") as DocumentParseStageTiming[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function buildStageTimings(taskType: DocumentParseTaskType): DocumentParseStageTiming[] {
  return TASK_STAGE_SEQUENCES[taskType].map((stage) => ({
    stage,
    label: TASK_STAGE_LABELS[stage] ?? stage,
    status: "pending",
    started_at: null,
    finished_at: null,
    duration_ms: null,
  }));
}

function computeProgress(stageTimings: DocumentParseStageTiming[]): number {
  if (stageTimings.length === 0) {
    return 0;
  }
  const done = stageTimings.filter((item) => ["completed", "skipped", "failed"].includes(item.status)).length;
  return Math.max(0, Math.min(100, Math.round((done / stageTimings.length) * 100)));
}

function diffMs(startedAt: string | null, finishedAt: string | null): number | null {
  if (!startedAt || !finishedAt) {
    return null;
  }
  const start = Date.parse(startedAt);
  const end = Date.parse(finishedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    return null;
  }
  return Math.max(0, end - start);
}

async function parseDocumentFromBytes(input: {
  parser: PythonDocumentParserExecutor;
  documentId: string;
  filename: string;
  data: Uint8Array;
}): Promise<ParsedDocument> {
  const extension = extname(input.filename).toLowerCase() || ".bin";
  const tempDir = await mkdtemp(join(tmpdir(), "fs-explorer-parse-"));
  const tempPath = join(tempDir, `source-${input.documentId}${extension}`);
  try {
    await writeFile(tempPath, input.data);
    return await input.parser.parseDocument(tempPath);
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
  }
}

function pageChunkCounts(documentChunks: StorageDocumentChunkRecord[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const chunk of documentChunks) {
    counts.set(chunk.pageNo, (counts.get(chunk.pageNo) ?? 0) + 1);
  }
  return counts;
}

export class DocumentLibraryService {
  private readonly traditionalRag: TraditionalRagService;

  private readonly pendingTasks: QueuedTask[] = [];

  private drainingQueue = false;

  constructor(
    readonly storage: SqliteStorageBackend,
    readonly blobStore: BlobStore = new LocalBlobStore(),
    private readonly parser: PythonDocumentParserExecutor = new PythonDocumentParserBridge(),
  ) {
    this.traditionalRag = new TraditionalRagService(storage, blobStore);
  }

  createDocumentCatalog() {
    return createLibraryDocumentCatalog({
      storage: this.storage,
      blobStore: this.blobStore,
    });
  }

  async cleanupInFlightTasks(): Promise<number> {
    const queued = this.storage.listDocumentParseTasks({ status: "queued", limit: 10_000, offset: 0 }).items;
    const running = this.storage.listDocumentParseTasks({ status: "running", limit: 10_000, offset: 0 }).items;
    const tasks = [...queued, ...running];
    for (const task of tasks) {
      if (task.document_id) {
        await this.cleanupDocumentArtifacts(task.document_id, { preserveSource: true });
        this.storage.updateDocumentUploadStatus(task.document_id, "failed");
        this.storage.updateDocumentFeatureFlags(task.document_id, { hasEmbeddings: false });
      }
    }
    return this.storage.deleteActiveDocumentParseTasks();
  }

  listDocumentParseTasks(input: {
    status?: string | null;
    taskType?: string | null;
    documentId?: string | null;
    page?: number;
    pageSize?: number;
  }): { items: DocumentParseTaskPayload[]; total: number } {
    const page = Math.max(Number(input.page ?? 1), 1);
    const pageSize = Math.max(Number(input.pageSize ?? 20), 1);
    const result = this.storage.listDocumentParseTasks({
      status: (input.status as "queued" | "running" | "completed" | "failed" | null) ?? null,
      taskType: (input.taskType as DocumentParseTaskType | null) ?? null,
      documentId: input.documentId ?? null,
      limit: pageSize,
      offset: (page - 1) * pageSize,
    });
    return {
      items: result.items.map((task) => serializeDocumentParseTask(task)),
      total: result.total,
    };
  }

  getDocumentParseTask(taskId: string): DocumentParseTaskPayload | null {
    const task = this.storage.getDocumentParseTask(taskId);
    return task ? serializeDocumentParseTask(task) : null;
  }

  deleteDocumentParseTask(taskId: string): boolean {
    const task = this.requireTask(taskId);
    if (!["completed", "failed"].includes(task.status)) {
      throw new Error("Only completed or failed tasks can be deleted.");
    }
    return this.storage.deleteDocumentParseTask(taskId);
  }

  async uploadDocument(input: UploadDocumentInput): Promise<UploadDocumentResult> {
    const filename = validateStorageFilename(input.filename || "");
    const corpusId = ensureLibraryCorpus(this.storage);
    const existing = this.storage.listDocuments(corpusId, false);
    const normalizedFilename = filename.toLowerCase();
    if (existing.some((item) => String(item.original_filename || "").toLowerCase() === normalizedFilename)) {
      throw new Error("A document with the same filename already exists.");
    }

    const docId = randomDocumentId();
    const taskId = randomTaskId();
    const sourceObjectKey = buildDocumentObjectKey(docId, filename);
    const pagesPrefix = buildDocumentPagesKeyPrefix(filename);
    const chunkingStrategy = normalizeChunkingStrategy(input.chunkingStrategy, "small_to_big");
    const fixedChunkChars =
      chunkingStrategy === "fixed"
        ? (normalizeFixedChunkChars(input.fixedChunkChars) ?? DEFAULT_FIXED_CHUNK_CHARS)
        : null;
    const taskOptions: UploadParseOptions = {
      enable_embedding: input.enableEmbedding !== false,
      enable_image_semantic: input.enableImageSemantic !== false,
      chunking_strategy: chunkingStrategy,
      fixed_chunk_chars: fixedChunkChars,
    };
    const stageTimings = buildStageTimings("upload_parse");
    this.storage.createDocumentParseTask({
      id: taskId,
      documentId: null,
      documentFilename: filename,
      taskType: "upload_parse",
      status: "queued",
      progressPercent: 0,
      currentStage: null,
      optionsJson: JSON.stringify(taskOptions),
      stageTimingsJson: JSON.stringify(stageTimings),
    });

    try {
      const persistedBlobHead = await this.runTaskStage(taskId, stageTimings, "store_source", async () =>
        this.blobStore.put({
          objectKey: sourceObjectKey,
          data: input.data,
        }),
      );
      const fileInfo = await stat(persistedBlobHead.absolutePath);
      const sourceHash = computeBufferSha256(input.data);
      const documentRecord: StorageDocumentRecord = {
        id: docId,
        corpusId,
        relativePath: filename,
        absolutePath: persistedBlobHead.absolutePath,
        content: "",
        metadataJson: "{}",
        fileMtime: Number(fileInfo.mtimeMs / 1000),
        fileSize: Number(persistedBlobHead.size),
        contentSha256: sourceHash,
        originalFilename: filename,
        objectKey: sourceObjectKey,
        sourceObjectKey,
        pagesPrefix,
        storageUri: persistedBlobHead.storageUri,
        contentType: input.contentType ?? null,
        uploadStatus: "processing",
        pageCount: 0,
        parsedContentSha256: null,
        parsedIsComplete: false,
        embeddingEnabled: taskOptions.enable_embedding,
        hasEmbeddings: false,
        imageSemanticEnabled: taskOptions.enable_image_semantic,
        retrievalChunkingStrategy: taskOptions.chunking_strategy,
        fixedChunkChars: taskOptions.fixed_chunk_chars,
      };
      this.storage.upsertDocumentStub(documentRecord);
      this.storage.updateDocumentParseTask(taskId, { documentId: docId });
      this.enqueueTask({
        taskId,
        taskType: "upload_parse",
        documentId: docId,
        filename,
        sourceHash,
        filePath: persistedBlobHead.absolutePath,
        options: taskOptions,
      });
      return {
        document: serializeDocumentSummary(this.requireDocument(docId)),
        task: serializeDocumentParseTask(this.requireTask(taskId)),
      };
    } catch (error) {
      await this.blobStore.delete({ objectKey: sourceObjectKey }).catch(() => undefined);
      if (this.storage.getDocument(docId)) {
        this.storage.deleteDocument(docId);
      }
      await this.failTask(taskId, stageTimings, null, error);
      throw error;
    }
  }

  async reparseDocument(input: {
    docId: string;
    force?: boolean;
    enableEmbedding?: boolean;
    enableImageSemantic?: boolean;
    chunkingStrategy?: RetrievalChunkingStrategy;
    fixedChunkChars?: number | null;
  }): Promise<ReparseDocumentResult> {
    const document = this.requireDocument(input.docId);
    const taskId = randomTaskId();
    const chunkingStrategy = normalizeChunkingStrategy(
      input.chunkingStrategy,
      document.retrieval_chunking_strategy,
    );
    const fixedChunkChars =
      chunkingStrategy === "fixed"
        ? (normalizeFixedChunkChars(input.fixedChunkChars) ??
          document.fixed_chunk_chars ??
          DEFAULT_FIXED_CHUNK_CHARS)
        : null;
    const options: ReparseOptions = {
      force: input.force === true,
      enable_embedding: input.enableEmbedding ?? document.embedding_enabled,
      enable_image_semantic: input.enableImageSemantic ?? document.image_semantic_enabled,
      chunking_strategy: chunkingStrategy,
      fixed_chunk_chars: fixedChunkChars,
    };
    this.storage.updateDocumentUploadStatus(document.id, "processing");
    this.storage.createDocumentParseTask({
      id: taskId,
      documentId: document.id,
      documentFilename: document.original_filename || document.relative_path,
      taskType: "reparse",
      status: "queued",
      progressPercent: 0,
      currentStage: null,
      optionsJson: JSON.stringify(options),
      stageTimingsJson: JSON.stringify(buildStageTimings("reparse")),
    });
    this.enqueueTask({
      taskId,
      taskType: "reparse",
      documentId: document.id,
      options,
    });
    return {
      document: serializeDocumentSummary(this.requireDocument(document.id)),
      task: serializeDocumentParseTask(this.requireTask(taskId)),
    };
  }

  async createEmbeddingTask(docId: string): Promise<DocumentParseTaskPayload> {
    const document = this.requireDocument(docId);
    if (document.upload_status !== "completed") {
      throw new Error("Embedding can only be created for completed documents.");
    }
    this.storage.updateDocumentUploadStatus(docId, "processing");
    this.storage.updateDocumentFeatureFlags(docId, {
      embeddingEnabled: true,
      hasEmbeddings: false,
    });
    const taskId = randomTaskId();
    this.storage.createDocumentParseTask({
      id: taskId,
      documentId: docId,
      documentFilename: document.original_filename || document.relative_path,
      taskType: "embed_only",
      status: "queued",
      progressPercent: 0,
      currentStage: null,
      optionsJson: JSON.stringify({ enable_embedding: true } satisfies EmbedOnlyOptions),
      stageTimingsJson: JSON.stringify(buildStageTimings("embed_only")),
    });
    this.enqueueTask({
      taskId,
      taskType: "embed_only",
      documentId: docId,
      options: { enable_embedding: true },
    });
    return serializeDocumentParseTask(this.requireTask(taskId));
  }

  async deleteDocument(input: { docId: string }): Promise<DeleteDocumentResult> {
    const document = this.requireDocument(input.docId);
    const pagesPrefix = String(document.pages_prefix || "");
    if (pagesPrefix) {
      await this.blobStore.deletePrefix({ prefix: pagesPrefix }).catch(() => undefined);
    }
    const imageRows = this.storage.listImageSemanticsForDocument(document.id);
    for (const image of imageRows) {
      if (image.object_key) {
        await this.blobStore.delete({ objectKey: image.object_key }).catch(() => undefined);
      }
    }
    const sourceObjectKey = String(document.source_object_key || document.object_key || "");
    if (sourceObjectKey) {
      await this.blobStore.delete({ objectKey: sourceObjectKey }).catch(() => undefined);
    }
    await this.traditionalRag.deleteDocumentIndex(document.id);
    const taskResult = this.storage.listDocumentParseTasks({
      documentId: document.id,
      limit: 10_000,
      offset: 0,
    });
    for (const task of taskResult.items) {
      this.storage.deleteDocumentParseTask(task.id);
    }
    const deletedDocument = this.storage.deleteDocument(document.id);
    if (!deletedDocument) {
      throw new Error("Document not found");
    }
    return {
      document: {
        ...serializeDocumentSummary({ ...deletedDocument, is_deleted: true }),
        is_deleted: true,
        status: "deleted",
      },
      deleted: true,
    };
  }

  async materializeDocument(docId: string): Promise<StoredDocument> {
    return materializeDocument({
      storage: this.storage,
      blobStore: this.blobStore,
      document: this.requireDocument(docId),
    });
  }

  private enqueueTask(task: QueuedTask): void {
    this.pendingTasks.push(task);
    void this.drainTaskQueue();
  }

  private async drainTaskQueue(): Promise<void> {
    if (this.drainingQueue) {
      return;
    }
    this.drainingQueue = true;
    try {
      while (this.pendingTasks.length > 0) {
        const task = this.pendingTasks.shift();
        if (!task) {
          continue;
        }
        await this.executeQueuedTask(task);
      }
    } finally {
      this.drainingQueue = false;
    }
  }

  private async executeQueuedTask(task: QueuedTask): Promise<void> {
    const storedTask = this.requireTask(task.taskId);
    const stageTimings = parseStageTimings(storedTask);
    try {
      if (task.taskType === "upload_parse") {
        await this.executeUploadParseTask(task, stageTimings);
      } else if (task.taskType === "reparse") {
        await this.executeReparseTask(task, stageTimings);
      } else {
        await this.executeEmbedOnlyTask(task, stageTimings);
      }
      await this.completeTask(task.taskId, stageTimings);
    } catch (error) {
      if (task.documentId) {
        this.storage.updateDocumentUploadStatus(task.documentId, "failed");
        this.storage.updateDocumentFeatureFlags(task.documentId, { hasEmbeddings: false });
      }
      await this.failTask(task.taskId, stageTimings, null, error);
      if (task.taskType === "upload_parse") {
        await this.cleanupDocumentArtifacts(task.documentId, { preserveSource: true });
      }
    }
  }

  private async executeUploadParseTask(
    task: Extract<QueuedTask, { taskType: "upload_parse" }>,
    stageTimings: DocumentParseStageTiming[],
  ): Promise<void> {
    const document = this.requireDocument(task.documentId);
    const parsedDocument = await this.runTaskStage(task.taskId, stageTimings, "parse_document", async () => {
      const sourceBytes = await this.blobStore.get({ objectKey: document.source_object_key });
      return parseDocumentFromBytes({
        parser: this.parser,
        documentId: document.id,
        filename: task.filename,
        data: sourceBytes,
      });
    });
    const storedPages = await this.runTaskStage(task.taskId, stageTimings, "persist_pages", async () =>
      persistDocumentPages({
        blobStore: this.blobStore,
        documentId: document.id,
        originalFilename: task.filename,
        contentType: document.content_type,
        parsedDocument,
        syntheticPages: extname(task.filename).toLowerCase() !== ".pdf",
      }),
    );
    await this.finishIndexingPhases({
      taskId: task.taskId,
      stageTimings,
      document,
      parsedDocument,
      storedPages,
      sourceHash: task.sourceHash,
      enableEmbedding: task.options.enable_embedding,
      enableImageSemantic: task.options.enable_image_semantic,
      chunkingStrategy: task.options.chunking_strategy,
      fixedChunkChars: task.options.fixed_chunk_chars,
    });
  }

  private async executeReparseTask(
    task: Extract<QueuedTask, { taskType: "reparse" }>,
    stageTimings: DocumentParseStageTiming[],
  ): Promise<void> {
    const document = await this.materializeDocument(task.documentId);
    const sourceObjectKey = String(document.source_object_key || document.object_key || "");
    if (!sourceObjectKey) {
      throw new Error("Document source object key is missing.");
    }
    const sourceBytes = await this.blobStore.get({ objectKey: sourceObjectKey });
    const sourceHash = computeBufferSha256(sourceBytes);
    const parsedDocument = await this.runTaskStage(task.taskId, stageTimings, "parse_document", async () =>
      parseDocumentFromBytes({
        parser: this.parser,
        documentId: document.id,
        filename: document.original_filename || document.relative_path || sourceObjectKey,
        data: sourceBytes,
      }),
    );
    const storedPages = await this.runTaskStage(task.taskId, stageTimings, "persist_pages", async () =>
      persistDocumentPages({
        blobStore: this.blobStore,
        documentId: document.id,
        originalFilename: document.original_filename || document.relative_path,
        contentType: document.content_type,
        parsedDocument,
        syntheticPages: extname(document.original_filename || document.relative_path).toLowerCase() !== ".pdf",
      }),
    );
    await this.finishIndexingPhases({
      taskId: task.taskId,
      stageTimings,
      document,
      parsedDocument,
      storedPages,
      sourceHash,
      enableEmbedding: task.options.enable_embedding,
      enableImageSemantic: task.options.enable_image_semantic,
      chunkingStrategy: task.options.chunking_strategy,
      fixedChunkChars: task.options.fixed_chunk_chars,
    });
  }

  private async executeEmbedOnlyTask(
    task: Extract<QueuedTask, { taskType: "embed_only" }>,
    stageTimings: DocumentParseStageTiming[],
  ): Promise<void> {
    const document = this.requireDocument(task.documentId);
    const documentChunks = await this.runTaskStage(task.taskId, stageTimings, "load_document_chunks", async () =>
      document.retrieval_chunking_strategy === "fixed"
        ? this.storage.listFixedRetrievalChunks(document.id).map((chunk) => ({
            id: chunk.id,
            unitText: chunk.content_md,
          }))
        : this.storage.listDocumentChunks(document.id).map((chunk) => ({
            id: chunk.id,
            unitText: chunk.content_md,
          })),
    );
    const embeddingsWritten = await this.runTaskStage(task.taskId, stageTimings, "build_embeddings", async () =>
      this.traditionalRag.buildEmbeddingsForChunks(document.id, documentChunks),
    );
    await this.runTaskStage(task.taskId, stageTimings, "finalize", async () => {
      this.storage.updateDocumentFeatureFlags(document.id, {
        embeddingEnabled: true,
        hasEmbeddings: embeddingsWritten > 0,
      });
      this.storage.updateDocumentUploadStatus(document.id, "completed");
    });
  }

  private async finishIndexingPhases(input: {
    taskId: string;
    stageTimings: DocumentParseStageTiming[];
    document: StoredDocument;
    parsedDocument: ParsedDocument;
    storedPages: Awaited<ReturnType<typeof persistDocumentPages>>;
    sourceHash: string;
    enableEmbedding: boolean;
    enableImageSemantic: boolean;
    chunkingStrategy: RetrievalChunkingStrategy;
    fixedChunkChars: number | null;
  }): Promise<void> {
    let imageRenderMap: Map<string, ImageChunkRendering> | null = null;
    if (input.enableImageSemantic) {
      imageRenderMap = await this.runTaskStage(input.taskId, input.stageTimings, "process_images", async () =>
        this.traditionalRag.renderImages({
          documentId: input.document.id,
          filePath: input.document.absolute_path,
          originalFilename: input.document.original_filename || input.document.relative_path,
          parsedDocument: input.parsedDocument,
          enableImageSemantic: true,
          retrievalChunkingStrategy: input.chunkingStrategy,
        }),
      );
    } else {
      await this.skipTaskStage(input.taskId, input.stageTimings, "process_images");
    }

    const sourceChunks = await this.runTaskStage(
      input.taskId,
      input.stageTimings,
      "build_document_chunks",
      async () =>
        this.traditionalRag.createDocumentChunks({
          documentId: input.document.id,
          parsedDocument: input.parsedDocument,
          imageRenderMap:
            imageRenderMap ??
            (await this.traditionalRag.renderImages({
              documentId: input.document.id,
              filePath: input.document.absolute_path,
              originalFilename: input.document.original_filename || input.document.relative_path,
              parsedDocument: input.parsedDocument,
              enableImageSemantic: false,
              retrievalChunkingStrategy: input.chunkingStrategy,
            })),
        }),
    );

    const builtRetrieval = await this.runTaskStage(
      input.taskId,
      input.stageTimings,
      "build_retrieval_chunks",
      async () => ({
        retrievalChunks:
          input.chunkingStrategy === "fixed"
            ? []
            : this.traditionalRag.createRetrievalChunks(input.document.id, sourceChunks),
        fixedRetrievalChunks:
          input.chunkingStrategy === "fixed"
            ? this.traditionalRag.createFixedRetrievalChunks({
                documentId: input.document.id,
                sourceChunks,
                fixedChunkChars: input.fixedChunkChars ?? DEFAULT_FIXED_CHUNK_CHARS,
              })
            : [],
      }),
    );
    const { retrievalChunks, fixedRetrievalChunks } = builtRetrieval;

    const documentChunks = await this.runTaskStage(
      input.taskId,
      input.stageTimings,
      "build_indexed_document_chunks",
      async () => {
        const units =
          input.chunkingStrategy === "fixed"
            ? this.traditionalRag.createIndexedDocumentChunks(
                input.document.id,
                sourceChunks,
                [],
                fixedRetrievalChunks,
              )
            : this.traditionalRag.createIndexedDocumentChunks(input.document.id, sourceChunks, retrievalChunks);
        this.storage.replaceDocumentChunks(input.document.id, units);
        this.storage.replaceRetrievalChunks(input.document.id, retrievalChunks);
        this.storage.replaceFixedRetrievalChunks(input.document.id, fixedRetrievalChunks);
        const chunkCounts = pageChunkCounts(units);
        this.storage.syncDocumentPages(
          input.document.id,
          input.storedPages.map((page) => ({
            ...pageRecordFromManifest(page, { documentId: input.document.id }),
            chunkCount: chunkCounts.get(page.pageNo) ?? 0,
          })),
        );
        return units;
      },
    );

    let embeddingsWritten = 0;
    if (input.enableEmbedding) {
      embeddingsWritten = await this.runTaskStage(
        input.taskId,
        input.stageTimings,
        "build_embeddings",
        async () =>
          this.traditionalRag.buildEmbeddingsForChunks(
            input.document.id,
            (input.chunkingStrategy === "fixed" ? fixedRetrievalChunks : documentChunks).map((chunk) => ({
              id: chunk.id,
              unitText: chunk.contentMd,
            })),
          ),
      );
    } else {
      await this.traditionalRag.deleteDocumentIndex(input.document.id);
      await this.skipTaskStage(input.taskId, input.stageTimings, "build_embeddings");
    }

    await this.runTaskStage(input.taskId, input.stageTimings, "finalize", async () => {
      this.storage.updateDocumentParseState(input.document.id, input.sourceHash, true);
      this.storage.updateDocumentFeatureFlags(input.document.id, {
        embeddingEnabled: input.enableEmbedding,
        hasEmbeddings: input.enableEmbedding && embeddingsWritten > 0,
        imageSemanticEnabled: input.enableImageSemantic,
        retrievalChunkingStrategy: input.chunkingStrategy,
        fixedChunkChars: input.fixedChunkChars,
      });
      this.storage.updateDocumentUploadStatus(input.document.id, "completed");
    });
  }

  private async runTaskStage<T>(
    taskId: string,
    stageTimings: DocumentParseStageTiming[],
    stageName: string,
    action: () => Promise<T>,
  ): Promise<T> {
    const stage = stageTimings.find((item) => item.stage === stageName);
    if (!stage) {
      throw new Error(`Unknown task stage: ${stageName}`);
    }
    const startedAt = nowIso();
    stage.status = "running";
    stage.started_at = startedAt;
    stage.finished_at = null;
    stage.duration_ms = null;
    this.persistTaskProgress(taskId, stageTimings, {
      status: "running",
      currentStage: stageName,
      startedAt,
      errorMessage: null,
    });
    try {
      const result = await action();
      const finishedAt = nowIso();
      stage.status = "completed";
      stage.finished_at = finishedAt;
      stage.duration_ms = diffMs(stage.started_at, finishedAt);
      this.persistTaskProgress(taskId, stageTimings, {
        status: "running",
        currentStage: stageName,
      });
      return result;
    } catch (error) {
      const finishedAt = nowIso();
      stage.status = "failed";
      stage.finished_at = finishedAt;
      stage.duration_ms = diffMs(stage.started_at, finishedAt);
      this.persistTaskProgress(taskId, stageTimings, {
        status: "failed",
        currentStage: stageName,
        finishedAt,
        errorMessage: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  private async skipTaskStage(
    taskId: string,
    stageTimings: DocumentParseStageTiming[],
    stageName: string,
  ): Promise<void> {
    const stage = stageTimings.find((item) => item.stage === stageName);
    if (!stage) {
      return;
    }
    stage.status = "skipped";
    stage.started_at = null;
    stage.finished_at = nowIso();
    stage.duration_ms = 0;
    this.persistTaskProgress(taskId, stageTimings, {
      status: "running",
      currentStage: stageName,
    });
  }

  private async completeTask(taskId: string, stageTimings: DocumentParseStageTiming[]): Promise<void> {
    const task = this.requireTask(taskId);
    const finishedAt = nowIso();
    this.storage.updateDocumentParseTask(taskId, {
      status: "completed",
      progressPercent: 100,
      currentStage: null,
      stageTimingsJson: JSON.stringify(stageTimings),
      errorMessage: null,
      finishedAt,
      totalDurationMs: diffMs(task.started_at, finishedAt),
    });
  }

  private async failTask(
    taskId: string,
    stageTimings: DocumentParseStageTiming[],
    stageName: string | null,
    error: unknown,
  ): Promise<void> {
    if (stageName) {
      const stage = stageTimings.find((item) => item.stage === stageName);
      if (stage) {
        stage.status = "failed";
        stage.finished_at = nowIso();
        stage.duration_ms = diffMs(stage.started_at, stage.finished_at);
      }
    }
    const task = this.storage.getDocumentParseTask(taskId);
    const finishedAt = nowIso();
    this.storage.updateDocumentParseTask(taskId, {
      status: "failed",
      currentStage: stageName,
      stageTimingsJson: JSON.stringify(stageTimings),
      errorMessage: error instanceof Error ? error.message : String(error),
      finishedAt,
      startedAt: task?.started_at ?? nowIso(),
      totalDurationMs: diffMs(task?.started_at ?? nowIso(), finishedAt),
    });
  }

  private persistTaskProgress(
    taskId: string,
    stageTimings: DocumentParseStageTiming[],
    input: {
      status: "running" | "failed";
      currentStage: string | null;
      startedAt?: string | null;
      finishedAt?: string | null;
      errorMessage?: string | null;
    },
  ): void {
    const existing = this.requireTask(taskId);
    this.storage.updateDocumentParseTask(taskId, {
      status: input.status,
      progressPercent: input.status === "failed" ? computeProgress(stageTimings) : computeProgress(stageTimings),
      currentStage: input.currentStage,
      stageTimingsJson: JSON.stringify(stageTimings),
      errorMessage: input.errorMessage ?? null,
      startedAt: existing.started_at ?? input.startedAt ?? nowIso(),
      finishedAt: input.finishedAt ?? null,
      totalDurationMs:
        input.finishedAt && (existing.started_at ?? input.startedAt)
          ? diffMs(existing.started_at ?? input.startedAt ?? null, input.finishedAt)
          : existing.total_duration_ms,
    });
  }

  private async cleanupDocumentArtifacts(
    docId: string,
    input: { preserveSource: boolean },
  ): Promise<void> {
    const document = this.storage.getDocument(docId);
    if (!document) {
      return;
    }
    const images = this.storage.listImageSemanticsForDocument(docId);
    for (const image of images) {
      if (image.object_key) {
        await this.blobStore.delete({ objectKey: image.object_key }).catch(() => undefined);
      }
    }
    if (document.pages_prefix) {
      await this.blobStore.deletePrefix({ prefix: document.pages_prefix }).catch(() => undefined);
    }
    if (!input.preserveSource && document.source_object_key) {
      await this.blobStore.delete({ objectKey: document.source_object_key }).catch(() => undefined);
    }
    await this.traditionalRag.deleteDocumentIndex(docId);
    this.storage.deleteImageSemanticsForDocument(docId);
    this.storage.syncDocumentPages(docId, []);
    this.storage.replaceFixedRetrievalChunks(docId, []);
    this.storage.replaceRetrievalChunks(docId, []);
    this.storage.replaceDocumentChunks(docId, []);
    this.storage.updateDocumentParseState(docId, null, false);
    this.storage.updateDocumentFeatureFlags(docId, { hasEmbeddings: false });
  }

  private requireTask(taskId: string): StoredDocumentParseTask {
    const task = this.storage.getDocumentParseTask(taskId);
    if (!task) {
      throw new Error("Document parse task not found.");
    }
    return task;
  }

  private requireDocument(docId: string): StoredDocument {
    const document = this.storage.getDocument(docId);
    if (!document) {
      throw new Error("Document not found");
    }
    return document;
  }
}

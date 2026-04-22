/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/explore_sessions.py
Reference: legacy/python/src/fs_explorer/document_library.py
Reference: legacy/python/src/fs_explorer/document_pages.py
*/

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { resolve } from "node:path";

import Fastify, {
  type FastifyInstance,
  type FastifyReply,
  type FastifyRequest,
} from "fastify";
import multipart from "@fastify/multipart";
import cors from "@fastify/cors";
import fastifyStatic from "@fastify/static";

import type { ActionModel } from "../agent/agent.js";
import { LocalBlobStore } from "../runtime/blob-store.js";
import type { PythonDocumentParserExecutor } from "../runtime/document-parsing.js";
import {
  getLibraryCorpusId,
  LIBRARY_CORPUS_ROOT,
  serializeDocumentSummary,
} from "../runtime/document-library.js";
import { DocumentLibraryService } from "../runtime/document-library-service.js";
import { loadDocumentPages } from "../runtime/document-pages.js";
import { encodeSseEvent } from "../runtime/explore-sessions.js";
import { ExplorationWorkflowService } from "../runtime/exploration-workflow.js";
import { IndexSearchService } from "../runtime/index-search.js";
import { createDefaultActionModel } from "../runtime/openai-compatible-model.js";
import { resolveSqliteDbPath } from "../storage/resolve-db-path.js";
import { SqliteStorage } from "../storage/sqlite.js";
import type { BlobStore } from "../types/library.js";
import type { SqliteStorageBackend, StoredDocument } from "../types/storage.js";

interface HttpServerOptions {
  storage?: SqliteStorageBackend;
  blobStore?: BlobStore;
  model?: ActionModel;
  parser?: PythonDocumentParserExecutor;
  dbPath?: string | null;
  skillsRoot?: string;
  frontendDistDir?: string | null;
  corsOrigins?: string[] | boolean;
  logger?: boolean;
}

interface ErrorPayload {
  error_code: string;
  message: string;
  details?: Record<string, unknown>;
}

function makeTraceId(): string {
  return randomUUID().replaceAll("-", "");
}

function withTrace(
  reply: FastifyReply,
  payload: Record<string, unknown>,
  input: { statusCode?: number; traceId?: string } = {},
): FastifyReply {
  reply.header("X-Trace-Id", input.traceId ?? makeTraceId());
  return reply.code(input.statusCode ?? 200).send(payload);
}

function errorResponse(
  reply: FastifyReply,
  input: {
    statusCode: number;
    errorCode: string;
    message: string;
    traceId?: string;
    details?: Record<string, unknown>;
  },
): FastifyReply {
  const payload: ErrorPayload = {
    error_code: input.errorCode,
    message: input.message,
  };
  if (input.details) {
    payload.details = input.details;
  }
  return withTrace(reply, payload as unknown as Record<string, unknown>, {
    statusCode: input.statusCode,
    traceId: input.traceId,
  });
}

function toStringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function toBoolValue(value: unknown, fallback = false): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    return ["1", "true", "yes", "on"].includes(value.toLowerCase());
  }
  return fallback;
}

function toIntValue(value: unknown, fallback: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toBatchMode(value: unknown): "auto" | "off" | "force" {
  const mode = String(value ?? "auto").trim();
  return mode === "off" || mode === "force" ? mode : "auto";
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return [...new Set(value.map((item) => String(item).trim()).filter(Boolean))];
}

function collectionIdsFromBody(body: Record<string, unknown>): string[] {
  return [
    ...new Set([
      ...toStringArray(body.collection_ids),
      ...(toStringValue(body.collection_id, "").trim()
        ? [toStringValue(body.collection_id).trim()]
        : []),
    ]),
  ];
}

function parseJsonObject(value: string): Record<string, unknown> {
  const parsed = JSON.parse(value || "{}") as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {};
  }
  return parsed as Record<string, unknown>;
}

function serializeCollection(collection: {
  id: string;
  name: string;
  is_deleted: boolean;
  created_at: string;
  updated_at: string;
  document_count?: number;
}): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    id: collection.id,
    name: collection.name,
    is_deleted: collection.is_deleted,
    created_at: collection.created_at,
    updated_at: collection.updated_at,
  };
  if (typeof collection.document_count === "number") {
    payload.document_count = collection.document_count;
  }
  return payload;
}

function serializeDocumentPage(page: {
  page_no: number;
  file_path: string;
  heading: string | null;
  source_locator: string | null;
  markdown: string;
  char_count: number;
  page_label: string;
  is_synthetic_page: boolean;
}): Record<string, unknown> {
  return {
    page_no: page.page_no,
    file_path: page.file_path,
    heading: page.heading,
    source_locator: page.source_locator,
    markdown: page.markdown,
    char_count: page.char_count,
    page_label: page.page_label,
    is_synthetic_page: page.is_synthetic_page,
  };
}

function requireDocument(storage: SqliteStorageBackend, docId: string): StoredDocument {
  const document = storage.getDocument(docId);
  if (!document || document.is_deleted) {
    throw new Error("Document not found");
  }
  return document;
}

function makeStorage(input: HttpServerOptions): { storage: SqliteStorageBackend; owned: boolean } {
  if (input.storage) {
    return { storage: input.storage, owned: false };
  }
  const storage = new SqliteStorage({
    dbPath: resolveSqliteDbPath(input.dbPath),
  });
  storage.initialize();
  return { storage, owned: true };
}

function makeModelFallback(): ActionModel {
  return {
    generateAction() {
      return {
        action: {
          final_result:
            "Exploration model is not configured for this Node HTTP server instance.",
        },
        reason: "No ActionModel was provided to createHttpServer.",
      };
    },
  };
}

async function filePartToBuffer(part: { toBuffer(): Promise<Buffer> }): Promise<Buffer> {
  return part.toBuffer();
}

async function sendFrontendIndex(reply: FastifyReply, frontendDist: string): Promise<FastifyReply> {
  const indexPath = resolve(frontendDist, "index.html");
  if (!existsSync(indexPath)) {
    return reply.type("text/html").send("<h1>Agentic File Search Node API</h1>");
  }
  return reply.type("text/html").send(await readFile(indexPath, "utf8"));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolveDelay) => {
    setTimeout(resolveDelay, ms);
  });
}

export async function createHttpServer(
  options: HttpServerOptions = {},
): Promise<FastifyInstance> {
  const { storage, owned: ownsStorage } = makeStorage(options);
  const blobStore = options.blobStore ?? new LocalBlobStore();
  const libraryService = new DocumentLibraryService(storage, blobStore, options.parser);
  const workflowService = new ExplorationWorkflowService({
    storage,
    blobStore,
    model: options.model ?? createDefaultActionModel() ?? makeModelFallback(),
    skillsRoot: options.skillsRoot,
  });
  const app = Fastify({
    logger:
      options.logger ??
      (process.env.FS_EXPLORER_LOG_LEVEL
        ? { level: process.env.FS_EXPLORER_LOG_LEVEL }
        : false),
    bodyLimit: Number(process.env.FS_EXPLORER_BODY_LIMIT_BYTES || 1024 * 1024 * 32),
  });
  const corsOrigins =
    options.corsOrigins ??
    (process.env.FS_EXPLORER_CORS_ORIGINS
      ? process.env.FS_EXPLORER_CORS_ORIGINS.split(",").map((item) => item.trim()).filter(Boolean)
      : false);
  await app.register(cors, {
    origin: corsOrigins,
  });
  await app.register(multipart, {
    limits: {
      files: 1,
      fileSize: Number(process.env.FS_EXPLORER_UPLOAD_LIMIT_BYTES || 1024 * 1024 * 512),
    },
  });
  app.addHook("onClose", async () => {
    if (ownsStorage) {
      storage.close();
    }
  });

  const frontendDist = resolve(options.frontendDistDir ?? "frontend/dist");
  const frontendAssets = resolve(frontendDist, "assets");
  if (existsSync(frontendAssets)) {
    await app.register(fastifyStatic, {
      root: frontendAssets,
      prefix: "/assets/",
      wildcard: false,
    });
  }

  app.get("/", async (_request, reply) => {
    return sendFrontendIndex(reply, frontendDist);
  });

  app.get("/api/health", async () => ({
    ok: true,
    service: "agentic-file-search",
    runtime: "node",
  }));

  app.get("/api/index/status", async (request, reply) => {
    const query = request.query as Record<string, unknown>;
    const service = new IndexSearchService({
      storage,
      blobStore,
      rootPath: toStringValue(query.folder, LIBRARY_CORPUS_ROOT),
    });
    return service.getIndexStatus();
  });

  app.post("/api/index", async (_request, reply) =>
    errorResponse(reply, {
      statusCode: 501,
      errorCode: "index_build_not_supported",
      message:
        "Folder indexing is not implemented in the Node phase-1 server. Upload documents into the library instead.",
    }),
  );

  app.post("/api/index/auto-profile", async (_request, reply) =>
    errorResponse(reply, {
      statusCode: 501,
      errorCode: "auto_profile_not_supported",
      message:
        "Metadata auto-profile discovery is not implemented in the Node phase-1 server.",
    }),
  );

  app.post("/api/search", async (request, reply) => {
    const body = (request.body ?? {}) as Record<string, unknown>;
    const traceId = makeTraceId();
    try {
      const service = new IndexSearchService({
        storage,
        blobStore,
        documentIds: toStringArray(body.document_ids),
        collectionId: toStringValue(body.collection_id, "") || null,
        collectionIds: collectionIdsFromBody(body),
      });
      const result = await service.search({
        query: toStringValue(body.query),
        filters: toStringValue(body.filters, "") || null,
        limit: toIntValue(body.limit, 5),
      });
      return withTrace(reply, result as unknown as Record<string, unknown>, { traceId });
    } catch (error) {
      return errorResponse(reply, {
        statusCode: /filter/i.test(String(error)) ? 400 : 500,
        errorCode: "search_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.get("/api/documents", async (request, reply) => {
    const traceId = makeTraceId();
    const query = request.query as Record<string, unknown>;
    const includeDeleted = toBoolValue(query.include_deleted, false);
    const page = toIntValue(query.page, 1);
    const pageSize = toIntValue(query.page_size, 20);
    if (page < 1 || pageSize < 1) {
      return errorResponse(reply, {
        statusCode: 400,
        errorCode: "invalid_pagination",
        message: "`page` and `page_size` must be positive integers.",
        traceId,
      });
    }
    try {
      const corpusId = getLibraryCorpusId(storage, { createIfMissing: false });
      let documents = corpusId ? storage.listDocuments(corpusId, includeDeleted) : [];
      const q = toStringValue(query.q).trim().toLowerCase();
      if (q) {
        documents = documents.filter((document) => {
          const metadataText = JSON.stringify(parseJsonObject(document.metadata_json)).toLowerCase();
          const haystack = [
            document.original_filename,
            document.relative_path,
            document.absolute_path,
            metadataText,
          ]
            .join(" ")
            .toLowerCase();
          return haystack.includes(q);
        });
      }
      const total = documents.length;
      const items = documents.slice((page - 1) * pageSize, page * pageSize);
      return withTrace(
        reply,
        {
          library: "default",
          corpus_id: corpusId ?? "",
          page,
          page_size: pageSize,
          total,
          items: items.map((document) => serializeDocumentSummary(document)),
        },
        { traceId },
      );
    } catch (error) {
      return errorResponse(reply, {
        statusCode: 500,
        errorCode: "document_list_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.post("/api/documents", async (request, reply) => {
    const traceId = makeTraceId();
    try {
      const part = await request.file();
      if (!part) {
        return errorResponse(reply, {
          statusCode: 400,
          errorCode: "missing_file",
          message: "No file uploaded.",
          traceId,
        });
      }
      const result = await libraryService.uploadDocument({
        filename: part.filename,
        data: await filePartToBuffer(part),
        contentType: part.mimetype,
      });
      return withTrace(
        reply,
        {
          document: result.document,
          upload_result: result.uploadResult,
        },
        { statusCode: 201, traceId },
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const duplicate = /same filename/i.test(message);
      return errorResponse(reply, {
        statusCode: duplicate ? 409 : 500,
        errorCode: duplicate ? "duplicate_filename" : "document_upload_failed",
        message,
        traceId,
      });
    }
  });

  app.get("/api/documents/:docId", async (request, reply) => {
    const traceId = makeTraceId();
    const { docId } = request.params as { docId: string };
    try {
      const document = requireDocument(storage, docId);
      const pages = storage.listDocumentPages(docId);
      return withTrace(
        reply,
        {
          document: serializeDocumentSummary(document, { pageCount: pages.length }),
          page_summary: {
            page_count: pages.length,
            latest_page_no: pages.length ? Math.max(...pages.map((page) => page.page_no)) : 0,
            synthetic_pages: pages.filter((page) => page.is_synthetic_page).length,
            pages_prefix: document.pages_prefix || "",
          },
        },
        { traceId },
      );
    } catch (error) {
      return errorResponse(reply, {
        statusCode: 404,
        errorCode: "document_not_found",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.patch("/api/documents/:docId", async (request, reply) => {
    const traceId = makeTraceId();
    const { docId } = request.params as { docId: string };
    const body = (request.body ?? {}) as { metadata?: Record<string, unknown> };
    try {
      const updated = storage.updateDocumentMetadata(
        docId,
        JSON.stringify(body.metadata ?? {}, Object.keys(body.metadata ?? {}).sort()),
      );
      if (!updated) {
        return errorResponse(reply, {
          statusCode: 404,
          errorCode: "document_not_found",
          message: "Document not found",
          traceId,
        });
      }
      return withTrace(reply, { document: serializeDocumentSummary(updated) }, { traceId });
    } catch (error) {
      return errorResponse(reply, {
        statusCode: 500,
        errorCode: "document_update_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.delete("/api/documents/:docId", async (request, reply) => {
    const traceId = makeTraceId();
    const { docId } = request.params as { docId: string };
    try {
      const result = await libraryService.deleteDocument({ docId });
      return withTrace(
        reply,
        {
          document: result.document,
          deleted: result.deleted,
        },
        { traceId },
      );
    } catch (error) {
      return errorResponse(reply, {
        statusCode: /not found/i.test(String(error)) ? 404 : 500,
        errorCode: /not found/i.test(String(error))
          ? "document_not_found"
          : "document_delete_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.post("/api/documents/:docId/parse", async (request, reply) => {
    const traceId = makeTraceId();
    const { docId } = request.params as { docId: string };
    const body = (request.body ?? {}) as Record<string, unknown>;
    const mode = toStringValue(body.mode, "incremental");
    if (!["incremental", "full"].includes(mode)) {
      return errorResponse(reply, {
        statusCode: 400,
        errorCode: "invalid_parse_mode",
        message: "`mode` must be either `incremental` or `full`.",
        traceId,
      });
    }
    try {
      const result = await libraryService.reparseDocument({
        docId,
        force: toBoolValue(body.force, false) || mode === "full",
      });
      return withTrace(
        reply,
        {
          document_id: result.documentId,
          page_count: result.pageCount,
          pages_updated: result.pagesUpdated,
          from_cache: result.fromCache,
          page_naming_scheme: result.pageNamingScheme,
        },
        { traceId },
      );
    } catch (error) {
      return errorResponse(reply, {
        statusCode: /not found/i.test(String(error)) ? 404 : 500,
        errorCode: /not found/i.test(String(error))
          ? "document_not_found"
          : "document_parse_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.get("/api/documents/:docId/pages", async (request, reply) => {
    const traceId = makeTraceId();
    const { docId } = request.params as { docId: string };
    const query = request.query as Record<string, unknown>;
    const page = toIntValue(query.page, 1);
    const pageSize = toIntValue(query.page_size, 20);
    if (page < 1 || pageSize < 1) {
      return errorResponse(reply, {
        statusCode: 400,
        errorCode: "invalid_pagination",
        message: "`page` and `page_size` must be positive integers.",
        traceId,
      });
    }
    try {
      requireDocument(storage, docId);
      const pages = await loadDocumentPages({ storage, blobStore, documentId: docId });
      const items = pages.slice((page - 1) * pageSize, page * pageSize);
      return withTrace(
        reply,
        {
          document_id: docId,
          page,
          page_size: pageSize,
          total: pages.length,
          items: items.map(serializeDocumentPage),
        },
        { traceId },
      );
    } catch (error) {
      return errorResponse(reply, {
        statusCode: /not found/i.test(String(error)) ? 404 : 500,
        errorCode: /not found/i.test(String(error))
          ? "document_not_found"
          : "document_pages_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.get("/api/collections", async () => ({
    items: storage.listCollections(false).map((collection) =>
      serializeCollection({
        ...collection,
        document_count: storage.countCollectionDocuments(collection.id),
      }),
    ),
  }));

  app.post("/api/collections", async (request, reply) => {
    const traceId = makeTraceId();
    const body = (request.body ?? {}) as { name?: string };
    const name = String(body.name ?? "").trim();
    if (!name) {
      return errorResponse(reply, {
        statusCode: 400,
        errorCode: "invalid_collection_name",
        message: "Collection name is required.",
        traceId,
      });
    }
    try {
      const collection = storage.createCollection(name);
      return withTrace(
        reply,
        {
          collection: serializeCollection({ ...collection, document_count: 0 }),
        },
        { statusCode: 201, traceId },
      );
    } catch (error) {
      const duplicate = /already exists/i.test(String(error));
      return errorResponse(reply, {
        statusCode: duplicate ? 409 : 500,
        errorCode: duplicate ? "duplicate_collection_name" : "collection_create_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.get("/api/collections/:collectionId", async (request, reply) => {
    const traceId = makeTraceId();
    const { collectionId } = request.params as { collectionId: string };
    const collection = storage.getCollection(collectionId);
    if (!collection || collection.is_deleted) {
      return errorResponse(reply, {
        statusCode: 404,
        errorCode: "collection_not_found",
        message: "Collection not found",
        traceId,
      });
    }
    return withTrace(reply, {
      collection: serializeCollection({
        ...collection,
        document_count: storage.countCollectionDocuments(collectionId),
      }),
    }, { traceId });
  });

  app.patch("/api/collections/:collectionId", async (request, reply) => {
    const traceId = makeTraceId();
    const { collectionId } = request.params as { collectionId: string };
    const body = (request.body ?? {}) as { name?: string };
    const name = String(body.name ?? "").trim();
    if (!name) {
      return errorResponse(reply, {
        statusCode: 400,
        errorCode: "invalid_collection_name",
        message: "Collection name is required.",
        traceId,
      });
    }
    try {
      const collection = storage.updateCollection(collectionId, name);
      if (!collection || collection.is_deleted) {
        return errorResponse(reply, {
          statusCode: 404,
          errorCode: "collection_not_found",
          message: "Collection not found",
          traceId,
        });
      }
      return withTrace(
        reply,
        {
          collection: serializeCollection({
            ...collection,
            document_count: storage.countCollectionDocuments(collectionId),
          }),
        },
        { traceId },
      );
    } catch (error) {
      const duplicate = /already exists/i.test(String(error));
      return errorResponse(reply, {
        statusCode: duplicate ? 409 : 500,
        errorCode: duplicate ? "duplicate_collection_name" : "collection_update_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.delete("/api/collections/:collectionId", async (request, reply) => {
    const traceId = makeTraceId();
    const { collectionId } = request.params as { collectionId: string };
    const collection = storage.setCollectionDeleted(collectionId, true);
    if (!collection) {
      return errorResponse(reply, {
        statusCode: 404,
        errorCode: "collection_not_found",
        message: "Collection not found",
        traceId,
      });
    }
    return withTrace(reply, { collection: serializeCollection(collection), deleted: true }, { traceId });
  });

  app.get("/api/collections/:collectionId/documents", async (request, reply) => {
    const traceId = makeTraceId();
    const { collectionId } = request.params as { collectionId: string };
    const collection = storage.getCollection(collectionId);
    if (!collection || collection.is_deleted) {
      return errorResponse(reply, {
        statusCode: 404,
        errorCode: "collection_not_found",
        message: "Collection not found",
        traceId,
      });
    }
    return withTrace(reply, {
      collection: serializeCollection({
        ...collection,
        document_count: storage.countCollectionDocuments(collectionId),
      }),
      items: storage.listCollectionDocuments(collectionId, false).map((document) =>
        serializeDocumentSummary(document),
      ),
    }, { traceId });
  });

  app.post("/api/collections/:collectionId/documents", async (request, reply) => {
    const traceId = makeTraceId();
    const { collectionId } = request.params as { collectionId: string };
    const collection = storage.getCollection(collectionId);
    if (!collection || collection.is_deleted) {
      return errorResponse(reply, {
        statusCode: 404,
        errorCode: "collection_not_found",
        message: "Collection not found",
        traceId,
      });
    }
    const body = (request.body ?? {}) as { document_ids?: unknown[] };
    try {
      const attached = storage.attachDocumentsToCollection(collectionId, toStringArray(body.document_ids));
      return withTrace(reply, {
        collection: serializeCollection({
          ...collection,
          document_count: storage.countCollectionDocuments(collectionId),
        }),
      attached,
      items: storage.listCollectionDocuments(collectionId, false).map((document) =>
        serializeDocumentSummary(document),
      ),
      }, { traceId });
    } catch (error) {
      return errorResponse(reply, {
        statusCode: /FOREIGN KEY/i.test(String(error)) ? 400 : 500,
        errorCode: "collection_documents_update_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.put("/api/collections/:collectionId/documents", async (request, reply) => {
    const traceId = makeTraceId();
    const { collectionId } = request.params as { collectionId: string };
    const collection = storage.getCollection(collectionId);
    if (!collection || collection.is_deleted) {
      return errorResponse(reply, {
        statusCode: 404,
        errorCode: "collection_not_found",
        message: "Collection not found",
        traceId,
      });
    }
    const body = (request.body ?? {}) as { document_ids?: unknown[] };
    try {
      const replaced = storage.replaceCollectionDocuments(collectionId, toStringArray(body.document_ids));
      return withTrace(reply, {
        collection: serializeCollection({
          ...collection,
          document_count: storage.countCollectionDocuments(collectionId),
        }),
        replaced,
        items: storage.listCollectionDocuments(collectionId, false).map((document) =>
          serializeDocumentSummary(document),
        ),
      }, { traceId });
    } catch (error) {
      return errorResponse(reply, {
        statusCode: /FOREIGN KEY/i.test(String(error)) ? 400 : 500,
        errorCode: "collection_documents_replace_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.delete("/api/collections/:collectionId/documents/:docId", async (request, reply) => {
    const traceId = makeTraceId();
    const { collectionId, docId } = request.params as { collectionId: string; docId: string };
    const collection = storage.getCollection(collectionId);
    if (!collection || collection.is_deleted) {
      return errorResponse(reply, {
        statusCode: 404,
        errorCode: "collection_not_found",
        message: "Collection not found",
        traceId,
      });
    }
    return withTrace(reply, {
      removed: storage.detachDocumentFromCollection(collectionId, docId),
    }, { traceId });
  });

  app.get("/api/documents/:docId/collections", async (request, reply) => {
    const traceId = makeTraceId();
    const { docId } = request.params as { docId: string };
    try {
      requireDocument(storage, docId);
      return withTrace(reply, {
        document_id: docId,
        items: storage.listDocumentCollections(docId, false).map(serializeCollection),
      }, { traceId });
    } catch (error) {
      return errorResponse(reply, {
        statusCode: 404,
        errorCode: "document_not_found",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.put("/api/documents/:docId/collections", async (request, reply) => {
    const traceId = makeTraceId();
    const { docId } = request.params as { docId: string };
    const body = (request.body ?? {}) as { collection_ids?: unknown[] };
    const collectionIds = toStringArray(body.collection_ids);
    try {
      requireDocument(storage, docId);
      for (const collectionId of collectionIds) {
        const collection = storage.getCollection(collectionId);
        if (!collection || collection.is_deleted) {
          return errorResponse(reply, {
            statusCode: 404,
            errorCode: "collection_not_found",
            message: "Collection not found",
            traceId,
          });
        }
      }
      storage.replaceDocumentCollections(docId, collectionIds);
      return withTrace(reply, {
        document_id: docId,
        items: storage.listDocumentCollections(docId, false).map(serializeCollection),
      }, { traceId });
    } catch (error) {
      return errorResponse(reply, {
        statusCode: /Document not found/i.test(String(error)) ? 404 : 500,
        errorCode: /Document not found/i.test(String(error))
          ? "document_not_found"
          : "document_collections_update_failed",
        message: error instanceof Error ? error.message : String(error),
        traceId,
      });
    }
  });

  app.post("/api/explore/sessions", async (request, reply) => {
    const body = (request.body ?? {}) as Record<string, unknown>;
    try {
      const result = await workflowService.startSession({
        task: toStringValue(body.task).trim(),
        documentIds: toStringArray(body.document_ids),
        collectionId: toStringValue(body.collection_id, "") || null,
        collectionIds: collectionIdsFromBody(body),
        dbPath: toStringValue(body.db_path, "") || null,
        enableSemantic: toBoolValue(body.enable_semantic, false),
        enableMetadata: toBoolValue(body.enable_metadata, false),
        batchMode: toBatchMode(body.batch_mode),
        batchSize: toIntValue(body.batch_size, 5),
        batchThreshold: toIntValue(body.batch_threshold, 10),
      });
      return result;
    } catch (error) {
      return reply.code(/selected|task/i.test(String(error)) ? 400 : 500).send({
        detail: error instanceof Error ? error.message : String(error),
      });
    }
  });

  app.get("/api/explore/sessions/:sessionId", async (request, reply) => {
    const { sessionId } = request.params as { sessionId: string };
    const session = workflowService.getSession(sessionId);
    if (!session) {
      return reply.code(404).send({ detail: "Session not found" });
    }
    return session.snapshot();
  });

  app.get("/api/explore/sessions/:sessionId/events", async (request, reply) => {
    const { sessionId } = request.params as { sessionId: string };
    const subscription = workflowService.sessions.subscribe(sessionId);
    if (!subscription.session || !subscription.queue || !subscription.history) {
      return reply.code(404).send({ detail: "Session not found" });
    }
    reply.hijack();
    reply.raw.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });
    reply.raw.socket?.setNoDelay(true);
    reply.raw.flushHeaders?.();
    const { session, queue, history } = subscription;
    let closed = false;
    const keepalive = setInterval(() => {
      if (!closed && !reply.raw.destroyed) {
        reply.raw.write(": keepalive\n\n");
      }
    }, 15_000);
    const handleClose = () => {
      if (closed) {
        return;
      }
      closed = true;
      workflowService.sessions.unsubscribe(session, queue);
    };
    // Reference: legacy/python/src/fs_explorer/server.py checks disconnects
    // while streaming the response. On Node/Fastify, tie cleanup to the
    // underlying socket rather than request-body completion.
    reply.raw.socket?.on("close", handleClose);
    request.raw.on("close", () => {
      if (request.raw.aborted) {
        handleClose();
      }
    });
    void (async () => {
      try {
        reply.raw.write(": connected\n\n");
        // Reference: legacy/python/src/fs_explorer/server.py StreamingResponse
        // yields after the request coroutine returns, so the browser has a beat
        // to install EventSource listeners before replayed history arrives.
        await delay(25);
        if (closed) {
          return;
        }
        for (const event of history) {
          reply.raw.write(encodeSseEvent(session.sessionId, event));
          if (["complete", "error"].includes(event.type)) {
            return;
          }
        }
        for await (const event of queue) {
          if (closed) {
            break;
          }
          reply.raw.write(encodeSseEvent(session.sessionId, event));
          if (["complete", "error"].includes(event.type)) {
            break;
          }
        }
      } finally {
        clearInterval(keepalive);
        reply.raw.socket?.off("close", handleClose);
        workflowService.sessions.unsubscribe(session, queue);
        if (!reply.raw.destroyed) {
          reply.raw.end();
        }
      }
    })();
    await once(reply.raw, "close").catch(() => undefined);
    return reply;
  });

  app.post("/api/explore/sessions/:sessionId/reply", async (request, reply) => {
    const { sessionId } = request.params as { sessionId: string };
    const body = (request.body ?? {}) as { response?: string };
    try {
      return await workflowService.replyToSession({
        sessionId,
        response: String(body.response ?? "").trim(),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const status = /not found/i.test(message) ? 404 : /not waiting|empty/i.test(message) ? 409 : 500;
      return reply.code(status).send({ detail: message });
    }
  });

  app.get("/*", async (request, reply) => {
    const url = request.url || "";
    if (url.startsWith("/api/")) {
      return reply.code(404).send({ detail: "Not found" });
    }
    return sendFrontendIndex(reply, frontendDist);
  });

  return app;
}

export async function runServer(input: {
  host?: string;
  port?: number;
  options?: HttpServerOptions;
} = {}): Promise<FastifyInstance> {
  const app = await createHttpServer(input.options ?? {});
  await app.listen({
    host: input.host ?? "127.0.0.1",
    port: input.port ?? 8000,
  });
  return app;
}

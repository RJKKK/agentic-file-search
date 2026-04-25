/*
Reference: legacy/python/src/fs_explorer/storage/base.py
Reference: legacy/python/src/fs_explorer/storage/postgres.py
Reference: legacy/python/src/fs_explorer/document_library.py
*/

import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

import Database from "better-sqlite3";

import type { DocumentCatalog, DocumentRecord, DocumentSummary } from "../types/skills.js";
import type {
  DocumentParseTaskStatus,
  DocumentParseTaskType,
  DocumentChunkSizeClass,
  DocumentChunkKeywordHit,
  PublicCollectionRecord,
  SqliteStorageBackend,
  StorageDocumentParseTaskRecord,
  StorageDocumentChunkRecord,
  SqliteStorageOptions,
  StorageDocumentPageRecord,
  StorageDocumentRecord,
  StorageImageSemanticCacheRecord,
  StorageImageSemanticRecord,
  StorageRetrievalChunkRecord,
  StoredCollection,
  StoredDocument,
  StoredDocumentChunk,
  StoredDocumentParseTask,
  StoredDocumentPage,
  StoredImageSemanticCache,
  StoredImageSemantic,
  StoredRetrievalChunk,
} from "../types/storage.js";

type SqliteRow = Record<string, unknown>;

function stableId(prefix: string, value: string): string {
  const digest = createHash("sha1").update(value, "utf8").digest("hex");
  return `${prefix}_${digest}`;
}

function utcNowIso(): string {
  return new Date().toISOString();
}

function normalizeMetadataJson(raw: string): string {
  if (!raw.trim()) {
    return "{}";
  }
  try {
    const parsed = JSON.parse(raw);
    return JSON.stringify(parsed);
  } catch {
    return raw;
  }
}

function normalizeJsonText(raw: string | null | undefined, fallback = "[]"): string {
  const value = String(raw ?? "").trim();
  if (!value) {
    return fallback;
  }
  try {
    return JSON.stringify(JSON.parse(value));
  } catch {
    return fallback;
  }
}

function normalizeStoredDocument(row: SqliteRow): StoredDocument {
  return {
    id: String(row.id),
    corpus_id: String(row.corpus_id),
    relative_path: String(row.relative_path),
    absolute_path: String(row.absolute_path),
    original_filename: String(row.original_filename ?? row.relative_path ?? ""),
    object_key: String(row.object_key ?? ""),
    source_object_key: String(row.source_object_key ?? row.object_key ?? ""),
    pages_prefix: String(row.pages_prefix ?? ""),
    storage_uri: String(row.storage_uri ?? ""),
    content_type: row.content_type == null ? null : String(row.content_type),
    upload_status: String(row.upload_status ?? "uploaded"),
    page_count: Number(row.page_count ?? 0),
    content: String(row.content ?? ""),
    metadata_json: normalizeMetadataJson(String(row.metadata_json ?? "{}")),
    file_mtime: Number(row.file_mtime ?? 0),
    file_size: Number(row.file_size ?? 0),
    content_sha256: String(row.content_sha256 ?? ""),
    parsed_content_sha256:
      row.parsed_content_sha256 == null ? null : String(row.parsed_content_sha256),
    parsed_is_complete: Boolean(row.parsed_is_complete),
    embedding_enabled: Boolean(row.embedding_enabled ?? 1),
    has_embeddings: Boolean(row.has_embeddings ?? 0),
    image_semantic_enabled: Boolean(row.image_semantic_enabled ?? 1),
    last_indexed_at: String(row.last_indexed_at ?? ""),
    is_deleted: Boolean(row.is_deleted),
  };
}

function normalizeStoredCollection(row: SqliteRow): StoredCollection {
  return {
    id: String(row.id),
    name: String(row.name),
    kind: String(row.kind) === "corpus_scope" ? "corpus_scope" : "user",
    corpus_id: row.corpus_id == null ? null : String(row.corpus_id),
    root_path: row.root_path == null ? null : String(row.root_path),
    is_deleted: Boolean(row.is_deleted),
    created_at: String(row.created_at),
    updated_at: String(row.updated_at),
  };
}

function toPublicCollectionRecord(collection: StoredCollection): PublicCollectionRecord {
  return {
    id: collection.id,
    name: collection.name,
    is_deleted: collection.is_deleted,
    created_at: collection.created_at,
    updated_at: collection.updated_at,
  };
}

function normalizeCollectionName(name: string): string {
  return String(name || "").trim();
}

function normalizeStoredDocumentPage(row: SqliteRow): StoredDocumentPage {
  return {
    document_id: String(row.document_id),
    page_no: Number(row.page_no),
    object_key: String(row.object_key ?? ""),
    content_hash: String(row.content_hash ?? ""),
    char_count: Number(row.char_count ?? 0),
    chunk_count: Number(row.chunk_count ?? 0),
    is_synthetic_page: Boolean(row.is_synthetic_page),
    heading: row.heading == null ? null : String(row.heading),
    source_locator: row.source_locator == null ? null : String(row.source_locator),
  };
}

function normalizeStoredImageSemantic(row: SqliteRow): StoredImageSemantic {
  return {
    image_hash: String(row.image_hash),
    source_document_id: String(row.source_document_id),
    source_page_no: Number(row.source_page_no),
    source_image_index: Number(row.source_image_index),
    mime_type: row.mime_type == null ? null : String(row.mime_type),
    width: row.width == null ? null : Number(row.width),
    height: row.height == null ? null : Number(row.height),
    bbox_json: row.bbox_json == null ? null : String(row.bbox_json),
    object_key: row.object_key == null ? null : String(row.object_key),
    storage_uri: row.storage_uri == null ? null : String(row.storage_uri),
    has_text: row.has_text == null ? null : Boolean(row.has_text),
    interference_score: row.interference_score == null ? null : Number(row.interference_score),
    is_dropped: Boolean(row.is_dropped),
    recognizable: row.recognizable == null ? null : Boolean(row.recognizable),
    accessible_url: row.accessible_url == null ? null : String(row.accessible_url),
    semantic_text: row.semantic_text == null ? null : String(row.semantic_text),
    semantic_model: row.semantic_model == null ? null : String(row.semantic_model),
  };
}

function normalizeStoredDocumentChunk(row: SqliteRow): StoredDocumentChunk {
  return {
    id: String(row.id),
    document_id: String(row.document_id),
    ordinal: Number(row.ordinal ?? 0),
    reference_retrieval_chunk_id:
      row.reference_retrieval_chunk_id == null ? null : String(row.reference_retrieval_chunk_id),
    page_no: Number(row.page_no),
    document_index: Number(row.document_index),
    page_index: Number(row.page_index),
    block_type: String(row.block_type),
    bbox_json: String(row.bbox_json ?? "[]"),
    content_md: String(row.content_md ?? ""),
    size_class: String(row.size_class ?? "normal") as DocumentChunkSizeClass,
    summary_text: row.summary_text == null ? null : String(row.summary_text),
    is_split_from_oversized: Boolean(row.is_split_from_oversized),
    split_index: Number(row.split_index ?? 0),
    split_count: Number(row.split_count ?? 1),
    merged_page_nos_json: normalizeJsonText(String(row.merged_page_nos_json ?? "[]")),
    merged_bboxes_json: normalizeJsonText(String(row.merged_bboxes_json ?? "[]")),
  };
}

function normalizeStoredRetrievalChunk(row: SqliteRow): StoredRetrievalChunk {
  return {
    id: String(row.id),
    document_id: String(row.document_id),
    ordinal: Number(row.ordinal ?? 0),
    content_md: String(row.content_md ?? row.chunk_text ?? ""),
    size_class: String(row.size_class ?? "normal") as DocumentChunkSizeClass,
    summary_text: row.summary_text == null ? null : String(row.summary_text),
    source_document_chunk_ids_json: normalizeJsonText(
      String(
        row.source_document_chunk_ids_json ??
          JSON.stringify(row.source_document_chunk_id ? [row.source_document_chunk_id] : []),
      ),
    ),
    page_nos_json: normalizeJsonText(String(row.page_nos_json ?? "[]")),
    source_locator: row.source_locator == null ? null : String(row.source_locator),
    bboxes_json: normalizeJsonText(String(row.bboxes_json ?? "[]")),
  };
}

function normalizeStoredImageSemanticCache(row: SqliteRow): StoredImageSemanticCache {
  return {
    image_hash: String(row.image_hash),
    prompt_version: String(row.prompt_version),
    recognizable: Boolean(row.recognizable),
    image_kind: row.image_kind == null ? null : String(row.image_kind),
    contains_text: row.contains_text == null ? null : Boolean(row.contains_text),
    visible_text: row.visible_text == null ? null : String(row.visible_text),
    summary: row.summary == null ? null : String(row.summary),
    entities_json: normalizeJsonText(String(row.entities_json ?? "[]")),
    keywords_json: normalizeJsonText(String(row.keywords_json ?? "[]")),
    qa_hints_json: normalizeJsonText(String(row.qa_hints_json ?? "[]")),
    drop_reason: row.drop_reason == null ? null : String(row.drop_reason),
    semantic_model: row.semantic_model == null ? null : String(row.semantic_model),
  };
}

function normalizeStoredDocumentParseTask(row: SqliteRow): StoredDocumentParseTask {
  return {
    id: String(row.id),
    document_id: row.document_id == null ? null : String(row.document_id),
    document_filename: String(row.document_filename ?? ""),
    task_type: String(row.task_type ?? "upload_parse") as DocumentParseTaskType,
    status: String(row.status ?? "queued") as DocumentParseTaskStatus,
    progress_percent: Number(row.progress_percent ?? 0),
    current_stage: row.current_stage == null ? null : String(row.current_stage),
    options_json: normalizeJsonText(String(row.options_json ?? "{}"), "{}"),
    stage_timings_json: normalizeJsonText(String(row.stage_timings_json ?? "[]")),
    error_message: row.error_message == null ? null : String(row.error_message),
    created_at: String(row.created_at ?? ""),
    started_at: row.started_at == null ? null : String(row.started_at),
    finished_at: row.finished_at == null ? null : String(row.finished_at),
    updated_at: String(row.updated_at ?? ""),
    total_duration_ms: row.total_duration_ms == null ? null : Number(row.total_duration_ms),
  };
}

export class SqliteStorage implements SqliteStorageBackend {
  private readonly db: Database.Database;

  constructor(private readonly options: SqliteStorageOptions) {
    if (!options.readOnly) {
      mkdirSync(dirname(options.dbPath), { recursive: true });
    }
    this.db = new Database(options.dbPath, {
      readonly: options.readOnly ?? false,
      fileMustExist: options.readOnly ?? false,
    });
    this.db.pragma("foreign_keys = ON");
    if (!options.readOnly) {
      this.db.pragma("journal_mode = WAL");
    }
  }

  static stableId(prefix: string, value: string): string {
    return stableId(prefix, value);
  }

  static makeDocumentId(corpusId: string, relativePath: string): string {
    return stableId("doc", `${corpusId}:${relativePath}`);
  }

  initialize(): void {
    const migrate = this.db.transaction(() => {
      this.createSchemaTables();
      this.migrateSchemaColumns({ invalidateTraditionalRagIndex: false });
      this.backfillSchemaDefaults();
      this.createSchemaIndexes();
    });
    migrate();
  }

  private createSchemaTables(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS collections (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'user',
        corpus_id TEXT,
        root_path TEXT,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        corpus_id TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        absolute_path TEXT NOT NULL,
        original_filename TEXT NOT NULL DEFAULT '',
        object_key TEXT NOT NULL DEFAULT '',
        source_object_key TEXT NOT NULL DEFAULT '',
        pages_prefix TEXT NOT NULL DEFAULT '',
        storage_uri TEXT NOT NULL DEFAULT '',
        content_type TEXT,
        upload_status TEXT NOT NULL DEFAULT 'uploaded',
        page_count INTEGER NOT NULL DEFAULT 0,
        content TEXT NOT NULL DEFAULT '',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        file_mtime REAL NOT NULL,
        file_size INTEGER NOT NULL,
        content_sha256 TEXT NOT NULL,
        parsed_content_sha256 TEXT,
        parsed_is_complete INTEGER NOT NULL DEFAULT 0,
        embedding_enabled INTEGER NOT NULL DEFAULT 1,
        has_embeddings INTEGER NOT NULL DEFAULT 0,
        image_semantic_enabled INTEGER NOT NULL DEFAULT 1,
        last_indexed_at TEXT NOT NULL,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        UNIQUE(corpus_id, relative_path)
      );

      CREATE TABLE IF NOT EXISTS collection_documents (
        collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        PRIMARY KEY (collection_id, document_id)
      );

      CREATE TABLE IF NOT EXISTS document_pages (
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        page_no INTEGER NOT NULL,
        object_key TEXT NOT NULL,
        heading TEXT,
        source_locator TEXT,
        content_hash TEXT NOT NULL,
        char_count INTEGER NOT NULL DEFAULT 0,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        is_synthetic_page INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (document_id, page_no)
      );

      CREATE TABLE IF NOT EXISTS document_chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL DEFAULT 0,
        reference_retrieval_chunk_id TEXT,
        page_no INTEGER NOT NULL,
        document_index INTEGER NOT NULL,
        page_index INTEGER NOT NULL,
        block_type TEXT NOT NULL,
        bbox_json TEXT NOT NULL,
        content_md TEXT NOT NULL,
        size_class TEXT NOT NULL,
        summary_text TEXT,
        is_split_from_oversized INTEGER NOT NULL DEFAULT 0,
        split_index INTEGER NOT NULL DEFAULT 0,
        split_count INTEGER NOT NULL DEFAULT 1,
        merged_page_nos_json TEXT NOT NULL DEFAULT '[]',
        merged_bboxes_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS retrieval_chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        source_document_chunk_id TEXT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        content_md TEXT NOT NULL DEFAULT '',
        size_class TEXT NOT NULL,
        summary_text TEXT,
        source_document_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
        page_nos_json TEXT NOT NULL DEFAULT '[]',
        source_locator TEXT,
        bboxes_json TEXT NOT NULL DEFAULT '[]',
        is_split_from_oversized INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS image_semantics (
        image_hash TEXT PRIMARY KEY,
        source_document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        source_page_no INTEGER NOT NULL,
        source_image_index INTEGER NOT NULL,
        mime_type TEXT,
        width INTEGER,
        height INTEGER,
        bbox_json TEXT,
        object_key TEXT,
        storage_uri TEXT,
        has_text INTEGER,
        interference_score REAL,
        is_dropped INTEGER NOT NULL DEFAULT 0,
        recognizable INTEGER,
        accessible_url TEXT,
        semantic_text TEXT,
        semantic_model TEXT,
        last_enhanced_at TEXT
      );

      CREATE TABLE IF NOT EXISTS image_semantic_cache (
        image_hash TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        recognizable INTEGER NOT NULL,
        image_kind TEXT,
        contains_text INTEGER,
        visible_text TEXT,
        summary TEXT,
        entities_json TEXT NOT NULL DEFAULT '[]',
        keywords_json TEXT NOT NULL DEFAULT '[]',
        qa_hints_json TEXT NOT NULL DEFAULT '[]',
        drop_reason TEXT,
        semantic_model TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (image_hash, prompt_version)
      );

      CREATE TABLE IF NOT EXISTS document_parse_tasks (
        id TEXT PRIMARY KEY,
        document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
        document_filename TEXT NOT NULL,
        task_type TEXT NOT NULL,
        status TEXT NOT NULL,
        progress_percent INTEGER NOT NULL DEFAULT 0,
        current_stage TEXT,
        options_json TEXT NOT NULL DEFAULT '{}',
        stage_timings_json TEXT NOT NULL DEFAULT '[]',
        error_message TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        updated_at TEXT NOT NULL,
        total_duration_ms INTEGER
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_chunks_fts USING fts5(
        chunk_id UNINDEXED,
        document_id UNINDEXED,
        chunk_text
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
        chunk_id UNINDEXED,
        document_id UNINDEXED,
        chunk_text
      );

    `);
  }

  private createSchemaIndexes(): void {
    this.db.exec(`
      CREATE UNIQUE INDEX IF NOT EXISTS idx_collections_scope_root
      ON collections (root_path)
      WHERE kind = 'corpus_scope';

      CREATE UNIQUE INDEX IF NOT EXISTS idx_collections_scope_corpus
      ON collections (corpus_id)
      WHERE kind = 'corpus_scope';

      CREATE INDEX IF NOT EXISTS idx_documents_corpus_deleted
      ON documents (corpus_id, is_deleted);

      CREATE INDEX IF NOT EXISTS idx_collection_documents_collection
      ON collection_documents (collection_id, document_id);

      CREATE INDEX IF NOT EXISTS idx_document_pages_document
      ON document_pages (document_id, page_no);

      CREATE INDEX IF NOT EXISTS idx_document_chunks_document
      ON document_chunks (document_id, document_index);

      CREATE INDEX IF NOT EXISTS idx_document_chunks_ordinal
      ON document_chunks (document_id, ordinal);

      CREATE INDEX IF NOT EXISTS idx_document_chunks_reference
      ON document_chunks (reference_retrieval_chunk_id);

      CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_document
      ON retrieval_chunks (document_id, ordinal);

      CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_source
      ON retrieval_chunks (source_document_chunk_id);

      CREATE INDEX IF NOT EXISTS idx_image_semantics_source_page
      ON image_semantics (source_document_id, source_page_no);

      CREATE INDEX IF NOT EXISTS idx_image_semantic_cache_hash
      ON image_semantic_cache (image_hash, prompt_version);

      CREATE INDEX IF NOT EXISTS idx_document_parse_tasks_status
      ON document_parse_tasks (status, created_at DESC);

      CREATE INDEX IF NOT EXISTS idx_document_parse_tasks_document
      ON document_parse_tasks (document_id, created_at DESC);
    `);
  }

  private migrateSchemaColumns(input: { invalidateTraditionalRagIndex: boolean }): void {
    let invalidateTraditionalRagIndex = input.invalidateTraditionalRagIndex;
    const addColumn = (tableName: string, columnName: string, definition: string) => {
      const added = this.addColumnIfMissing(tableName, columnName, definition);
      return added;
    };

    addColumn("collections", "kind", "TEXT NOT NULL DEFAULT 'user'");
    addColumn("collections", "corpus_id", "TEXT");
    addColumn("collections", "root_path", "TEXT");
    addColumn("collections", "is_deleted", "INTEGER NOT NULL DEFAULT 0");
    addColumn("collections", "created_at", "TEXT NOT NULL DEFAULT ''");
    addColumn("collections", "updated_at", "TEXT NOT NULL DEFAULT ''");

    addColumn("documents", "corpus_id", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "relative_path", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "absolute_path", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "original_filename", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "object_key", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "source_object_key", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "pages_prefix", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "storage_uri", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "content_type", "TEXT");
    addColumn("documents", "upload_status", "TEXT NOT NULL DEFAULT 'uploaded'");
    addColumn("documents", "page_count", "INTEGER NOT NULL DEFAULT 0");
    addColumn("documents", "content", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "metadata_json", "TEXT NOT NULL DEFAULT '{}'");
    addColumn("documents", "file_mtime", "REAL NOT NULL DEFAULT 0");
    addColumn("documents", "file_size", "INTEGER NOT NULL DEFAULT 0");
    addColumn("documents", "content_sha256", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "parsed_content_sha256", "TEXT");
    addColumn("documents", "parsed_is_complete", "INTEGER NOT NULL DEFAULT 0");
    addColumn("documents", "embedding_enabled", "INTEGER NOT NULL DEFAULT 1");
    addColumn("documents", "has_embeddings", "INTEGER NOT NULL DEFAULT 0");
    addColumn("documents", "image_semantic_enabled", "INTEGER NOT NULL DEFAULT 1");
    addColumn("documents", "last_indexed_at", "TEXT NOT NULL DEFAULT ''");
    addColumn("documents", "is_deleted", "INTEGER NOT NULL DEFAULT 0");

    addColumn("document_pages", "object_key", "TEXT NOT NULL DEFAULT ''");
    addColumn("document_pages", "heading", "TEXT");
    addColumn("document_pages", "source_locator", "TEXT");
    addColumn("document_pages", "content_hash", "TEXT NOT NULL DEFAULT ''");
    addColumn("document_pages", "char_count", "INTEGER NOT NULL DEFAULT 0");
    addColumn("document_pages", "chunk_count", "INTEGER NOT NULL DEFAULT 0");
    addColumn("document_pages", "is_synthetic_page", "INTEGER NOT NULL DEFAULT 0");
    addColumn("document_pages", "updated_at", "TEXT NOT NULL DEFAULT ''");

    invalidateTraditionalRagIndex =
      addColumn("document_chunks", "ordinal", "INTEGER NOT NULL DEFAULT 0") || invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("document_chunks", "reference_retrieval_chunk_id", "TEXT") || invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("document_chunks", "is_split_from_oversized", "INTEGER NOT NULL DEFAULT 0") ||
      invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("document_chunks", "split_index", "INTEGER NOT NULL DEFAULT 0") || invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("document_chunks", "split_count", "INTEGER NOT NULL DEFAULT 1") || invalidateTraditionalRagIndex;

    invalidateTraditionalRagIndex =
      addColumn("retrieval_chunks", "content_md", "TEXT NOT NULL DEFAULT ''") || invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("retrieval_chunks", "summary_text", "TEXT") || invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("retrieval_chunks", "source_document_chunk_ids_json", "TEXT NOT NULL DEFAULT '[]'") ||
      invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("retrieval_chunks", "page_nos_json", "TEXT NOT NULL DEFAULT '[]'") || invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("retrieval_chunks", "source_locator", "TEXT") || invalidateTraditionalRagIndex;
    invalidateTraditionalRagIndex =
      addColumn("retrieval_chunks", "bboxes_json", "TEXT NOT NULL DEFAULT '[]'") || invalidateTraditionalRagIndex;

    addColumn("image_semantics", "source_document_id", "TEXT NOT NULL DEFAULT ''");
    addColumn("image_semantics", "source_page_no", "INTEGER NOT NULL DEFAULT 0");
    addColumn("image_semantics", "source_image_index", "INTEGER NOT NULL DEFAULT 0");
    addColumn("image_semantics", "mime_type", "TEXT");
    addColumn("image_semantics", "width", "INTEGER");
    addColumn("image_semantics", "height", "INTEGER");
    addColumn("image_semantics", "bbox_json", "TEXT");
    addColumn("image_semantics", "object_key", "TEXT");
    addColumn("image_semantics", "storage_uri", "TEXT");
    addColumn("image_semantics", "has_text", "INTEGER");
    addColumn("image_semantics", "interference_score", "REAL");
    addColumn("image_semantics", "is_dropped", "INTEGER NOT NULL DEFAULT 0");
    addColumn("image_semantics", "recognizable", "INTEGER");
    addColumn("image_semantics", "accessible_url", "TEXT");
    addColumn("image_semantics", "semantic_text", "TEXT");
    addColumn("image_semantics", "semantic_model", "TEXT");
    addColumn("image_semantics", "last_enhanced_at", "TEXT");

    addColumn("document_parse_tasks", "document_id", "TEXT");
    addColumn("document_parse_tasks", "document_filename", "TEXT NOT NULL DEFAULT ''");
    addColumn("document_parse_tasks", "task_type", "TEXT NOT NULL DEFAULT 'upload_parse'");
    addColumn("document_parse_tasks", "status", "TEXT NOT NULL DEFAULT 'queued'");
    addColumn("document_parse_tasks", "progress_percent", "INTEGER NOT NULL DEFAULT 0");
    addColumn("document_parse_tasks", "current_stage", "TEXT");
    addColumn("document_parse_tasks", "options_json", "TEXT NOT NULL DEFAULT '{}'");
    addColumn("document_parse_tasks", "stage_timings_json", "TEXT NOT NULL DEFAULT '[]'");
    addColumn("document_parse_tasks", "error_message", "TEXT");
    addColumn("document_parse_tasks", "created_at", "TEXT NOT NULL DEFAULT ''");
    addColumn("document_parse_tasks", "started_at", "TEXT");
    addColumn("document_parse_tasks", "finished_at", "TEXT");
    addColumn("document_parse_tasks", "updated_at", "TEXT NOT NULL DEFAULT ''");
    addColumn("document_parse_tasks", "total_duration_ms", "INTEGER");

    if (invalidateTraditionalRagIndex) {
      this.invalidateTraditionalRagIndexState();
    }
  }

  private addColumnIfMissing(tableName: string, columnName: string, definition: string): boolean {
    const columns = this.tableColumns(tableName);
    if (columns.has(columnName)) {
      return false;
    }
    this.db.exec(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition}`);
    return true;
  }

  private backfillSchemaDefaults(): void {
    this.db.exec(`
      UPDATE documents
      SET original_filename = relative_path
      WHERE original_filename = '' AND relative_path <> '';

      UPDATE documents
      SET source_object_key = object_key
      WHERE source_object_key = '' AND object_key <> '';

      UPDATE documents
      SET upload_status = 'uploaded'
      WHERE upload_status = '';

      UPDATE documents
      SET embedding_enabled = 1
      WHERE embedding_enabled NOT IN (0, 1);

      UPDATE documents
      SET has_embeddings = 0
      WHERE has_embeddings NOT IN (0, 1);

      UPDATE documents
      SET image_semantic_enabled = 1
      WHERE image_semantic_enabled NOT IN (0, 1);

      UPDATE collections
      SET kind = 'user'
      WHERE kind = '';

      UPDATE collections
      SET created_at = updated_at
      WHERE created_at = '' AND updated_at <> '';

      UPDATE collections
      SET updated_at = created_at
      WHERE updated_at = '' AND created_at <> '';
    `);
  }

  private invalidateTraditionalRagIndexState(): void {
    this.db.exec(`
      DELETE FROM document_chunks_fts;
      DELETE FROM retrieval_chunks_fts;
      DELETE FROM retrieval_chunks;
      DELETE FROM document_chunks;
      UPDATE documents
      SET has_embeddings = 0
      WHERE has_embeddings <> 0;
    `);
  }

  private tableExists(tableName: string): boolean {
    const row = this.db
      .prepare("SELECT 1 AS found FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1")
      .get(tableName) as SqliteRow | undefined;
    return Boolean(row?.found);
  }

  private tableColumns(tableName: string): Set<string> {
    const rows = this.db.prepare(`PRAGMA table_info(${tableName})`).all() as SqliteRow[];
    return new Set(rows.map((row) => String(row.name)));
  }

  close(): void {
    this.db.close();
  }

  getOrCreateCorpus(rootPath: string): string {
    const corpusId = stableId("corpus", rootPath);
    const collectionId = stableId("collection", `scope:${rootPath}`);
    const now = utcNowIso();
    this.db
      .prepare(
        `
          INSERT OR IGNORE INTO collections (
            id, name, kind, corpus_id, root_path, is_deleted, created_at, updated_at
          )
          VALUES (?, ?, 'corpus_scope', ?, ?, 0, ?, ?)
        `,
      )
      .run(collectionId, `__corpus_scope__:${rootPath}`, corpusId, rootPath, now, now);
    const row = this.db
      .prepare(
        `
          SELECT corpus_id
          FROM collections
          WHERE kind = 'corpus_scope' AND root_path = ?
          LIMIT 1
        `,
      )
      .get(rootPath) as SqliteRow | undefined;
    if (!row?.corpus_id) {
      throw new Error(`Failed to create corpus for path: ${rootPath}`);
    }
    return String(row.corpus_id);
  }

  getCorpusId(rootPath: string): string | null {
    const row = this.db
      .prepare(
        `
          SELECT corpus_id
          FROM collections
          WHERE kind = 'corpus_scope' AND root_path = ?
          LIMIT 1
        `,
      )
      .get(rootPath) as SqliteRow | undefined;
    return row?.corpus_id == null ? null : String(row.corpus_id);
  }

  getCorpusRootPath(corpusId: string): string | null {
    const row = this.db
      .prepare(
        `
          SELECT root_path
          FROM collections
          WHERE kind = 'corpus_scope' AND corpus_id = ?
          LIMIT 1
        `,
      )
      .get(corpusId) as SqliteRow | undefined;
    return row?.root_path == null ? null : String(row.root_path);
  }

  upsertDocumentStub(document: StorageDocumentRecord): void {
    const now = utcNowIso();
    const run = this.db.transaction(() => {
      this.db
        .prepare(
          `
            INSERT INTO documents (
              id, corpus_id, relative_path, absolute_path, original_filename, object_key,
              source_object_key, pages_prefix, storage_uri, content_type, upload_status,
              page_count, content, metadata_json, file_mtime, file_size, content_sha256,
              parsed_content_sha256, parsed_is_complete, embedding_enabled, has_embeddings,
              image_semantic_enabled, last_indexed_at, is_deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(id) DO UPDATE SET
              corpus_id = excluded.corpus_id,
              relative_path = excluded.relative_path,
              absolute_path = excluded.absolute_path,
              original_filename = excluded.original_filename,
              object_key = excluded.object_key,
              source_object_key = excluded.source_object_key,
              pages_prefix = excluded.pages_prefix,
              storage_uri = excluded.storage_uri,
              content_type = excluded.content_type,
              upload_status = excluded.upload_status,
              page_count = excluded.page_count,
              content = excluded.content,
              metadata_json = excluded.metadata_json,
              file_mtime = excluded.file_mtime,
              file_size = excluded.file_size,
              content_sha256 = excluded.content_sha256,
              parsed_content_sha256 = excluded.parsed_content_sha256,
              parsed_is_complete = excluded.parsed_is_complete,
              embedding_enabled = excluded.embedding_enabled,
              has_embeddings = excluded.has_embeddings,
              image_semantic_enabled = excluded.image_semantic_enabled,
              last_indexed_at = excluded.last_indexed_at,
              is_deleted = 0
          `,
        )
        .run(
          document.id,
          document.corpusId,
          document.relativePath,
          document.absolutePath,
          document.originalFilename ?? document.relativePath,
          document.objectKey ?? "",
          document.sourceObjectKey ?? document.objectKey ?? "",
          document.pagesPrefix ?? "",
          document.storageUri ?? "",
          document.contentType ?? null,
          document.uploadStatus ?? "uploaded",
          document.pageCount ?? 0,
          document.content ?? "",
          normalizeMetadataJson(document.metadataJson),
          document.fileMtime,
          document.fileSize,
          document.contentSha256,
          document.parsedContentSha256 ?? null,
          document.parsedIsComplete ? 1 : 0,
          document.embeddingEnabled == null ? 1 : document.embeddingEnabled ? 1 : 0,
          document.hasEmbeddings ? 1 : 0,
          document.imageSemanticEnabled == null ? 1 : document.imageSemanticEnabled ? 1 : 0,
          now,
        );

      // Reference: legacy/python/src/fs_explorer/storage/postgres.py upsert_document_stub clears stale page/image rows.
      this.db.prepare("DELETE FROM document_pages WHERE document_id = ?").run(document.id);
      this.db.prepare("DELETE FROM retrieval_chunks_fts WHERE document_id = ?").run(document.id);
      this.db.prepare("DELETE FROM document_chunks_fts WHERE document_id = ?").run(document.id);
      this.db.prepare("DELETE FROM retrieval_chunks WHERE document_id = ?").run(document.id);
      this.db.prepare("DELETE FROM document_chunks WHERE document_id = ?").run(document.id);
      this.db.prepare("DELETE FROM image_semantics WHERE source_document_id = ?").run(document.id);
    });

    run();
  }

  getDocument(docId: string): StoredDocument | null {
    const row = this.db.prepare("SELECT * FROM documents WHERE id = ? LIMIT 1").get(docId) as
      | SqliteRow
      | undefined;
    return row ? normalizeStoredDocument(row) : null;
  }

  listDocuments(corpusId: string, includeDeleted = false): StoredDocument[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM documents
          WHERE corpus_id = ?
            ${includeDeleted ? "" : "AND is_deleted = 0"}
          ORDER BY relative_path
        `,
      )
      .all(corpusId) as SqliteRow[];
    return rows.map(normalizeStoredDocument);
  }

  listDocumentsByIds(docIds: string[], includeDeleted = false): StoredDocument[] {
    if (docIds.length === 0) {
      return [];
    }
    const placeholders = docIds.map(() => "?").join(", ");
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM documents
          WHERE id IN (${placeholders})
            ${includeDeleted ? "" : "AND is_deleted = 0"}
          ORDER BY relative_path
        `,
      )
      .all(...docIds) as SqliteRow[];
    return rows.map(normalizeStoredDocument);
  }

  setDocumentDeleted(docId: string, isDeleted: boolean): StoredDocument | null {
    const result = this.db
      .prepare("UPDATE documents SET is_deleted = ?, last_indexed_at = ? WHERE id = ?")
      .run(isDeleted ? 1 : 0, utcNowIso(), docId);
    if (result.changes === 0) {
      return null;
    }
    return this.getDocument(docId);
  }

  deleteDocument(docId: string): StoredDocument | null {
    const existing = this.getDocument(docId);
    if (!existing) {
      return null;
    }
    this.db.prepare("DELETE FROM retrieval_chunks_fts WHERE document_id = ?").run(docId);
    this.db.prepare("DELETE FROM document_chunks_fts WHERE document_id = ?").run(docId);
    this.db.prepare("DELETE FROM documents WHERE id = ?").run(docId);
    return existing;
  }

  updateDocumentAbsolutePath(docId: string, absolutePath: string): StoredDocument | null {
    const result = this.db
      .prepare("UPDATE documents SET absolute_path = ? WHERE id = ?")
      .run(absolutePath, docId);
    if (result.changes === 0) {
      return null;
    }
    return this.getDocument(docId);
  }

  updateDocumentMetadata(docId: string, metadataJson: string): StoredDocument | null {
    const result = this.db
      .prepare("UPDATE documents SET metadata_json = ? WHERE id = ?")
      .run(normalizeMetadataJson(metadataJson), docId);
    if (result.changes === 0) {
      return null;
    }
    return this.getDocument(docId);
  }

  updateDocumentParseState(
    docId: string,
    parsedContentSha256: string | null,
    parsedIsComplete: boolean,
  ): StoredDocument | null {
    const result = this.db
      .prepare(
        `
          UPDATE documents
          SET parsed_content_sha256 = ?,
              parsed_is_complete = CASE
                WHEN ? = 1 THEN 1
                ELSE parsed_is_complete
              END,
              last_indexed_at = ?
          WHERE id = ?
        `,
      )
      .run(parsedContentSha256, parsedIsComplete ? 1 : 0, utcNowIso(), docId);
    if (result.changes === 0) {
      return null;
    }
    return this.getDocument(docId);
  }

  updateDocumentUploadStatus(docId: string, uploadStatus: string): StoredDocument | null {
    const result = this.db
      .prepare("UPDATE documents SET upload_status = ?, last_indexed_at = ? WHERE id = ?")
      .run(uploadStatus, utcNowIso(), docId);
    if (result.changes === 0) {
      return null;
    }
    return this.getDocument(docId);
  }

  updateDocumentFeatureFlags(
    docId: string,
    input: {
      embeddingEnabled?: boolean;
      hasEmbeddings?: boolean;
      imageSemanticEnabled?: boolean;
    },
  ): StoredDocument | null {
    const updates: string[] = [];
    const params: Array<string | number> = [];
    if (input.embeddingEnabled != null) {
      updates.push("embedding_enabled = ?");
      params.push(input.embeddingEnabled ? 1 : 0);
    }
    if (input.hasEmbeddings != null) {
      updates.push("has_embeddings = ?");
      params.push(input.hasEmbeddings ? 1 : 0);
    }
    if (input.imageSemanticEnabled != null) {
      updates.push("image_semantic_enabled = ?");
      params.push(input.imageSemanticEnabled ? 1 : 0);
    }
    if (updates.length === 0) {
      return this.getDocument(docId);
    }
    updates.push("last_indexed_at = ?");
    params.push(utcNowIso(), docId);
    const result = this.db
      .prepare(`UPDATE documents SET ${updates.join(", ")} WHERE id = ?`)
      .run(...params);
    if (result.changes === 0) {
      return null;
    }
    return this.getDocument(docId);
  }

  listDocumentPages(documentId: string, pageNos?: number[] | null): StoredDocumentPage[] {
    if (pageNos && pageNos.length === 0) {
      return [];
    }
    const params: Array<string | number> = [documentId];
    let sql = `
      SELECT document_id, page_no, object_key, heading, source_locator, content_hash, char_count, chunk_count, is_synthetic_page
      FROM document_pages
      WHERE document_id = ?
    `;
    if (pageNos && pageNos.length > 0) {
      const uniquePageNos = [...new Set(pageNos)].sort((a, b) => a - b);
      sql += ` AND page_no IN (${uniquePageNos.map(() => "?").join(", ")})`;
      params.push(...uniquePageNos);
    }
    sql += " ORDER BY page_no ASC";
    const rows = this.db.prepare(sql).all(...params) as SqliteRow[];
    return rows.map(normalizeStoredDocumentPage);
  }

  replaceDocumentChunks(
    documentId: string,
    chunks: StorageDocumentChunkRecord[],
  ): { inserted: number; deleted: number } {
    const deleted = Number(
      this.db.prepare("DELETE FROM document_chunks WHERE document_id = ?").run(documentId).changes ?? 0,
    );
    this.db.prepare("DELETE FROM document_chunks_fts WHERE document_id = ?").run(documentId);
    if (chunks.length === 0) {
      return { inserted: 0, deleted };
    }
    const now = utcNowIso();
    const run = this.db.transaction(() => {
        const statement = this.db.prepare(
          `
          INSERT INTO document_chunks (
            id, document_id, ordinal, reference_retrieval_chunk_id, page_no, document_index, page_index,
            block_type, bbox_json, content_md, size_class, summary_text, is_split_from_oversized,
            split_index, split_count, merged_page_nos_json, merged_bboxes_json, created_at, updated_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
        );
      const ftsStatement = this.db.prepare(
        `
          INSERT INTO document_chunks_fts (chunk_id, document_id, chunk_text)
          VALUES (?, ?, ?)
        `,
      );
      for (const chunk of chunks) {
        statement.run(
          chunk.id,
          chunk.documentId,
          chunk.ordinal ?? 0,
          chunk.referenceRetrievalChunkId ?? null,
          chunk.pageNo,
          chunk.documentIndex,
          chunk.pageIndex,
          chunk.blockType,
          chunk.bboxJson,
          chunk.contentMd,
          chunk.sizeClass,
          chunk.summaryText ?? null,
          chunk.isSplitFromOversized ? 1 : 0,
          chunk.splitIndex ?? 0,
          chunk.splitCount ?? 1,
          chunk.mergedPageNosJson ?? "[]",
          chunk.mergedBboxesJson ?? "[]",
          now,
          now,
        );
        ftsStatement.run(chunk.id, chunk.documentId, chunk.contentMd);
      }
    });
    run();
    return { inserted: chunks.length, deleted };
  }

  listDocumentChunks(documentId: string): StoredDocumentChunk[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM document_chunks
          WHERE document_id = ?
          ORDER BY document_index ASC, split_index ASC, ordinal ASC
        `,
      )
      .all(documentId) as SqliteRow[];
    return rows.map(normalizeStoredDocumentChunk);
  }

  getDocumentChunk(chunkId: string): StoredDocumentChunk | null {
    const row = this.db
      .prepare("SELECT * FROM document_chunks WHERE id = ? LIMIT 1")
      .get(chunkId) as SqliteRow | undefined;
    return row ? normalizeStoredDocumentChunk(row) : null;
  }

  replaceRetrievalChunks(
    documentId: string,
    chunks: StorageRetrievalChunkRecord[],
  ): { inserted: number; deleted: number } {
    const deletedFts = Number(
      this.db.prepare("DELETE FROM retrieval_chunks_fts WHERE document_id = ?").run(documentId).changes ?? 0,
    );
    const deletedChunks = Number(
      this.db.prepare("DELETE FROM retrieval_chunks WHERE document_id = ?").run(documentId).changes ?? 0,
    );
    if (chunks.length === 0) {
      return { inserted: 0, deleted: Math.max(deletedFts, deletedChunks) };
    }
    const now = utcNowIso();
    const run = this.db.transaction(() => {
      const chunkStatement = this.db.prepare(
        `
          INSERT INTO retrieval_chunks (
            id, document_id, source_document_chunk_id, ordinal, chunk_text, content_md,
            size_class, summary_text, source_document_chunk_ids_json, page_nos_json, source_locator, bboxes_json,
            is_split_from_oversized, created_at, updated_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      );
      for (const chunk of chunks) {
        const sourceDocumentChunkIds = JSON.parse(chunk.sourceDocumentChunkIdsJson || "[]") as string[];
        chunkStatement.run(
          chunk.id,
          chunk.documentId,
          sourceDocumentChunkIds[0] ?? "",
          chunk.ordinal,
          chunk.contentMd,
          chunk.contentMd,
          chunk.sizeClass,
          chunk.summaryText ?? null,
          chunk.sourceDocumentChunkIdsJson,
          chunk.pageNosJson,
          chunk.sourceLocator ?? null,
          chunk.bboxesJson ?? "[]",
          0,
          now,
          now,
        );
      }
    });
    run();
    return { inserted: chunks.length, deleted: Math.max(deletedFts, deletedChunks) };
  }

  getRetrievalChunk(chunkId: string): StoredRetrievalChunk | null {
    const row = this.db
      .prepare("SELECT * FROM retrieval_chunks WHERE id = ? LIMIT 1")
      .get(chunkId) as SqliteRow | undefined;
    return row ? normalizeStoredRetrievalChunk(row) : null;
  }

  listRetrievalChunks(documentId: string): StoredRetrievalChunk[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM retrieval_chunks
          WHERE document_id = ?
          ORDER BY ordinal ASC
        `,
      )
      .all(documentId) as SqliteRow[];
    return rows.map(normalizeStoredRetrievalChunk);
  }

  keywordSearchDocumentChunks(input: {
    query: string;
    documentIds: string[];
    limit: number;
  }): DocumentChunkKeywordHit[] {
    const documentIds = [...new Set(input.documentIds.map((item) => String(item).trim()).filter(Boolean))];
    if (!input.query.trim() || documentIds.length === 0) {
      return [];
    }
    const placeholders = documentIds.map(() => "?").join(", ");
    const rows = this.db
      .prepare(
        `
          SELECT
            dc.id AS document_chunk_id,
            dc.document_id,
            dc.reference_retrieval_chunk_id,
            dc.ordinal,
            dc.content_md,
            dc.size_class,
            dc.is_split_from_oversized,
            -bm25(document_chunks_fts) AS score
          FROM document_chunks_fts
          JOIN document_chunks dc ON dc.id = document_chunks_fts.chunk_id
          WHERE document_chunks_fts MATCH ?
            AND dc.document_id IN (${placeholders})
          ORDER BY score DESC, dc.ordinal ASC
          LIMIT ?
        `,
      )
      .all(input.query, ...documentIds, Math.max(Number(input.limit || 10), 1)) as SqliteRow[];
    return rows.map((row) => ({
      document_chunk_id: String(row.document_chunk_id),
      document_id: String(row.document_id),
      reference_retrieval_chunk_id:
        row.reference_retrieval_chunk_id == null ? null : String(row.reference_retrieval_chunk_id),
      ordinal: Number(row.ordinal ?? 0),
      content_md: String(row.content_md ?? ""),
      size_class: String(row.size_class ?? "normal") as DocumentChunkSizeClass,
      is_split_from_oversized: Boolean(row.is_split_from_oversized),
      score: Number(row.score ?? 0),
    }));
  }

  syncDocumentPages(
    documentId: string,
    pages: StorageDocumentPageRecord[],
  ): { upserted: number; untouched: number; deleted: number } {
    const existing = new Map(
      this.listDocumentPages(documentId).map((item) => [item.page_no, item] as const),
    );
    const desiredPages = new Set(pages.map((page) => page.pageNo));

    let upserted = 0;
    let untouched = 0;
    let deleted = 0;

    const run = this.db.transaction(() => {
      for (const page of pages) {
        const previous = existing.get(page.pageNo);
        const current = {
          object_key: page.objectKey,
          heading: page.heading ?? null,
          source_locator: page.sourceLocator ?? null,
          content_hash: page.contentHash,
          char_count: page.charCount,
          chunk_count: page.chunkCount ?? 0,
          is_synthetic_page: page.isSyntheticPage,
        };
        if (
          previous &&
          previous.object_key === current.object_key &&
          previous.heading === current.heading &&
          previous.source_locator === current.source_locator &&
          previous.content_hash === current.content_hash &&
          previous.char_count === current.char_count &&
          previous.chunk_count === current.chunk_count &&
          previous.is_synthetic_page === current.is_synthetic_page
        ) {
          untouched += 1;
          continue;
        }
        this.db
          .prepare(
            `
              INSERT INTO document_pages (
                document_id, page_no, object_key, heading, source_locator,
                content_hash, char_count, chunk_count, is_synthetic_page, updated_at
              )
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              ON CONFLICT(document_id, page_no) DO UPDATE SET
                object_key = excluded.object_key,
                heading = excluded.heading,
                source_locator = excluded.source_locator,
                content_hash = excluded.content_hash,
                char_count = excluded.char_count,
                chunk_count = excluded.chunk_count,
                is_synthetic_page = excluded.is_synthetic_page,
                updated_at = excluded.updated_at
            `,
          )
          .run(
            documentId,
            page.pageNo,
            page.objectKey,
            page.heading ?? null,
            page.sourceLocator ?? null,
            page.contentHash,
            page.charCount,
            page.chunkCount ?? 0,
            page.isSyntheticPage ? 1 : 0,
            utcNowIso(),
          );
        upserted += 1;
      }

      if (desiredPages.size > 0) {
        const sortedPages = [...desiredPages].sort((a, b) => a - b);
        const result = this.db
          .prepare(
            `
              DELETE FROM document_pages
              WHERE document_id = ?
                AND page_no NOT IN (${sortedPages.map(() => "?").join(", ")})
            `,
          )
          .run(documentId, ...sortedPages);
        deleted = Number(result.changes ?? 0);
      } else {
        const result = this.db
          .prepare("DELETE FROM document_pages WHERE document_id = ?")
          .run(documentId);
        deleted = Number(result.changes ?? 0);
      }

      this.db
        .prepare("UPDATE documents SET page_count = ?, last_indexed_at = ? WHERE id = ?")
        .run(pages.length, utcNowIso(), documentId);
    });

    run();
    return { upserted, untouched, deleted };
  }

  upsertImageSemantics(images: StorageImageSemanticRecord[]): number {
    if (images.length === 0) {
      return 0;
    }
    const run = this.db.transaction(() => {
      const statement = this.db.prepare(
        `
          INSERT INTO image_semantics (
            image_hash, source_document_id, source_page_no, source_image_index,
            mime_type, width, height, bbox_json, object_key, storage_uri,
            has_text, interference_score, is_dropped, recognizable, accessible_url,
            semantic_text, semantic_model
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(image_hash) DO UPDATE SET
            mime_type = COALESCE(image_semantics.mime_type, excluded.mime_type),
            width = COALESCE(image_semantics.width, excluded.width),
            height = COALESCE(image_semantics.height, excluded.height),
            bbox_json = COALESCE(excluded.bbox_json, image_semantics.bbox_json),
            object_key = COALESCE(excluded.object_key, image_semantics.object_key),
            storage_uri = COALESCE(excluded.storage_uri, image_semantics.storage_uri),
            has_text = COALESCE(excluded.has_text, image_semantics.has_text),
            interference_score = COALESCE(excluded.interference_score, image_semantics.interference_score),
            is_dropped = COALESCE(excluded.is_dropped, image_semantics.is_dropped),
            recognizable = COALESCE(excluded.recognizable, image_semantics.recognizable),
            accessible_url = COALESCE(excluded.accessible_url, image_semantics.accessible_url)
        `,
      );
      for (const image of images) {
        statement.run(
          image.imageHash,
          image.sourceDocumentId,
          image.sourcePageNo,
          image.sourceImageIndex,
          image.mimeType ?? null,
          image.width ?? null,
          image.height ?? null,
          image.bboxJson ?? null,
          image.objectKey ?? null,
          image.storageUri ?? null,
          image.hasText == null ? null : image.hasText ? 1 : 0,
          image.interferenceScore ?? null,
          image.isDropped ? 1 : 0,
          image.recognizable == null ? null : image.recognizable ? 1 : 0,
          image.accessibleUrl ?? null,
          image.semanticText ?? null,
          image.semanticModel ?? null,
        );
      }
    });
    run();
    return images.length;
  }

  getImageSemantics(imageHashes: string[]): Record<string, StoredImageSemantic> {
    if (imageHashes.length === 0) {
      return {};
    }
    const placeholders = imageHashes.map(() => "?").join(", ");
    const rows = this.db
      .prepare(
        `
          SELECT
            image_hash, source_document_id, source_page_no, source_image_index,
            mime_type, width, height, bbox_json, object_key, storage_uri,
            has_text, interference_score, is_dropped, recognizable, accessible_url,
            semantic_text, semantic_model
          FROM image_semantics
          WHERE image_hash IN (${placeholders})
        `,
      )
      .all(...imageHashes) as SqliteRow[];
    return Object.fromEntries(rows.map((row) => [String(row.image_hash), normalizeStoredImageSemantic(row)]));
  }

  listImageSemanticsForDocument(documentId: string, pageNos?: number[] | null): StoredImageSemantic[] {
    if (pageNos && pageNos.length === 0) {
      return [];
    }
    const params: Array<string | number> = [documentId];
    let sql = `
      SELECT
        image_hash, source_document_id, source_page_no, source_image_index,
        mime_type, width, height, bbox_json, object_key, storage_uri,
        has_text, interference_score, is_dropped, recognizable, accessible_url,
        semantic_text, semantic_model
      FROM image_semantics
      WHERE source_document_id = ?
    `;
    if (pageNos && pageNos.length > 0) {
      const uniquePageNos = [...new Set(pageNos)].sort((a, b) => a - b);
      sql += ` AND source_page_no IN (${uniquePageNos.map(() => "?").join(", ")})`;
      params.push(...uniquePageNos);
    }
    sql += " ORDER BY source_page_no ASC, source_image_index ASC";
    const rows = this.db.prepare(sql).all(...params) as SqliteRow[];
    return rows.map(normalizeStoredImageSemantic);
  }

  deleteImageSemanticsForDocument(documentId: string): number {
    const result = this.db
      .prepare("DELETE FROM image_semantics WHERE source_document_id = ?")
      .run(documentId);
    return Number(result.changes ?? 0);
  }

  updateImageSemantic(imageHash: string, semanticText: string, semanticModel?: string | null): void {
    this.db
      .prepare(
        `
          UPDATE image_semantics
          SET semantic_text = ?, semantic_model = ?, last_enhanced_at = ?
          WHERE image_hash = ?
        `,
      )
      .run(semanticText, semanticModel ?? null, utcNowIso(), imageHash);
  }

  getImageSemanticCache(
    imageHash: string,
    promptVersion: string,
  ): StoredImageSemanticCache | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM image_semantic_cache
          WHERE image_hash = ? AND prompt_version = ?
          LIMIT 1
        `,
      )
      .get(imageHash, promptVersion) as SqliteRow | undefined;
    return row ? normalizeStoredImageSemanticCache(row) : null;
  }

  upsertImageSemanticCache(record: StorageImageSemanticCacheRecord): void {
    const now = utcNowIso();
    this.db
      .prepare(
        `
          INSERT INTO image_semantic_cache (
            image_hash, prompt_version, recognizable, image_kind, contains_text,
            visible_text, summary, entities_json, keywords_json, qa_hints_json,
            drop_reason, semantic_model, created_at, updated_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(image_hash, prompt_version) DO UPDATE SET
            recognizable = excluded.recognizable,
            image_kind = excluded.image_kind,
            contains_text = excluded.contains_text,
            visible_text = excluded.visible_text,
            summary = excluded.summary,
            entities_json = excluded.entities_json,
            keywords_json = excluded.keywords_json,
            qa_hints_json = excluded.qa_hints_json,
            drop_reason = excluded.drop_reason,
            semantic_model = excluded.semantic_model,
            updated_at = excluded.updated_at
        `,
      )
      .run(
        record.imageHash,
        record.promptVersion,
        record.recognizable ? 1 : 0,
        record.imageKind ?? null,
        record.containsText == null ? null : record.containsText ? 1 : 0,
        record.visibleText ?? null,
        record.summary ?? null,
        record.entitiesJson ?? "[]",
        record.keywordsJson ?? "[]",
        record.qaHintsJson ?? "[]",
        record.dropReason ?? null,
        record.semanticModel ?? null,
        now,
        now,
      );
  }

  createDocumentParseTask(task: StorageDocumentParseTaskRecord): StoredDocumentParseTask {
    const now = utcNowIso();
    this.db
      .prepare(
        `
          INSERT INTO document_parse_tasks (
            id, document_id, document_filename, task_type, status, progress_percent,
            current_stage, options_json, stage_timings_json, error_message,
            created_at, started_at, finished_at, updated_at, total_duration_ms
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        task.id,
        task.documentId ?? null,
        task.documentFilename,
        task.taskType,
        task.status,
        Number(task.progressPercent ?? 0),
        task.currentStage ?? null,
        normalizeJsonText(task.optionsJson ?? "{}", "{}"),
        normalizeJsonText(task.stageTimingsJson ?? "[]"),
        task.errorMessage ?? null,
        task.createdAt ?? now,
        task.startedAt ?? null,
        task.finishedAt ?? null,
        task.updatedAt ?? now,
        task.totalDurationMs ?? null,
      );
    const created = this.getDocumentParseTask(task.id);
    if (!created) {
      throw new Error("Failed to create document parse task.");
    }
    return created;
  }

  getDocumentParseTask(taskId: string): StoredDocumentParseTask | null {
    const row = this.db
      .prepare("SELECT * FROM document_parse_tasks WHERE id = ? LIMIT 1")
      .get(taskId) as SqliteRow | undefined;
    return row ? normalizeStoredDocumentParseTask(row) : null;
  }

  listDocumentParseTasks(input: {
    status?: DocumentParseTaskStatus | null;
    taskType?: DocumentParseTaskType | null;
    documentId?: string | null;
    limit?: number;
    offset?: number;
  } = {}): { items: StoredDocumentParseTask[]; total: number } {
    const conditions: string[] = [];
    const params: Array<string | number> = [];
    if (input.status) {
      conditions.push("status = ?");
      params.push(input.status);
    }
    if (input.taskType) {
      conditions.push("task_type = ?");
      params.push(input.taskType);
    }
    if (input.documentId) {
      conditions.push("document_id = ?");
      params.push(input.documentId);
    }
    const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
    const totalRow = this.db
      .prepare(`SELECT COUNT(*) AS count FROM document_parse_tasks ${where}`)
      .get(...params) as SqliteRow | undefined;
    const limit = Math.max(Number(input.limit ?? 20), 1);
    const offset = Math.max(Number(input.offset ?? 0), 0);
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM document_parse_tasks
          ${where}
          ORDER BY created_at DESC
          LIMIT ? OFFSET ?
        `,
      )
      .all(...params, limit, offset) as SqliteRow[];
    return {
      items: rows.map(normalizeStoredDocumentParseTask),
      total: Number(totalRow?.count ?? 0),
    };
  }

  updateDocumentParseTask(
    taskId: string,
    patch: {
      documentId?: string | null;
      status?: DocumentParseTaskStatus;
      progressPercent?: number;
      currentStage?: string | null;
      optionsJson?: string;
      stageTimingsJson?: string;
      errorMessage?: string | null;
      startedAt?: string | null;
      finishedAt?: string | null;
      totalDurationMs?: number | null;
    },
  ): StoredDocumentParseTask | null {
    const updates: string[] = [];
    const params: Array<string | number | null> = [];
    if ("documentId" in patch) {
      updates.push("document_id = ?");
      params.push(patch.documentId ?? null);
    }
    if (patch.status != null) {
      updates.push("status = ?");
      params.push(patch.status);
    }
    if (patch.progressPercent != null) {
      updates.push("progress_percent = ?");
      params.push(Math.max(0, Math.min(100, Math.round(patch.progressPercent))));
    }
    if ("currentStage" in patch) {
      updates.push("current_stage = ?");
      params.push(patch.currentStage ?? null);
    }
    if (patch.optionsJson != null) {
      updates.push("options_json = ?");
      params.push(normalizeJsonText(patch.optionsJson, "{}"));
    }
    if (patch.stageTimingsJson != null) {
      updates.push("stage_timings_json = ?");
      params.push(normalizeJsonText(patch.stageTimingsJson));
    }
    if ("errorMessage" in patch) {
      updates.push("error_message = ?");
      params.push(patch.errorMessage ?? null);
    }
    if ("startedAt" in patch) {
      updates.push("started_at = ?");
      params.push(patch.startedAt ?? null);
    }
    if ("finishedAt" in patch) {
      updates.push("finished_at = ?");
      params.push(patch.finishedAt ?? null);
    }
    if ("totalDurationMs" in patch) {
      updates.push("total_duration_ms = ?");
      params.push(patch.totalDurationMs ?? null);
    }
    if (updates.length === 0) {
      return this.getDocumentParseTask(taskId);
    }
    updates.push("updated_at = ?");
    params.push(utcNowIso(), taskId);
    const result = this.db
      .prepare(`UPDATE document_parse_tasks SET ${updates.join(", ")} WHERE id = ?`)
      .run(...params);
    if (result.changes === 0) {
      return null;
    }
    return this.getDocumentParseTask(taskId);
  }

  deleteDocumentParseTask(taskId: string): boolean {
    const result = this.db
      .prepare("DELETE FROM document_parse_tasks WHERE id = ?")
      .run(taskId);
    return Boolean(result.changes);
  }

  deleteActiveDocumentParseTasks(): number {
    const result = this.db
      .prepare("DELETE FROM document_parse_tasks WHERE status IN ('queued', 'running')")
      .run();
    return Number(result.changes ?? 0);
  }

  createCollection(name: string): PublicCollectionRecord {
    const normalizedName = normalizeCollectionName(name);
    if (!normalizedName) {
      throw new Error("Collection name is required.");
    }
    if (this.getActiveCollectionByName(normalizedName)) {
      throw new Error("Collection name already exists.");
    }
    const createdAt = utcNowIso();
    const collectionId = stableId("collection", `${normalizedName}:${createdAt}`);
    this.db
      .prepare(
        `
          INSERT INTO collections (id, name, kind, corpus_id, root_path, is_deleted, created_at, updated_at)
          VALUES (?, ?, 'user', NULL, NULL, 0, ?, ?)
        `,
      )
      .run(collectionId, normalizedName, createdAt, createdAt);
    const collection = this.getCollection(collectionId);
    if (!collection) {
      throw new Error("Failed to create collection.");
    }
    return collection;
  }

  listCollections(includeDeleted = false): PublicCollectionRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT id, name, kind, corpus_id, root_path, is_deleted, created_at, updated_at
          FROM collections
          WHERE kind = 'user'
            ${includeDeleted ? "" : "AND is_deleted = 0"}
          ORDER BY lower(name), created_at
        `,
      )
      .all() as SqliteRow[];
    return rows.map(normalizeStoredCollection).map(toPublicCollectionRecord);
  }

  getCollection(collectionId: string): PublicCollectionRecord | null {
    const row = this.db
      .prepare(
        `
          SELECT id, name, kind, corpus_id, root_path, is_deleted, created_at, updated_at
          FROM collections
          WHERE id = ? AND kind = 'user'
          LIMIT 1
        `,
      )
      .get(collectionId) as SqliteRow | undefined;
    return row ? toPublicCollectionRecord(normalizeStoredCollection(row)) : null;
  }

  getActiveCollectionByName(name: string): PublicCollectionRecord | null {
    const row = this.db
      .prepare(
        `
          SELECT id, name, kind, corpus_id, root_path, is_deleted, created_at, updated_at
          FROM collections
          WHERE kind = 'user' AND is_deleted = 0 AND name = ?
          LIMIT 1
        `,
      )
      .get(normalizeCollectionName(name)) as SqliteRow | undefined;
    return row ? toPublicCollectionRecord(normalizeStoredCollection(row)) : null;
  }

  updateCollection(collectionId: string, name: string): PublicCollectionRecord | null {
    const normalizedName = normalizeCollectionName(name);
    if (!normalizedName) {
      throw new Error("Collection name is required.");
    }
    const existing = this.getActiveCollectionByName(normalizedName);
    if (existing && existing.id !== collectionId) {
      throw new Error("Collection name already exists.");
    }
    const result = this.db
      .prepare(
        `
          UPDATE collections
          SET name = ?, updated_at = ?
          WHERE id = ? AND kind = 'user' AND is_deleted = 0
        `,
      )
      .run(normalizedName, utcNowIso(), collectionId);
    if (result.changes === 0) {
      return null;
    }
    return this.getCollection(collectionId);
  }

  setCollectionDeleted(collectionId: string, isDeleted: boolean): PublicCollectionRecord | null {
    const result = this.db
      .prepare(
        `
          UPDATE collections
          SET is_deleted = ?, updated_at = ?
          WHERE id = ? AND kind = 'user'
        `,
      )
      .run(isDeleted ? 1 : 0, utcNowIso(), collectionId);
    if (result.changes === 0) {
      return null;
    }
    return this.getCollection(collectionId);
  }

  countCollectionDocuments(collectionId: string): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM collection_documents cd
          JOIN documents d ON d.id = cd.document_id
          WHERE cd.collection_id = ? AND d.is_deleted = 0
        `,
      )
      .get(collectionId) as SqliteRow | undefined;
    return Number(row?.count ?? 0);
  }

  listCollectionDocuments(collectionId: string, includeDeleted = false): StoredDocument[] {
    const ids = (this.db
      .prepare(
        `
          SELECT d.id
          FROM collection_documents cd
          JOIN documents d ON d.id = cd.document_id
          WHERE cd.collection_id = ?
            ${includeDeleted ? "" : "AND d.is_deleted = 0"}
          ORDER BY d.relative_path
        `,
      )
      .all(collectionId) as SqliteRow[]).map((row) => String(row.id));
    return this.listDocumentsByIds(ids, includeDeleted);
  }

  listDocumentCollections(docId: string, includeDeleted = false): PublicCollectionRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT c.id, c.name, c.kind, c.corpus_id, c.root_path, c.is_deleted, c.created_at, c.updated_at
          FROM collection_documents cd
          JOIN collections c ON c.id = cd.collection_id
          WHERE cd.document_id = ?
            AND c.kind = 'user'
            ${includeDeleted ? "" : "AND c.is_deleted = 0"}
          ORDER BY lower(c.name), c.created_at
        `,
      )
      .all(docId) as SqliteRow[];
    return rows.map(normalizeStoredCollection).map(toPublicCollectionRecord);
  }

  attachDocumentsToCollection(collectionId: string, documentIds: string[]): number {
    if (documentIds.length === 0) {
      return 0;
    }
    const uniqueIds = [...new Set(documentIds.map((item) => String(item).trim()).filter(Boolean))].sort();
    const run = this.db.transaction(() => {
      const statement = this.db.prepare(
        `
          INSERT OR IGNORE INTO collection_documents (collection_id, document_id)
          VALUES (?, ?)
        `,
      );
      for (const docId of uniqueIds) {
        statement.run(collectionId, docId);
      }
    });
    run();
    return uniqueIds.length;
  }

  replaceCollectionDocuments(collectionId: string, documentIds: string[]): number {
    const uniqueIds = [...new Set(documentIds.map((item) => String(item).trim()).filter(Boolean))].sort();
    const run = this.db.transaction(() => {
      this.db.prepare("DELETE FROM collection_documents WHERE collection_id = ?").run(collectionId);
      if (uniqueIds.length === 0) {
        return;
      }
      const statement = this.db.prepare(
        `
          INSERT OR IGNORE INTO collection_documents (collection_id, document_id)
          VALUES (?, ?)
        `,
      );
      for (const docId of uniqueIds) {
        statement.run(collectionId, docId);
      }
    });
    run();
    return uniqueIds.length;
  }

  detachDocumentFromCollection(collectionId: string, docId: string): boolean {
    const result = this.db
      .prepare("DELETE FROM collection_documents WHERE collection_id = ? AND document_id = ?")
      .run(collectionId, docId);
    return Boolean(result.changes);
  }

  replaceDocumentCollections(docId: string, collectionIds: string[]): number {
    const uniqueIds = [...new Set(collectionIds.map((item) => String(item).trim()).filter(Boolean))].sort();
    const run = this.db.transaction(() => {
      this.db.prepare("DELETE FROM collection_documents WHERE document_id = ?").run(docId);
      if (uniqueIds.length === 0) {
        return;
      }
      const statement = this.db.prepare(
        `
          INSERT OR IGNORE INTO collection_documents (collection_id, document_id)
          VALUES (?, ?)
        `,
      );
      for (const collectionId of uniqueIds) {
        statement.run(collectionId, docId);
      }
    });
    run();
    return uniqueIds.length;
  }

  removeDocumentFromAllCollections(docId: string): number {
    const result = this.db
      .prepare("DELETE FROM collection_documents WHERE document_id = ?")
      .run(docId);
    return Number(result.changes ?? 0);
  }

  createDocumentCatalog(corpusId: string): DocumentCatalog {
    return {
      listDocuments: () => this.listDocuments(corpusId, false).map((document) => this.toDocumentSummary(document)),
      getDocument: (docId: string) => {
        const document = this.getDocument(docId);
        if (!document || document.is_deleted || document.corpus_id !== corpusId) {
          return null;
        }
        return this.toDocumentRecord(document);
      },
    };
  }

  private toDocumentSummary(document: StoredDocument): DocumentSummary {
    return {
      id: document.id,
      absolutePath: document.absolute_path,
      label: document.original_filename || document.relative_path,
      pageCount: document.page_count,
      pagesDir: document.pages_prefix || undefined,
    };
  }

  private toDocumentRecord(document: StoredDocument): DocumentRecord {
    return {
      ...this.toDocumentSummary(document),
      content: document.content,
    };
  }
}

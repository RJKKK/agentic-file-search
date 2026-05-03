import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import Database from "better-sqlite3";

import { resolveSqliteDbPath, SqliteStorage } from "../src/index.js";

describe("sqlite storage", () => {
  it("creates the traditional rag tables and no corpora table", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-schema-"));
    const dbPath = join(root, "storage.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();
    storage.close();

    const db = new Database(dbPath, { readonly: true });
    const rows = db
      .prepare(
        `
          SELECT name
          FROM sqlite_master
          WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
          ORDER BY name
        `,
      )
      .all() as Array<{ name: string }>;
    db.close();

    assert.deepEqual(rows.map((row) => row.name), [
      "collection_documents",
      "collections",
      "document_chunks",
      "document_chunks_fts",
      "document_chunks_fts_config",
      "document_chunks_fts_content",
      "document_chunks_fts_data",
      "document_chunks_fts_docsize",
      "document_chunks_fts_idx",
      "document_pages",
      "document_parse_tasks",
      "documents",
      "fixed_retrieval_chunks",
      "fixed_retrieval_chunks_fts",
      "fixed_retrieval_chunks_fts_config",
      "fixed_retrieval_chunks_fts_content",
      "fixed_retrieval_chunks_fts_data",
      "fixed_retrieval_chunks_fts_docsize",
      "fixed_retrieval_chunks_fts_idx",
      "image_semantic_cache",
      "image_semantics",
      "retrieval_chunks",
      "retrieval_chunks_fts",
      "retrieval_chunks_fts_config",
      "retrieval_chunks_fts_content",
      "retrieval_chunks_fts_data",
      "retrieval_chunks_fts_docsize",
      "retrieval_chunks_fts_idx",
    ]);
  });

  it("migrates older sqlite schemas during initialize", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-migrate-"));
    const dbPath = join(root, "storage.sqlite");
    const db = new Database(dbPath);
    db.exec(`
      CREATE TABLE collections (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        corpus_id TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        absolute_path TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        file_mtime REAL NOT NULL,
        file_size INTEGER NOT NULL,
        content_sha256 TEXT NOT NULL,
        last_indexed_at TEXT NOT NULL,
        is_deleted INTEGER NOT NULL DEFAULT 0
      );

      CREATE TABLE document_pages (
        document_id TEXT NOT NULL,
        page_no INTEGER NOT NULL,
        object_key TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        char_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (document_id, page_no)
      );

      INSERT INTO collections (id, name, is_deleted, created_at, updated_at)
      VALUES ('collection_legacy', 'Legacy', 0, '2024-01-01T00:00:00.000Z', '2024-01-01T00:00:00.000Z');

      INSERT INTO documents (
        id, corpus_id, relative_path, absolute_path, content, metadata_json,
        file_mtime, file_size, content_sha256, last_indexed_at, is_deleted
      )
      VALUES (
        'doc_legacy', 'corpus_legacy', 'legacy.pdf', 'legacy.pdf', 'Legacy body', '{}',
        1, 11, 'sha-legacy', '2024-01-01T00:00:00.000Z', 0
      );
    `);
    db.close();

    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    assert.equal(storage.getDocument("doc_legacy")?.original_filename, "legacy.pdf");
    assert.equal(storage.getDocument("doc_legacy")?.upload_status, "uploaded");
    assert.equal(storage.getDocument("doc_legacy")?.retrieval_chunking_strategy, "small_to_big");
    assert.equal(storage.getDocument("doc_legacy")?.fixed_chunk_chars, null);
    assert.deepEqual(storage.listCollections().map((item) => item.name), ["Legacy"]);

    const corpusId = storage.getOrCreateCorpus("blob://library/default");
    assert.equal(storage.getCorpusId("blob://library/default"), corpusId);
    const collection = storage.createCollection("Migrated");
    storage.attachDocumentsToCollection(collection.id, ["doc_legacy"]);
    assert.deepEqual(storage.listCollectionDocuments(collection.id).map((item) => item.id), [
      "doc_legacy",
    ]);

    storage.close();
  });

  it("preserves corpus semantics via hidden scope collections", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-corpus-"));
    const dbPath = join(root, "storage.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    const corpusId = storage.getOrCreateCorpus("blob://library/default");
    assert.equal(storage.getCorpusId("blob://library/default"), corpusId);
    assert.equal(storage.getCorpusRootPath(corpusId), "blob://library/default");
    assert.deepEqual(storage.listCollections(), []);

    const userCollection = storage.createCollection("Deals");
    assert.deepEqual(storage.listCollections().map((item) => item.name), ["Deals"]);
    assert.equal(storage.getCollection(userCollection.id)?.name, "Deals");

    storage.close();
  });

  it("enforces active collection names and replaces document mappings", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-collections-"));
    const dbPath = join(root, "storage.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    const corpusId = storage.getOrCreateCorpus("blob://library/default");
    const docA = SqliteStorage.makeDocumentId(corpusId, "alpha.md");
    const docB = SqliteStorage.makeDocumentId(corpusId, "beta.md");
    for (const [docId, relativePath] of [
      [docA, "alpha.md"],
      [docB, "beta.md"],
    ] as const) {
      storage.upsertDocumentStub({
        id: docId,
        corpusId,
        relativePath,
        absolutePath: relativePath,
        content: "",
        metadataJson: "{}",
        fileMtime: 1,
        fileSize: 10,
        contentSha256: `sha-${relativePath}`,
        originalFilename: relativePath,
      });
    }

    const deals = storage.createCollection("Deals");
    assert.throws(() => storage.createCollection("Deals"), /already exists/i);
    assert.equal(storage.countCollectionDocuments(deals.id), 0);

    const other = storage.createCollection("Other");
    assert.throws(() => storage.updateCollection(other.id, "Deals"), /already exists/i);

    storage.replaceCollectionDocuments(deals.id, [docA, docB]);
    assert.deepEqual(storage.listCollectionDocuments(deals.id).map((item) => item.id), [docA, docB]);
    assert.equal(storage.countCollectionDocuments(deals.id), 2);

    storage.replaceDocumentCollections(docA, [other.id]);
    assert.deepEqual(storage.listDocumentCollections(docA).map((item) => item.id), [other.id]);
    assert.deepEqual(storage.listCollectionDocuments(deals.id).map((item) => item.id), [docB]);

    storage.setCollectionDeleted(deals.id, true);
    const reused = storage.createCollection("Deals");
    assert.equal(reused.name, "Deals");

    storage.close();
  });

  it("enforces unique corpus-relative documents and exposes a corpus-bound catalog", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-docs-"));
    const dbPath = join(root, "storage.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    const corpusId = storage.getOrCreateCorpus("C:/docs");
    const otherCorpusId = storage.getOrCreateCorpus("C:/other");
    const docId = SqliteStorage.makeDocumentId(corpusId, "alpha.md");
    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "alpha.md",
      absolutePath: "C:/docs/alpha.md",
      content: "Alpha body",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 10,
      contentSha256: "sha-a",
      originalFilename: "alpha.md",
      pagesPrefix: "pages/alpha",
      pageCount: 2,
      uploadStatus: "indexed",
      parsedContentSha256: "sha-a",
      parsedIsComplete: true,
    });
    storage.upsertDocumentStub({
      id: SqliteStorage.makeDocumentId(otherCorpusId, "beta.md"),
      corpusId: otherCorpusId,
      relativePath: "beta.md",
      absolutePath: "C:/other/beta.md",
      content: "Beta body",
      metadataJson: "{}",
      fileMtime: 2,
      fileSize: 20,
      contentSha256: "sha-b",
      originalFilename: "beta.md",
    });

    assert.throws(() => {
      storage.upsertDocumentStub({
        id: "doc-conflict",
        corpusId,
        relativePath: "alpha.md",
        absolutePath: "C:/docs/alpha-v2.md",
        content: "Conflict body",
        metadataJson: "{}",
        fileMtime: 3,
        fileSize: 30,
        contentSha256: "sha-c",
        originalFilename: "alpha-v2.md",
      });
    });

    const catalog = storage.createDocumentCatalog(corpusId);
    assert.deepEqual(
      (await catalog.listDocuments()).map((item) => item.id),
      [docId],
    );
    assert.equal((await catalog.getDocument(docId))?.content, "Alpha body");
    assert.equal(await catalog.getDocument(SqliteStorage.makeDocumentId(otherCorpusId, "beta.md")), null);

    storage.close();
  });

  it("syncs pages and image semantics close to the legacy behavior", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-pages-"));
    const dbPath = join(root, "storage.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    const corpusId = storage.getOrCreateCorpus("C:/docs");
    const docId = SqliteStorage.makeDocumentId(corpusId, "alpha.md");
    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "alpha.md",
      absolutePath: "C:/docs/alpha.md",
      content: "Alpha body",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 10,
      contentSha256: "sha-a",
      originalFilename: "alpha.md",
    });

    const first = storage.syncDocumentPages(docId, [
      {
        documentId: docId,
        pageNo: 1,
        objectKey: "pages/alpha/page-0001.md",
        contentHash: "p1",
        charCount: 20,
        isSyntheticPage: false,
        heading: "Overview",
        sourceLocator: "page-1",
      },
      {
        documentId: docId,
        pageNo: 2,
        objectKey: "pages/alpha/page-0002.md",
        contentHash: "p2",
        charCount: 30,
        isSyntheticPage: false,
        heading: "Price",
        sourceLocator: "page-2",
      },
    ]);
    assert.deepEqual(first, { upserted: 2, untouched: 0, deleted: 0 });

    const second = storage.syncDocumentPages(docId, [
      {
        documentId: docId,
        pageNo: 1,
        objectKey: "pages/alpha/page-0001.md",
        contentHash: "p1",
        charCount: 20,
        isSyntheticPage: false,
        heading: "Overview",
        sourceLocator: "page-1",
      },
    ]);
    assert.deepEqual(second, { upserted: 0, untouched: 1, deleted: 1 });
    assert.equal(storage.getDocument(docId)?.page_count, 1);

    const written = storage.upsertImageSemantics([
      {
        imageHash: "img-1",
        sourceDocumentId: docId,
        sourcePageNo: 1,
        sourceImageIndex: 0,
        mimeType: "image/png",
        width: 100,
        height: 200,
      },
      {
        imageHash: "img-1",
        sourceDocumentId: docId,
        sourcePageNo: 1,
        sourceImageIndex: 0,
        semanticText: "ignored on conflict",
      },
    ]);
    assert.equal(written, 2);
    const image = storage.getImageSemantics(["img-1"])["img-1"];
    assert.equal(image?.mime_type, "image/png");
    assert.equal(image?.width, 100);
    assert.equal(image?.semantic_text, null);

    storage.updateImageSemantic("img-1", "A chart", "gemini");
    const updated = storage.listImageSemanticsForDocument(docId);
    assert.equal(updated[0]?.semantic_text, "A chart");
    assert.equal(updated[0]?.semantic_model, "gemini");

    storage.close();
  });

  it("stores and keyword-searches fixed retrieval chunks separately", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-fixed-rchunks-"));
    const dbPath = join(root, "storage.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    const corpusId = storage.getOrCreateCorpus("C:/docs");
    const docId = SqliteStorage.makeDocumentId(corpusId, "alpha.md");
    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "alpha.md",
      absolutePath: "C:/docs/alpha.md",
      content: "Alpha body",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 10,
      contentSha256: "sha-a",
      originalFilename: "alpha.md",
      retrievalChunkingStrategy: "fixed",
      fixedChunkChars: 800,
    });
    storage.replaceDocumentChunks(docId, [
      {
        id: "dchunk-1",
        documentId: docId,
        ordinal: 0,
        referenceRetrievalChunkId: "frchunk-1",
        pageNo: 1,
        documentIndex: 0,
        pageIndex: 0,
        blockType: "paragraph",
        bboxJson: JSON.stringify([0, 0, 10, 10]),
        contentMd: "Purchase price is 42 million dollars.",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: JSON.stringify([1]),
        mergedBboxesJson: JSON.stringify([[0, 0, 10, 10]]),
      },
      {
        id: "dchunk-3",
        documentId: docId,
        ordinal: 1,
        referenceRetrievalChunkId: "frchunk-2",
        pageNo: 2,
        documentIndex: 1,
        pageIndex: 0,
        blockType: "paragraph",
        bboxJson: JSON.stringify([0, 10, 10, 20]),
        contentMd: "Closing date is March 1.",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: JSON.stringify([2]),
        mergedBboxesJson: JSON.stringify([[0, 10, 10, 20]]),
      },
    ]);

    storage.replaceFixedRetrievalChunks(docId, [
      {
        id: "frchunk-1",
        documentId: docId,
        ordinal: 0,
        contentMd: "Purchase price is 42 million dollars.",
        sizeClass: "normal",
        summaryText: null,
        sourceDocumentChunkIdsJson: JSON.stringify(["dchunk-1"]),
        pageNosJson: JSON.stringify([1]),
        sourceLocator: "page-1",
        bboxesJson: JSON.stringify([[0, 0, 10, 10]]),
      },
      {
        id: "frchunk-2",
        documentId: docId,
        ordinal: 1,
        contentMd: "Closing date is March 1.",
        sizeClass: "normal",
        summaryText: null,
        sourceDocumentChunkIdsJson: JSON.stringify(["dchunk-3"]),
        pageNosJson: JSON.stringify([2]),
        sourceLocator: "page-2",
        bboxesJson: JSON.stringify([[0, 10, 10, 20]]),
      },
    ]);

    assert.equal(storage.listFixedRetrievalChunks(docId).length, 2);
    assert.match(storage.getFixedRetrievalChunk("frchunk-1")?.content_md ?? "", /Purchase price/i);

    const hits = storage.keywordSearchFixedRetrievalChunks({
      query: "\"purchase\" OR \"price\"",
      documentIds: [docId],
      limit: 5,
    });
    assert.equal(hits.length, 1);
    assert.equal(hits[0]?.retrieval_chunk_id, "frchunk-1");
    assert.equal(hits[0]?.document_id, docId);

    storage.close();
  });

  it("deletes documents and cascades pages, image semantics, and collection mappings", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-delete-"));
    const dbPath = join(root, "storage.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    const corpusId = storage.getOrCreateCorpus("C:/docs");
    const docId = SqliteStorage.makeDocumentId(corpusId, "alpha.md");
    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "alpha.md",
      absolutePath: "C:/docs/alpha.md",
      content: "Alpha body",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 10,
      contentSha256: "sha-a",
      originalFilename: "alpha.md",
    });
    storage.syncDocumentPages(docId, [
      {
        documentId: docId,
        pageNo: 1,
        objectKey: "pages/alpha/page-0001.md",
        contentHash: "p1",
        charCount: 20,
        isSyntheticPage: false,
      },
    ]);
    storage.upsertImageSemantics([
      {
        imageHash: "img-1",
        sourceDocumentId: docId,
        sourcePageNo: 1,
        sourceImageIndex: 0,
      },
    ]);
    const collection = storage.createCollection("Deals");
    storage.attachDocumentsToCollection(collection.id, [docId]);

    const deleted = storage.deleteDocument(docId);

    assert.equal(deleted?.id, docId);
    assert.equal(storage.getDocument(docId), null);
    assert.deepEqual(storage.listDocumentPages(docId), []);
    assert.deepEqual(storage.listImageSemanticsForDocument(docId), []);
    assert.deepEqual(storage.listCollectionDocuments(collection.id), []);

    storage.close();
  });

  it("resolves sqlite paths from explicit, new env, and local legacy env values", () => {
    const previousNew = process.env.FS_EXPLORER_SQLITE_PATH;
    const previousLegacy = process.env.FS_EXPLORER_DB_PATH;
    try {
      process.env.FS_EXPLORER_SQLITE_PATH = "db/new.sqlite";
      delete process.env.FS_EXPLORER_DB_PATH;
      assert.match(resolveSqliteDbPath(), /db[\\/]new\.sqlite$/);

      delete process.env.FS_EXPLORER_SQLITE_PATH;
      process.env.FS_EXPLORER_DB_PATH = "db/legacy.sqlite";
      assert.match(resolveSqliteDbPath(), /db[\\/]legacy\.sqlite$/);

      process.env.FS_EXPLORER_DB_PATH = "postgresql://user:pass@localhost/db";
      assert.doesNotMatch(resolveSqliteDbPath(), /postgresql:/);
      assert.match(resolveSqliteDbPath("db/override.sqlite"), /db[\\/]override\.sqlite$/);
    } finally {
      if (previousNew == null) {
        delete process.env.FS_EXPLORER_SQLITE_PATH;
      } else {
        process.env.FS_EXPLORER_SQLITE_PATH = previousNew;
      }
      if (previousLegacy == null) {
        delete process.env.FS_EXPLORER_DB_PATH;
      } else {
        process.env.FS_EXPLORER_DB_PATH = previousLegacy;
      }
    }
  });

  it("does not expose chunk or embedding APIs on the new storage surface", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-sqlite-surface-"));
    const dbPath = join(root, "storage.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    assert.equal("searchChunks" in storage, false);
    assert.equal("storeChunkEmbeddings" in storage, false);
    assert.equal("saveSchema" in storage, false);

    storage.close();
  });

  it("defaults the sqlite database into the data directory", () => {
    const resolved = resolveSqliteDbPath();
    assert.match(
      resolved.replaceAll("\\", "/"),
      /\/data\/agentic-file-search\.db$/,
    );
  });
});

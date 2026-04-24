import assert from "node:assert/strict";
import { mkdtemp, access } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import {
  DocumentLibraryService,
  LocalBlobStore,
  SqliteStorage,
  buildDocumentObjectKey,
  buildDocumentPagesKeyPrefix,
  loadDocumentPages,
} from "../src/index.js";
import type { ParsedDocument, PythonDocumentParserExecutor } from "../src/index.js";

class FakeParser implements PythonDocumentParserExecutor {
  calls = 0;
  filePaths: string[] = [];

  constructor(private readonly document: ParsedDocument) {}

  async parseDocument(filePath: string): Promise<ParsedDocument> {
    this.calls += 1;
    this.filePaths.push(filePath);
    return this.document;
  }
}

class FailingStorage extends SqliteStorage {
  override upsertDocumentStub(): void {
    throw new Error("boom");
  }
}

function buildParsedDocument(): ParsedDocument {
  return {
    parser_name: "fake",
    parser_version: "v1",
    units: [
      {
        unit_no: 1,
        markdown: "# Overview\nAlpha overview.",
        content_hash: "u1",
        heading: "Overview",
        source_locator: "page-1",
        images: [],
        blocks: [],
      },
      {
        unit_no: 2,
        markdown: "# Price\nPurchase price is 42.",
        content_hash: "u2",
        heading: "Price",
        source_locator: "page-2",
        images: [
          {
            image_hash: "img-1",
            page_no: 2,
            image_index: 0,
            mime_type: "image/png",
            width: 100,
            height: 50,
          },
        ],
        blocks: [],
      },
    ],
  };
}

async function waitForTask(service: DocumentLibraryService, taskId: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const task = service.getDocumentParseTask(taskId);
    if (task && ["completed", "failed"].includes(task.status)) {
      assert.notEqual(task.status, "failed");
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error(`Task ${taskId} did not finish in time`);
}

describe("document library service", () => {
  it("uploads documents, stores blobs/pages, and exposes a catalog with real pages_dir", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-library-service-upload-"));
    const dbPath = join(root, "library.sqlite");
    const objectRoot = join(root, "object-store");
    const storage = new SqliteStorage({ dbPath });
    const blobStore = new LocalBlobStore(objectRoot);
    const parser = new FakeParser(buildParsedDocument());
    storage.initialize();
    const service = new DocumentLibraryService(storage, blobStore, parser);

    const result = await service.uploadDocument({
      filename: "alpha.docx",
      data: Buffer.from("fake-docx", "utf8"),
      contentType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    });
    await waitForTask(service, result.task.id);

    assert.equal(result.document.original_filename, "alpha.docx");
    assert.equal(result.task.task_type, "upload_parse");

    const document = storage.getDocument(String(result.document.id));
    assert.ok(document);
    assert.equal(document?.page_count, 2);
    assert.equal(document?.upload_status, "completed");
    assert.equal(storage.listDocumentPages(String(result.document.id)).length, 2);

    const sourcePath = await blobStore.materialize({
      objectKey: buildDocumentObjectKey(String(result.document.id), "alpha.docx"),
    });
    await access(sourcePath, fsConstants.F_OK);

    const catalog = service.createDocumentCatalog();
    const listed = await catalog.listDocuments();
    assert.equal(listed[0]?.id, result.document.id);
    assert.match(String(listed[0]?.pagesDir), /documents[\\/]alpha\.docx[\\/]pages$/);
    const fullDocument = await catalog.getDocument(String(result.document.id));
    assert.match(String(fullDocument?.content), /Purchase price is 42/);

    const loadedPages = await loadDocumentPages({
      storage,
      blobStore,
      documentId: String(result.document.id),
    });
    assert.equal(loadedPages[0]?.markdown, "# Overview\nAlpha overview.\n");
    assert.equal(loadedPages[0]?.is_synthetic_page, true);

    storage.close();
  });

  it("rejects duplicate filenames using legacy original_filename semantics", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-library-service-duplicate-"));
    const dbPath = join(root, "library.sqlite");
    const objectRoot = join(root, "object-store");
    const storage = new SqliteStorage({ dbPath });
    const blobStore = new LocalBlobStore(objectRoot);
    const parser = new FakeParser(buildParsedDocument());
    storage.initialize();
    const service = new DocumentLibraryService(storage, blobStore, parser);

    const uploaded = await service.uploadDocument({
      filename: "Alpha.pdf",
      data: Buffer.from("pdf", "utf8"),
      contentType: "application/pdf",
    });
    await waitForTask(service, uploaded.task.id);

    await assert.rejects(
      service.uploadDocument({
        filename: "alpha.pdf",
        data: Buffer.from("pdf-two", "utf8"),
        contentType: "application/pdf",
      }),
      /same filename/i,
    );

    storage.close();
  });

  it("parses uploaded files through an ASCII temp path while preserving original filenames", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-library-service-unicode-"));
    const dbPath = join(root, "library.sqlite");
    const objectRoot = join(root, "object-store");
    const storage = new SqliteStorage({ dbPath });
    const blobStore = new LocalBlobStore(objectRoot);
    const parser = new FakeParser(buildParsedDocument());
    storage.initialize();
    const service = new DocumentLibraryService(storage, blobStore, parser);

    const filename = "宁德时代：2024年年度报告.pdf";
    const uploaded = await service.uploadDocument({
      filename,
      data: Buffer.from("%PDF fake", "utf8"),
      contentType: "application/pdf",
    });
    await waitForTask(service, uploaded.task.id);

    assert.equal(uploaded.document.original_filename, filename);
    assert.equal(parser.calls, 1);
    assert.match(parser.filePaths[0] ?? "", /fs-explorer-parse-/);
    assert.match(parser.filePaths[0] ?? "", /source-[a-f0-9]{32}\.pdf$/);
    assert.doesNotMatch(parser.filePaths[0] ?? "", /宁德时代/);

    const sourcePath = await blobStore.materialize({
      objectKey: buildDocumentObjectKey(String(uploaded.document.id), filename),
    });
    await access(sourcePath, fsConstants.F_OK);

    storage.close();
  });

  it("rolls back source and pages blobs when upload fails after page persistence", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-library-service-rollback-"));
    const dbPath = join(root, "library.sqlite");
    const objectRoot = join(root, "object-store");
    const storage = new FailingStorage({ dbPath });
    const blobStore = new LocalBlobStore(objectRoot);
    const parser = new FakeParser(buildParsedDocument());
    storage.initialize();
    const service = new DocumentLibraryService(storage, blobStore, parser);

    await assert.rejects(
      service.uploadDocument({
        filename: "alpha.docx",
        data: Buffer.from("docx", "utf8"),
        contentType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      }),
      /boom/,
    );

    assert.deepEqual(await blobStore.listPrefix({ prefix: buildDocumentPagesKeyPrefix("alpha.docx") }), []);
    assert.equal(
      await blobStore.head({ objectKey: buildDocumentObjectKey("ignored", "alpha.docx") }),
      null,
    );

    storage.close();
  });

  it("uses parse-state cache on reparse and rewrites pages when forced", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-library-service-reparse-"));
    const dbPath = join(root, "library.sqlite");
    const objectRoot = join(root, "object-store");
    const storage = new SqliteStorage({ dbPath });
    const blobStore = new LocalBlobStore(objectRoot);
    const parser = new FakeParser(buildParsedDocument());
    storage.initialize();
    const service = new DocumentLibraryService(storage, blobStore, parser);

    const uploaded = await service.uploadDocument({
      filename: "alpha.docx",
      data: Buffer.from("docx", "utf8"),
      contentType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    });
    await waitForTask(service, uploaded.task.id);

    assert.equal(parser.calls, 1);
    const reparsed = await service.reparseDocument({
      docId: String(uploaded.document.id),
      force: true,
    });
    await waitForTask(service, reparsed.task.id);
    assert.equal(parser.calls, 2);

    storage.close();
  });

  it("deletes source/pages blobs and database rows together", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-library-service-delete-"));
    const dbPath = join(root, "library.sqlite");
    const objectRoot = join(root, "object-store");
    const storage = new SqliteStorage({ dbPath });
    const blobStore = new LocalBlobStore(objectRoot);
    const parser = new FakeParser(buildParsedDocument());
    storage.initialize();
    const service = new DocumentLibraryService(storage, blobStore, parser);

    const uploaded = await service.uploadDocument({
      filename: "alpha.pdf",
      data: Buffer.from("pdf", "utf8"),
      contentType: "application/pdf",
    });
    await waitForTask(service, uploaded.task.id);

    const deleted = await service.deleteDocument({ docId: String(uploaded.document.id) });

    assert.equal(deleted.deleted, true);
    assert.equal(storage.getDocument(String(uploaded.document.id)), null);
    assert.deepEqual(await blobStore.listPrefix({ prefix: buildDocumentPagesKeyPrefix("alpha.pdf") }), []);
    assert.equal(
      await blobStore.head({ objectKey: buildDocumentObjectKey(String(uploaded.document.id), "alpha.pdf") }),
      null,
    );

    storage.close();
  });
});

import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import {
  LocalBlobStore,
  SqliteStorage,
  buildDocumentObjectKey,
  buildDocumentPagesKeyPrefix,
  ensureLibraryCorpus,
  getLibraryCorpusId,
  materializeDocument,
  resolveDocumentScope,
} from "../src/index.js";

describe("document library helpers", () => {
  it("preserves the library corpus identity and scope resolution", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-library-scope-"));
    const dbPath = join(root, "library.sqlite");
    const storage = new SqliteStorage({ dbPath });
    storage.initialize();

    const corpusId = ensureLibraryCorpus(storage);
    assert.equal(getLibraryCorpusId(storage), corpusId);
    assert.equal(buildDocumentObjectKey("doc-1", "alpha.pdf"), "documents/alpha.pdf/source/alpha.pdf");
    assert.equal(buildDocumentPagesKeyPrefix("alpha.pdf"), "documents/alpha.pdf/pages");

    const docId = SqliteStorage.makeDocumentId(corpusId, "alpha.pdf");
    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "alpha.pdf",
      absolutePath: "placeholder",
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 10,
      contentSha256: "sha-a",
      originalFilename: "alpha.pdf",
    });
    const collection = storage.createCollection("Deals");
    storage.attachDocumentsToCollection(collection.id, [docId]);

    const scope = resolveDocumentScope({
      storage,
      documentIds: [docId],
      collectionId: collection.id,
    });

    assert.equal(scope.corpusId, corpusId);
    assert.deepEqual(scope.documentIds, [docId]);
    assert.equal(scope.documents[0]?.id, docId);
    assert.equal(scope.collection?.name, "Deals");
    assert.equal(scope.isEmpty, false);

    storage.close();
  });

  it("materializes source blobs back into the stored absolute path", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-library-materialize-"));
    const dbPath = join(root, "library.sqlite");
    const objectRoot = join(root, "object-store");
    const storage = new SqliteStorage({ dbPath });
    const blobStore = new LocalBlobStore(objectRoot);
    storage.initialize();

    const corpusId = ensureLibraryCorpus(storage);
    const docId = SqliteStorage.makeDocumentId(corpusId, "alpha.pdf");
    const sourceObjectKey = buildDocumentObjectKey(docId, "alpha.pdf");
    await blobStore.put({
      objectKey: sourceObjectKey,
      data: Buffer.from("source", "utf8"),
    });
    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "alpha.pdf",
      absolutePath: "stale-path",
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 6,
      contentSha256: "sha-a",
      originalFilename: "alpha.pdf",
      objectKey: sourceObjectKey,
      sourceObjectKey,
    });

    const materialized = await materializeDocument({
      storage,
      blobStore,
      document: storage.getDocument(docId)!,
    });

    assert.match(materialized.absolute_path, /documents[\\/]alpha\.pdf[\\/]source[\\/]alpha\.pdf$/);
    assert.equal(storage.getDocument(docId)?.absolute_path, materialized.absolute_path);

    storage.close();
  });
});


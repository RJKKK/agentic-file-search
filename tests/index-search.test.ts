import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import {
  ensureLibraryCorpus,
  IndexSearchService,
  LocalBlobStore,
  pageMatchesQuery,
  searchTerms,
  SqliteStorage,
} from "../src/index.js";

async function createIndexedFixture() {
  const root = await mkdtemp(join(tmpdir(), "afs-index-search-"));
  const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
  storage.initialize();
  const blobStore = new LocalBlobStore(join(root, "object-store"));
  const corpusId = ensureLibraryCorpus(storage);

  for (const item of [
    {
      id: "doc-alpha",
      filename: "alpha.pdf",
      page: 1,
      heading: "Purchase Price",
      markdown: "The purchase price is $45,000,000. Closing follows later.",
    },
    {
      id: "doc-beta",
      filename: "beta.pdf",
      page: 1,
      heading: "Overview",
      markdown: "This document mentions price once, but not the full purchase price phrase.",
    },
  ]) {
    const prefix = `library/documents/${item.filename}/pages`;
    await mkdir(join(blobStore.rootDir, prefix), { recursive: true });
    const objectKey = `${prefix}/page-0001.md`;
    await writeFile(
      join(blobStore.rootDir, objectKey),
      [
        "---",
        `document_id: ${item.id}`,
        `page_no: ${item.page}`,
        `original_filename: ${item.filename}`,
        `heading: ${item.heading}`,
        "source_locator: page-1",
        "---",
        "",
        item.markdown,
      ].join("\n"),
    );
    storage.upsertDocumentStub({
      id: item.id,
      corpusId,
      relativePath: item.filename,
      absolutePath: join(root, item.filename),
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 100,
      contentSha256: `sha-${item.id}`,
      originalFilename: item.filename,
      pagesPrefix: prefix,
      pageCount: 1,
      uploadStatus: "uploaded",
      parsedContentSha256: `sha-${item.id}`,
      parsedIsComplete: true,
    });
    storage.syncDocumentPages(item.id, [
      {
        documentId: item.id,
        pageNo: 1,
        objectKey,
        contentHash: `hash-${item.id}`,
        charCount: item.markdown.length,
        isSyntheticPage: false,
        heading: item.heading,
        sourceLocator: "page-1",
      },
    ]);
  }

  return { root, storage, blobStore };
}

describe("index search service", () => {
  it("matches the legacy server query term and page scoring behavior", () => {
    assert.deepEqual(searchTerms("purchase price"), ["purchase price", "purchase", "price"]);
    const match = pageMatchesQuery({
      query: "purchase price",
      markdown: "The purchase price is $45,000,000.",
      heading: "Purchase Price",
    });
    assert.equal(match.matchCount, 6);
    assert.equal(match.score, 42);
    assert.equal(match.matchStart, 4);
  });

  it("searches scoped page manifests and returns ranked page hits", async () => {
    const fixture = await createIndexedFixture();
    const service = new IndexSearchService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      documentIds: ["doc-alpha", "doc-beta"],
      scopeLabel: "2 selected documents",
    });

    const result = await service.search({ query: "purchase price", limit: 5 });
    assert.equal(result.lazy_indexing.triggered, false);
    assert.deepEqual(result.document_ids, ["doc-alpha", "doc-beta"]);
    assert.equal(result.hits[0].doc_id, "doc-alpha");
    assert.equal(result.hits[0].source_unit_no, 1);
    assert.match(result.hits[0].text, /purchase price/);
    assert.match(service.renderSearchResult(result), /=== INDEXED SEARCH RESULTS ===/);

    fixture.storage.close();
  });

  it("searches the union of multiple collections and selected documents", async () => {
    const fixture = await createIndexedFixture();
    const collectionA = fixture.storage.createCollection("Deals A");
    const collectionB = fixture.storage.createCollection("Deals B");
    fixture.storage.attachDocumentsToCollection(collectionA.id, ["doc-alpha"]);
    fixture.storage.attachDocumentsToCollection(collectionB.id, ["doc-beta"]);

    const service = new IndexSearchService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      documentIds: ["doc-alpha"],
      collectionIds: [collectionA.id, collectionB.id],
    });

    const result = await service.search({ query: "purchase price", limit: 5 });
    assert.deepEqual(result.collection_ids, [collectionA.id, collectionB.id]);
    assert.deepEqual(result.document_ids, ["doc-alpha", "doc-beta"]);
    assert.equal(result.collection_id, collectionA.id);
    assert.equal(result.hits.length, 2);

    fixture.storage.close();
  });

  it("provides legacy list/get document and page-scope helpers", async () => {
    const fixture = await createIndexedFixture();
    const events: Array<{ type: string; data: Record<string, unknown> }> = [];
    const service = new IndexSearchService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      documentIds: ["doc-alpha"],
      scopeLabel: "1 selected document",
      emitRuntimeEvent(type, data) {
        events.push({ type, data });
      },
    });

    const listOutput = service.listIndexedDocuments();
    assert.match(listOutput, /Scope: 1 selected document/);
    assert.match(listOutput, /Use glob\/grep\/read on the pages_dir/);

    const document = await service.getDocument("doc-alpha");
    assert.match(document.rendered, /=== DOCUMENT doc-alpha ===/);
    assert.match(document.structured?.content ?? "", /purchase price/);

    const pageScope = service.resolveDocumentPageScope(document.structured!.absolute_path);
    assert.equal(pageScope?.document.id, "doc-alpha");
    const pageHits = await service.searchPagesForTarget(pageScope!.pagesDir, "purchase price");
    assert.deepEqual(pageHits?.hits.map((hit) => hit.source_unit_no), [1]);
    service.emit("candidate_pages_found", { document_id: "doc-alpha" });
    assert.equal(events[0].type, "candidate_pages_found");

    fixture.storage.close();
  });

  it("renders only short image semantics in document context and truncates long entries", async () => {
    const fixture = await createIndexedFixture();
    fixture.storage.upsertImageSemantics([
      {
        imageHash: "img-alpha-1",
        sourceDocumentId: "doc-alpha",
        sourcePageNo: 1,
        sourceImageIndex: 1,
        semanticText: "图片链接：![image](/api/assets/images/demo)\n\nShort retrieval summary.\n" + "A".repeat(500),
        semanticDetailText: "Detailed semantic markdown that should stay out of document context.",
        semanticModel: "gpt-4o-mini",
      },
    ]);
    const service = new IndexSearchService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      documentIds: ["doc-alpha"],
    });

    const document = await service.getDocument("doc-alpha");
    assert.match(document.rendered, /Image semantics cache:/);
    assert.match(document.rendered, /!\[image\]\(\/api\/assets\/images\/demo\)/);
    assert.match(document.rendered, /Short retrieval summary/);
    assert.match(document.rendered, /\[truncated\]/);
    assert.doesNotMatch(document.rendered, /!\[image\]\(\/api\/assets\/images\/dem(?:\)|$)/);
    assert.doesNotMatch(document.rendered, /Detailed semantic markdown/);

    fixture.storage.close();
  });

});

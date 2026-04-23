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

  it("lists scope page directories, searches candidates across scope, and resolves batch reads", async () => {
    const fixture = await createIndexedFixture();
    const service = new IndexSearchService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      documentIds: ["doc-alpha", "doc-beta"],
    });

    const scopes = await service.listPageScopes();
    assert.deepEqual(scopes.map((item) => item.doc_id), ["doc-alpha", "doc-beta"]);
    assert.deepEqual(scopes[0].page_range, { start: 1, end: 1 });

    const candidates = await service.searchPagesAcrossScope("purchase price", {
      maxTotalHits: 24,
      maxHitsPerDocument: 5,
    });
    assert.deepEqual(candidates.hits.map((hit) => hit.doc_id), ["doc-alpha", "doc-beta"]);
    assert.ok(candidates.hits.every((hit) => hit.source_unit_no === 1));

    const groups = await service.resolvePageBatch({
      filePaths: candidates.hits.map((hit) => hit.absolute_path),
      maxPages: 5,
      maxChars: 10_000,
    });
    assert.deepEqual(groups.map((group) => group.document.id), ["doc-alpha", "doc-beta"]);
    assert.deepEqual(groups.flatMap((group) => group.pages.map((page) => page.page_no)), [1, 1]);

    const rangeGroups = await service.resolvePageBatch({
      documentId: "doc-alpha",
      startPage: 1,
      endPage: 1,
    });
    assert.equal(rangeGroups[0].document.id, "doc-alpha");
    assert.equal(rangeGroups[0].pages[0].page_no, 1);

    fixture.storage.close();
  });

  it("loads same-document boundary context with strict direction handling", async () => {
    const fixture = await createIndexedFixture();
    const document = fixture.storage.getDocument("doc-alpha");
    assert.ok(document);

    const pageTwoKey = `${document!.pages_prefix}/page-0002.md`;
    const pageThreeKey = `${document!.pages_prefix}/page-0003.md`;
    const pageFourKey = `${document!.pages_prefix}/page-0004.md`;
    await writeFile(
      join(fixture.blobStore.rootDir, pageTwoKey),
      [
        "---",
        "document_id: doc-alpha",
        "page_no: 2",
        "original_filename: alpha.pdf",
        "heading: Closing",
        "source_locator: page-2",
        "---",
        "",
        "Closing follows later.",
      ].join("\n"),
    );
    await writeFile(
      join(fixture.blobStore.rootDir, pageThreeKey),
      [
        "---",
        "document_id: doc-alpha",
        "page_no: 3",
        "original_filename: alpha.pdf",
        "heading: Adjustments",
        "source_locator: page-3",
        "---",
        "",
        "Working capital adjustments continue.",
      ].join("\n"),
    );
    await writeFile(
      join(fixture.blobStore.rootDir, pageFourKey),
      [
        "---",
        "document_id: doc-alpha",
        "page_no: 4",
        "original_filename: alpha.pdf",
        "heading: Schedules",
        "source_locator: page-4",
        "---",
        "",
        "Schedule content begins here.",
      ].join("\n"),
    );
    fixture.storage.syncDocumentPages("doc-alpha", [
      {
        documentId: "doc-alpha",
        pageNo: 1,
        objectKey: `${document!.pages_prefix}/page-0001.md`,
        contentHash: "hash-doc-alpha",
        charCount: 58,
        isSyntheticPage: false,
        heading: "Purchase Price",
        sourceLocator: "page-1",
        leadingBlockMarkdown: "The purchase price is $45,000,000.",
        trailingBlockMarkdown: "Closing follows",
      },
      {
        documentId: "doc-alpha",
        pageNo: 2,
        objectKey: pageTwoKey,
        contentHash: "hash-doc-alpha-2",
        charCount: 22,
        isSyntheticPage: false,
        heading: "Closing",
        sourceLocator: "page-2",
        leadingBlockMarkdown: "Closing follows later.",
        trailingBlockMarkdown: "Closing follows later.",
      },
      {
        documentId: "doc-alpha",
        pageNo: 3,
        objectKey: pageThreeKey,
        contentHash: "hash-doc-alpha-3",
        charCount: 37,
        isSyntheticPage: false,
        heading: "Adjustments",
        sourceLocator: "page-3",
        leadingBlockMarkdown: "Working capital adjustments continue.",
        trailingBlockMarkdown: "Working capital adjustments continue.",
      },
      {
        documentId: "doc-alpha",
        pageNo: 4,
        objectKey: pageFourKey,
        contentHash: "hash-doc-alpha-4",
        charCount: 29,
        isSyntheticPage: false,
        heading: "Schedules",
        sourceLocator: "page-4",
        leadingBlockMarkdown: "Schedule content begins here.",
        trailingBlockMarkdown: "Schedule content begins here.",
      },
    ]);

    const service = new IndexSearchService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      documentIds: ["doc-alpha"],
    });

    const previousOnly = await service.getPageBoundaryContext({
      documentId: "doc-alpha",
      pageNo: 2,
      direction: "previous",
    });
    assert.equal(previousOnly?.mode, "single");
    assert.equal(previousOnly?.anchor_page?.page_no, 2);
    assert.match(previousOnly?.rendered ?? "", /CURRENT PAGE CONTENT \(2\)/);
    assert.match(previousOnly?.rendered ?? "", /Closing follows later\./);
    assert.match(previousOnly?.rendered ?? "", /PREVIOUS PAGE TAIL \(1\)/);
    assert.match(previousOnly?.rendered ?? "", /CURRENT PAGE HEAD \(2\)/);
    assert.doesNotMatch(previousOnly?.rendered ?? "", /CURRENT PAGE TAIL \(2\)/);
    assert.doesNotMatch(previousOnly?.rendered ?? "", /NEXT PAGE HEAD \(3\)/);

    const nextOnly = await service.getPageBoundaryContext({
      documentId: "doc-alpha",
      pageNo: 2,
      direction: "next",
    });
    assert.equal(nextOnly?.mode, "single");
    assert.equal(nextOnly?.anchor_page?.page_no, 2);
    assert.match(nextOnly?.rendered ?? "", /CURRENT PAGE CONTENT \(2\)/);
    assert.match(nextOnly?.rendered ?? "", /Closing follows later\./);
    assert.match(nextOnly?.rendered ?? "", /CURRENT PAGE TAIL \(2\)/);
    assert.match(nextOnly?.rendered ?? "", /NEXT PAGE HEAD \(3\)/);
    assert.doesNotMatch(nextOnly?.rendered ?? "", /CURRENT PAGE HEAD \(2\)/);
    assert.doesNotMatch(nextOnly?.rendered ?? "", /PREVIOUS PAGE TAIL \(1\)/);

    const bothSides = await service.getPageBoundaryContext({
      documentId: "doc-alpha",
      pageNo: 1,
      direction: "both",
    });
    assert.equal(bothSides?.mode, "single");
    assert.equal(bothSides?.anchor_page?.page_no, 1);
    assert.match(bothSides?.rendered ?? "", /CURRENT PAGE CONTENT \(1\)/);
    assert.match(bothSides?.rendered ?? "", /The purchase price is \$45,000,000\./);
    assert.match(bothSides?.rendered ?? "", /CURRENT PAGE HEAD \(1\)/);
    assert.match(bothSides?.rendered ?? "", /NEXT PAGE HEAD \(2\)/);
    assert.match(bothSides?.rendered ?? "", /Previous page is unavailable/);

    const rangePrevious = await service.getPageBoundaryContext({
      documentId: "doc-alpha",
      startPage: 2,
      endPage: 3,
      direction: "previous",
    });
    assert.equal(rangePrevious?.mode, "range");
    assert.equal(rangePrevious?.start_page?.page_no, 2);
    assert.equal(rangePrevious?.end_page?.page_no, 3);
    assert.match(rangePrevious?.rendered ?? "", /pages: 2-3/);
    assert.match(rangePrevious?.rendered ?? "", /CURRENT RANGE CONTENT/);
    assert.match(rangePrevious?.rendered ?? "", /--- PAGE 2 ---/);
    assert.match(rangePrevious?.rendered ?? "", /--- PAGE 3 ---/);
    assert.match(rangePrevious?.rendered ?? "", /Closing follows later\./);
    assert.match(rangePrevious?.rendered ?? "", /Working capital adjustments continue\./);
    assert.match(rangePrevious?.rendered ?? "", /START PAGE HEAD \(2\)/);
    assert.doesNotMatch(rangePrevious?.rendered ?? "", /END PAGE TAIL \(3\)/);
    assert.doesNotMatch(rangePrevious?.rendered ?? "", /NEXT PAGE HEAD \(4\)/);

    const rangeNext = await service.getPageBoundaryContext({
      documentId: "doc-alpha",
      startPage: 2,
      endPage: 3,
      direction: "next",
    });
    assert.equal(rangeNext?.mode, "range");
    assert.match(rangeNext?.rendered ?? "", /CURRENT RANGE CONTENT/);
    assert.match(rangeNext?.rendered ?? "", /Closing follows later\./);
    assert.match(rangeNext?.rendered ?? "", /Working capital adjustments continue\./);
    assert.match(rangeNext?.rendered ?? "", /END PAGE TAIL \(3\)/);
    assert.match(rangeNext?.rendered ?? "", /NEXT PAGE HEAD \(4\)/);
    assert.doesNotMatch(rangeNext?.rendered ?? "", /START PAGE HEAD \(2\)/);
    assert.doesNotMatch(rangeNext?.rendered ?? "", /PREVIOUS PAGE TAIL \(1\)/);

    const rangeBoth = await service.getPageBoundaryContext({
      documentId: "doc-alpha",
      startPage: 2,
      endPage: 3,
      direction: "both",
    });
    assert.equal(rangeBoth?.mode, "range");
    assert.match(rangeBoth?.rendered ?? "", /CURRENT RANGE CONTENT/);
    assert.match(rangeBoth?.rendered ?? "", /Closing follows later\./);
    assert.match(rangeBoth?.rendered ?? "", /Working capital adjustments continue\./);
    assert.match(rangeBoth?.rendered ?? "", /PREVIOUS PAGE TAIL \(1\)/);
    assert.match(rangeBoth?.rendered ?? "", /START PAGE HEAD \(2\)/);
    assert.match(rangeBoth?.rendered ?? "", /END PAGE TAIL \(3\)/);
    assert.match(rangeBoth?.rendered ?? "", /NEXT PAGE HEAD \(4\)/);

    fixture.storage.close();
  });
});

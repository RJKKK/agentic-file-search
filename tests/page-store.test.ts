import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import {
  LocalBlobStore,
  buildDocumentPageKey,
  buildDocumentPagesPrefix,
  buildDocumentSourceKey,
  extractBoundaryBlocks,
  parsePageFrontMatter,
  persistDocumentPages,
  renderPageMarkdown,
  sanitizeStoredPageMarkdown,
  validateStorageFilename,
} from "../src/index.js";

describe("page store", () => {
  it("keeps original filenames but builds storage keys from a safe document id", () => {
    assert.equal(validateStorageFilename("alpha.pdf"), "alpha.pdf");
    assert.equal(validateStorageFilename("bad:name.pdf"), "bad:name.pdf");
    assert.equal(buildDocumentSourceKey("doc-1", "alpha.pdf"), "documents/doc-1/source/alpha.pdf");
    assert.equal(
      buildDocumentSourceKey("doc-1", "宁德时代：2024年年度报告.pdf"),
      "documents/doc-1/source/%E5%AE%81%E5%BE%B7%E6%97%B6%E4%BB%A3%EF%BC%9A2024%E5%B9%B4%E5%B9%B4%E5%BA%A6%E6%8A%A5%E5%91%8A.pdf",
    );
    assert.equal(buildDocumentPagesPrefix("doc-1"), "documents/doc-1/pages");
    assert.equal(buildDocumentPageKey("doc-1", 1), "documents/doc-1/pages/page-0001.md");
  });

  it("renders pure markdown bodies and persists page blobs with page-0001 naming", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-page-store-"));
    const blobStore = new LocalBlobStore(root);

    assert.equal(
      renderPageMarkdown({
        documentId: "doc-1",
        originalFilename: "alpha.pdf",
        pageNo: 1,
        pageLabel: "1",
        body: "  Hello world  ",
      }),
      "Hello world\n",
    );

    const pages = await persistDocumentPages({
      blobStore,
      storageKey: "doc-1",
      documentId: "doc-1",
      originalFilename: "alpha.pdf",
      contentType: "application/pdf",
      syntheticPages: false,
      parsedDocument: {
        parser_name: "fake",
        parser_version: "v1",
        units: [
          {
            unit_no: 2,
            markdown: "Second",
            content_hash: "u2",
            heading: "Second",
            source_locator: "page-2",
            images: [],
          },
          {
            unit_no: 1,
            markdown: "First",
            content_hash: "u1",
            heading: "First",
            source_locator: "page-1",
            images: [],
          },
        ],
      },
    });

    assert.equal(pages[0]?.objectKey, "documents/doc-1/pages/page-0001.md");
    assert.equal(
      Buffer.from(await blobStore.get({ objectKey: "documents/doc-1/pages/page-0001.md" })).toString("utf8"),
      "First\n",
    );
    const [header, body] = parsePageFrontMatter("First\n");
    assert.deepEqual(header, {});
    assert.equal(body, "First\n");
  });

  it("removes Col placeholders from stored boundary blocks without rewriting the row shape", () => {
    const boundaries = extractBoundaryBlocks(
      [
        "|筹资活动产生的现金流量：|Col2|Col3|Col4|",
        "|---|---|---|---|",
        "|A|B|C|D|",
        "",
        "|尾部|Col2|Col12|",
      ].join("\n"),
    );

    assert.equal(
      boundaries.leadingBlockMarkdown,
      "|筹资活动产生的现金流量：||||\n|---|---|---|---|\n|A|B|C|D|",
    );
    assert.equal(boundaries.trailingBlockMarkdown, "|尾部|||");
  });

  it("also removes bracketed Col[x] placeholder cells without rewriting the row shape", () => {
    assert.equal(
      sanitizeStoredPageMarkdown("|筹资活动产生的现金流量：|Col2|Col3|Col4|Col[5]|"),
      "|筹资活动产生的现金流量：|||||",
    );
  });

  it("removes Col placeholders from persisted page file bodies", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-page-store-clean-"));
    const blobStore = new LocalBlobStore(root);

    assert.equal(sanitizeStoredPageMarkdown("|table|Col2|Col12|"), "|table|||");
    assert.equal(
      renderPageMarkdown({
        documentId: "doc-1",
        originalFilename: "alpha.pdf",
        pageNo: 1,
        pageLabel: "1",
        body: "|table|Col2|Col12|",
      }),
      "|table|||\n",
    );

    await persistDocumentPages({
      blobStore,
      storageKey: "doc-1",
      documentId: "doc-1",
      originalFilename: "alpha.pdf",
      contentType: "application/pdf",
      syntheticPages: false,
      parsedDocument: {
        parser_name: "fake",
        parser_version: "v1",
        units: [
          {
            unit_no: 1,
            markdown: "|table|Col2|Col3|\n|---|---|---|\n|A|B|C|",
            content_hash: "u1",
            heading: "Table",
            source_locator: "page-1",
            images: [],
          },
        ],
      },
    });

    assert.equal(
      Buffer.from(await blobStore.get({ objectKey: "documents/doc-1/pages/page-0001.md" })).toString("utf8"),
      "|table|||\n|---|---|---|\n|A|B|C|\n",
    );
  });
});

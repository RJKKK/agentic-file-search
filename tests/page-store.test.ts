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
  parsePageFrontMatter,
  persistDocumentPages,
  renderPageMarkdown,
  validateStorageFilename,
} from "../src/index.js";

describe("page store", () => {
  it("matches legacy filename validation and key builders", () => {
    assert.equal(validateStorageFilename("alpha.pdf"), "alpha.pdf");
    assert.equal(buildDocumentSourceKey("alpha.pdf"), "documents/alpha.pdf/source/alpha.pdf");
    assert.equal(buildDocumentPagesPrefix("alpha.pdf"), "documents/alpha.pdf/pages");
    assert.equal(buildDocumentPageKey("alpha.pdf", 1), "documents/alpha.pdf/pages/page-0001.md");
    assert.throws(() => validateStorageFilename("bad:name.pdf"));
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

    assert.equal(pages[0]?.objectKey, "documents/alpha.pdf/pages/page-0001.md");
    assert.equal(
      Buffer.from(await blobStore.get({ objectKey: "documents/alpha.pdf/pages/page-0001.md" })).toString("utf8"),
      "First\n",
    );
    const [header, body] = parsePageFrontMatter("First\n");
    assert.deepEqual(header, {});
    assert.equal(body, "First\n");
  });
});


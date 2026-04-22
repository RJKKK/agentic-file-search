import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import { LocalBlobStore } from "../src/index.js";

describe("local blob store", () => {
  it("supports put/get/head/materialize/delete/deletePrefix/listPrefix", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-blob-store-"));
    const store = new LocalBlobStore(root);

    const first = await store.put({
      objectKey: "documents/demo/source/demo.md",
      data: Buffer.from("hello", "utf8"),
    });
    await store.put({
      objectKey: "documents/demo/pages/page-0001.md",
      data: Buffer.from("page one", "utf8"),
    });
    await store.put({
      objectKey: "documents/demo/pages/page-0002.md",
      data: Buffer.from("page two", "utf8"),
    });

    assert.equal(Buffer.from(await store.get({ objectKey: first.objectKey })).toString("utf8"), "hello");
    assert.equal((await store.head({ objectKey: first.objectKey }))?.storageUri, `blob://library/default/${first.objectKey}`);
    assert.match(await store.materialize({ objectKey: first.objectKey }), /documents[\\/]demo[\\/]source/);
    assert.deepEqual(
      (await store.listPrefix({ prefix: "documents/demo/pages" })).map((item) => item.objectKey),
      ["documents/demo/pages/page-0001.md", "documents/demo/pages/page-0002.md"],
    );

    assert.equal(await store.delete({ objectKey: first.objectKey }), true);
    assert.equal(await store.head({ objectKey: first.objectKey }), null);
    assert.equal(await store.deletePrefix({ prefix: "documents/demo/pages" }), 2);
    assert.deepEqual(await store.listPrefix({ prefix: "documents/demo/pages" }), []);
  });

  it("rejects object keys that escape the object-store root", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-blob-escape-"));
    const store = new LocalBlobStore(root);

    await assert.rejects(
      store.put({
        objectKey: "../outside.txt",
        data: Buffer.from("x", "utf8"),
      }),
    );
  });
});


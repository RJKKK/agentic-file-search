import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import type { ActionModel } from "../src/agent/agent.js";
import { createHttpServer, LocalBlobStore, SqliteStorage } from "../src/index.js";
import type { PythonDocumentParserExecutor } from "../src/runtime/document-parsing.js";

class StaticParser implements PythonDocumentParserExecutor {
  async parseDocument() {
    return {
      parser_name: "test",
      parser_version: "1",
      units: [
        {
          unit_no: 1,
          markdown: "The purchase price is $45,000,000.",
          content_hash: "hash-1",
          heading: "Purchase Price",
          source_locator: "page-1",
          images: [],
        },
      ],
    };
  }
}

class StopModel implements ActionModel {
  async generateAction() {
    return {
      action: {
        final_result: "The purchase price is $45,000,000. [Source: alpha.pdf, page 1]",
      },
      reason: "Enough evidence.",
    };
  }
}

class DelayedSequenceModel implements ActionModel {
  private index = 0;

  async generateAction() {
    const current = this.index;
    this.index += 1;
    await new Promise((resolve) => setTimeout(resolve, 1_500));
    if (current === 0) {
      return {
        action: {
          tool_name: "glob",
          tool_input: [
            { parameter_name: "directory", parameter_value: "" },
            { parameter_name: "pattern", parameter_value: "page-*.md" },
          ],
        },
        reason: "Inspect pages first.",
      };
    }
    return {
      action: {
        final_result: "Done.",
      },
      reason: "Enough evidence.",
    };
  }
}

function multipartBody(input: { filename: string; content: string; contentType?: string }) {
  const boundary = "----afs-test-boundary";
  const body = Buffer.from(
    [
      `--${boundary}`,
      `Content-Disposition: form-data; name="file"; filename="${input.filename}"`,
      `Content-Type: ${input.contentType ?? "application/pdf"}`,
      "",
      input.content,
      `--${boundary}--`,
      "",
    ].join("\r\n"),
  );
  return {
    body,
    contentType: `multipart/form-data; boundary=${boundary}`,
  };
}

async function createAppFixture() {
  const root = await mkdtemp(join(tmpdir(), "afs-http-"));
  const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
  storage.initialize();
  const app = await createHttpServer({
    storage,
    blobStore: new LocalBlobStore(join(root, "object-store")),
    parser: new StaticParser(),
    model: new StopModel(),
    skillsRoot: join(process.cwd(), "skills"),
  });
  return { app, storage };
}

describe("http server", () => {
  it("supports document upload, listing, detail, pages, search, and delete", async () => {
    const { app, storage } = await createAppFixture();
    const health = await app.inject({ method: "GET", url: "/api/health" });
    assert.equal(health.statusCode, 200);
    assert.equal(health.json().ok, true);

    const root = await app.inject({ method: "GET", url: "/" });
    assert.equal(root.statusCode, 200);
    assert.match(root.body, /Agentic File Search|<html/i);

    const upload = multipartBody({
      filename: "alpha.pdf",
      content: "%PDF fake",
    });
    const uploaded = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": upload.contentType },
      payload: upload.body,
    });
    assert.equal(uploaded.statusCode, 201);
    assert.ok(uploaded.headers["x-trace-id"]);
    const uploadPayload = uploaded.json();
    const docId = uploadPayload.document.id as string;
    assert.equal(uploadPayload.upload_result.page_naming_scheme, "page-0001.md");

    const list = await app.inject({ method: "GET", url: "/api/documents?page=1&page_size=10" });
    assert.equal(list.statusCode, 200);
    assert.equal(list.json().total, 1);

    const detail = await app.inject({ method: "GET", url: `/api/documents/${docId}` });
    assert.equal(detail.statusCode, 200);
    assert.equal(detail.json().page_summary.page_count, 1);

    const pages = await app.inject({ method: "GET", url: `/api/documents/${docId}/pages` });
    assert.equal(pages.statusCode, 200);
    assert.match(pages.json().items[0].markdown, /purchase price/);

    const search = await app.inject({
      method: "POST",
      url: "/api/search",
      payload: {
        query: "purchase price",
        document_ids: [docId],
        limit: 5,
      },
    });
    assert.equal(search.statusCode, 200);
    assert.equal(search.json().hits[0].doc_id, docId);

    const removed = await app.inject({ method: "DELETE", url: `/api/documents/${docId}` });
    assert.equal(removed.statusCode, 200);
    assert.equal(removed.json().deleted, true);
    await app.close();
    storage.close();
  });

  it("supports collection CRUD and document attachment routes", async () => {
    const { app, storage } = await createAppFixture();
    const upload = multipartBody({ filename: "alpha.pdf", content: "%PDF fake" });
    const uploaded = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": upload.contentType },
      payload: upload.body,
    });
    const docId = uploaded.json().document.id as string;

    const created = await app.inject({
      method: "POST",
      url: "/api/collections",
      payload: { name: "Deals" },
    });
    assert.equal(created.statusCode, 200);
    const collectionId = created.json().collection.id as string;

    const attached = await app.inject({
      method: "POST",
      url: `/api/collections/${collectionId}/documents`,
      payload: { document_ids: [docId] },
    });
    assert.equal(attached.statusCode, 200);
    assert.equal(attached.json().attached, 1);

    const listed = await app.inject({
      method: "GET",
      url: `/api/collections/${collectionId}/documents`,
    });
    assert.equal(listed.statusCode, 200);
    assert.equal(listed.json().items[0].id, docId);

    const detached = await app.inject({
      method: "DELETE",
      url: `/api/collections/${collectionId}/documents/${docId}`,
    });
    assert.equal(detached.statusCode, 200);
    assert.equal(detached.json().removed, true);
    await app.close();
    storage.close();
  });

  it("supports exploration session create/get/reply and keeps unsupported index routes explicit", async () => {
    const { app, storage } = await createAppFixture();
    const upload = multipartBody({ filename: "alpha.pdf", content: "%PDF fake" });
    const uploaded = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": upload.contentType },
      payload: upload.body,
    });
    const docId = uploaded.json().document.id as string;

    const started = await app.inject({
      method: "POST",
      url: "/api/explore/sessions",
      payload: {
        task: "What is the purchase price?",
        document_ids: [docId],
      },
    });
    assert.equal(started.statusCode, 200);
    const sessionId = started.json().session_id as string;

    let snapshot = await app.inject({
      method: "GET",
      url: `/api/explore/sessions/${sessionId}`,
    });
    for (let attempt = 0; attempt < 20 && snapshot.json().status === "created"; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 25));
      snapshot = await app.inject({
        method: "GET",
        url: `/api/explore/sessions/${sessionId}`,
      });
    }
    assert.equal(snapshot.statusCode, 200);
    assert.equal(snapshot.json().status, "completed");

    const reply = await app.inject({
      method: "POST",
      url: `/api/explore/sessions/${sessionId}/reply`,
      payload: { response: "Use 2025." },
    });
    assert.equal(reply.statusCode, 409);

    const unsupported = await app.inject({ method: "POST", url: "/api/index", payload: {} });
    assert.equal(unsupported.statusCode, 501);
    assert.equal(unsupported.json().error_code, "index_build_not_supported");

    await app.close();
    storage.close();
  });

  it("streams live SSE events after the initial history replay", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-http-sse-"));
    const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
    storage.initialize();
    const app = await createHttpServer({
      storage,
      blobStore: new LocalBlobStore(join(root, "object-store")),
      parser: new StaticParser(),
      model: new DelayedSequenceModel(),
      skillsRoot: join(process.cwd(), "skills"),
    });
    await app.listen({ host: "127.0.0.1", port: 0 });
    const address = app.server.address();
    assert.ok(address && typeof address === "object");
    const baseUrl = `http://127.0.0.1:${address.port}`;

    const upload = multipartBody({ filename: "alpha.pdf", content: "%PDF fake" });
    const uploaded = await fetch(`${baseUrl}/api/documents`, {
      method: "POST",
      headers: { "content-type": upload.contentType },
      body: upload.body,
    });
    assert.equal(uploaded.status, 201);
    const uploadedPayload = (await uploaded.json()) as { document: { id: string } };

    const started = await fetch(`${baseUrl}/api/explore/sessions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        task: "What is the purchase price?",
        document_ids: [uploadedPayload.document.id],
      }),
    });
    assert.equal(started.status, 200);
    const startedPayload = (await started.json()) as { session_id: string };

    const response = await fetch(
      `${baseUrl}/api/explore/sessions/${startedPayload.session_id}/events`,
      {
        headers: { Accept: "text/event-stream" },
      },
    );
    assert.equal(response.status, 200);
    assert.ok(response.body);

    const reader = response.body!.getReader();
    let buffer = "";
    const eventTypes: string[] = [];
    const deadline = Date.now() + 3000;
    while (Date.now() < deadline && !eventTypes.includes("tool_call")) {
      const nextChunk = reader.read();
      const timeout = new Promise<{ timeout: true }>((resolve) => {
        setTimeout(() => resolve({ timeout: true }), 2_200);
      });
      const result = await Promise.race([nextChunk, timeout]);
      if ("timeout" in result) {
        continue;
      }
      const { value, done } = result;
      if (done) {
        break;
      }
      buffer += Buffer.from(value).toString("utf8");
      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex >= 0) {
        const block = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const eventLine = block
          .split("\n")
          .find((line) => line.startsWith("event: "));
        if (eventLine) {
          eventTypes.push(eventLine.slice("event: ".length));
        }
        separatorIndex = buffer.indexOf("\n\n");
      }
    }

    await reader.cancel();
    assert.deepEqual(eventTypes.slice(0, 2), ["start", "context_scope_updated"]);
    assert.ok(eventTypes.includes("tool_call"));

    await app.close();
    storage.close();
  });
});

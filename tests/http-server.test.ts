import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { once } from "node:events";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import type { ActionModel } from "../src/agent/agent.js";
import { createHttpServer, LocalBlobStore, SqliteStorage } from "../src/index.js";
import type { PythonDocumentParserExecutor } from "../src/runtime/document-parsing.js";
import { TraditionalRagService } from "../src/runtime/traditional-rag.js";

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
          blocks: [],
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

function multipartBody(input: {
  filename: string;
  content: string;
  contentType?: string;
  fields?: Record<string, string>;
}) {
  const boundary = "----afs-test-boundary";
  const fieldParts = Object.entries(input.fields ?? {}).flatMap(([name, value]) => [
    `--${boundary}`,
    `Content-Disposition: form-data; name="${name}"`,
    "",
    value,
  ]);
  const body = Buffer.from(
    [
      ...fieldParts,
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
  const blobStore = new LocalBlobStore(join(root, "object-store"));
  const app = await createHttpServer({
    storage,
    blobStore,
    parser: new StaticParser(),
    model: new StopModel(),
    skillsRoot: join(process.cwd(), "skills"),
  });
  return { app, storage, blobStore };
}

async function waitForDocumentDetail(
  app: Awaited<ReturnType<typeof createHttpServer>>,
  docId: string,
  predicate: (document: Record<string, unknown>) => boolean,
): Promise<Record<string, unknown>> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const detail = await app.inject({ method: "GET", url: `/api/documents/${docId}` });
    if (detail.statusCode === 200 && predicate(detail.json().document as Record<string, unknown>)) {
      return detail.json().document as Record<string, unknown>;
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error(`Document ${docId} did not reach the expected state.`);
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
      fields: {
        enable_embedding: "false",
        enable_image_semantic: "false",
      },
    });
    const uploaded = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": upload.contentType },
      payload: upload.body,
    });
    assert.equal(uploaded.statusCode, 202);
    assert.ok(uploaded.headers["x-trace-id"]);
    const uploadPayload = uploaded.json();
    const docId = uploadPayload.document.id as string;
    assert.equal(uploadPayload.document.retrieval_chunking_strategy, "small_to_big");
    assert.equal(uploadPayload.document.fixed_chunk_chars, null);
    assert.equal(uploadPayload.task.options.chunking_strategy, "small_to_big");
    assert.equal(uploadPayload.task.options.fixed_chunk_chars, null);

    const list = await app.inject({ method: "GET", url: "/api/documents?page=1&page_size=10" });
    assert.equal(list.statusCode, 200);
    assert.equal(list.json().total, 1);
    assert.equal(list.json().items[0].retrieval_chunking_strategy, "small_to_big");

    await waitForDocumentDetail(
      app,
      docId,
      (document) => document.upload_status === "completed" && document.page_count === 1,
    );
    const detail = await app.inject({ method: "GET", url: `/api/documents/${docId}` });
    assert.equal(detail.statusCode, 200);
    assert.equal(detail.json().page_summary.page_count, 1);
    assert.equal(detail.json().document.retrieval_chunking_strategy, "small_to_big");
    assert.equal(detail.json().document.fixed_chunk_chars, null);

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

  it("accepts fixed chunking options on upload and reparse, and serves fixed retrieval chunk content", async () => {
    const { app, storage } = await createAppFixture();
    const upload = multipartBody({
      filename: "fixed.pdf",
      content: "%PDF fake",
      fields: {
        enable_embedding: "false",
        enable_image_semantic: "false",
      },
    });
    const uploaded = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": upload.contentType },
      payload: upload.body,
    });
    assert.equal(uploaded.statusCode, 202);
    const initialPayload = uploaded.json();
    const docId = initialPayload.document.id as string;

    const fixedUpload = multipartBody({
      filename: "fixed-two.pdf",
      content: "%PDF fixed",
      fields: {
        enable_embedding: "false",
        enable_image_semantic: "false",
        chunking_strategy: "fixed",
        fixed_chunk_chars: "120",
      },
    });
    const fixedUploaded = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": fixedUpload.contentType },
      payload: fixedUpload.body,
    });
    assert.equal(fixedUploaded.statusCode, 202);
    const fixedPayload = fixedUploaded.json();
    const fixedDocId = fixedPayload.document.id as string;
    assert.equal(fixedPayload.document.retrieval_chunking_strategy, "fixed");
    assert.equal(fixedPayload.document.fixed_chunk_chars, 120);
    assert.equal(fixedPayload.task.options.chunking_strategy, "fixed");
    assert.equal(fixedPayload.task.options.fixed_chunk_chars, 120);

    const reparsed = await app.inject({
      method: "POST",
      url: `/api/documents/${encodeURIComponent(docId)}/parse`,
      payload: {
        force: true,
        chunking_strategy: "fixed",
        fixed_chunk_chars: 64,
        enable_embedding: false,
        enable_image_semantic: false,
      },
    });
    assert.equal(reparsed.statusCode, 202);
    assert.equal(reparsed.json().task.options.chunking_strategy, "fixed");
    assert.equal(reparsed.json().task.options.fixed_chunk_chars, 64);
    const reparsedDocument = await waitForDocumentDetail(
      app,
      docId,
      (document) =>
        document.retrieval_chunking_strategy === "fixed" && document.fixed_chunk_chars === 64,
    );
    assert.equal(reparsedDocument.retrieval_chunking_strategy, "fixed");
    assert.equal(reparsedDocument.fixed_chunk_chars, 64);

    storage.replaceDocumentChunks(fixedDocId, [
      {
        id: "manual-dchunk-1",
        documentId: fixedDocId,
        ordinal: 0,
        referenceRetrievalChunkId: "manual-frchunk-1",
        pageNo: 1,
        documentIndex: 0,
        pageIndex: 0,
        blockType: "paragraph",
        bboxJson: JSON.stringify([0, 0, 1, 1]),
        contentMd: "Purchase price fixed chunk.",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: JSON.stringify([1]),
        mergedBboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
    ]);
    storage.replaceFixedRetrievalChunks(fixedDocId, [
      {
        id: "manual-frchunk-1",
        documentId: fixedDocId,
        ordinal: 0,
        contentMd: "Purchase price fixed chunk.",
        sizeClass: "normal",
        summaryText: null,
        sourceDocumentChunkIdsJson: JSON.stringify(["manual-dchunk-1"]),
        pageNosJson: JSON.stringify([1]),
        sourceLocator: "page-1",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
    ]);
    const fixedChunkContent = await app.inject({
      method: "GET",
      url: "/api/retrieval-chunks/manual-frchunk-1/content",
    });
    assert.equal(fixedChunkContent.statusCode, 200);
    assert.match(fixedChunkContent.json().chunk.content_md, /fixed chunk/i);

    await app.close();
    storage.close();
  });

  it("supports collection CRUD and document attachment routes", async () => {
    const { app, storage } = await createAppFixture();
    const upload = multipartBody({
      filename: "alpha.pdf",
      content: "%PDF fake",
      fields: {
        enable_embedding: "false",
        enable_image_semantic: "false",
      },
    });
    const uploaded = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": upload.contentType },
      payload: upload.body,
    });
    const docId = uploaded.json().document.id as string;
    const uploadB = multipartBody({
      filename: "beta.pdf",
      content: "%PDF fake",
      fields: {
        enable_embedding: "false",
        enable_image_semantic: "false",
      },
    });
    const uploadedB = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": uploadB.contentType },
      payload: uploadB.body,
    });
    const docIdB = uploadedB.json().document.id as string;
    await Promise.all([
      waitForDocumentDetail(app, docId, (document) => document.upload_status === "completed"),
      waitForDocumentDetail(app, docIdB, (document) => document.upload_status === "completed"),
    ]);

    const created = await app.inject({
      method: "POST",
      url: "/api/collections",
      payload: { name: "Deals" },
    });
    assert.equal(created.statusCode, 201);
    const collectionId = created.json().collection.id as string;
    assert.equal(created.json().collection.document_count, 0);

    const duplicate = await app.inject({
      method: "POST",
      url: "/api/collections",
      payload: { name: " Deals " },
    });
    assert.equal(duplicate.statusCode, 409);
    assert.equal(duplicate.json().error_code, "duplicate_collection_name");

    const other = await app.inject({
      method: "POST",
      url: "/api/collections",
      payload: { name: "Other" },
    });
    const otherId = other.json().collection.id as string;

    const duplicateRename = await app.inject({
      method: "PATCH",
      url: `/api/collections/${otherId}`,
      payload: { name: "Deals" },
    });
    assert.equal(duplicateRename.statusCode, 409);

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
    assert.equal(listed.json().collection.document_count, 1);

    const replaced = await app.inject({
      method: "PUT",
      url: `/api/collections/${collectionId}/documents`,
      payload: { document_ids: [docId, docIdB] },
    });
    assert.equal(replaced.statusCode, 200);
    assert.deepEqual(
      replaced.json().items.map((item: { id: string }) => item.id),
      [docId, docIdB],
    );

    const scopedSearch = await app.inject({
      method: "POST",
      url: "/api/search",
      payload: {
        query: "purchase price",
        document_ids: [docId],
        collection_ids: [collectionId],
        limit: 10,
      },
    });
    assert.equal(scopedSearch.statusCode, 200);
    assert.equal(scopedSearch.json().collection_id, collectionId);
    assert.deepEqual(scopedSearch.json().collection_ids, [collectionId]);
    assert.deepEqual(scopedSearch.json().document_ids, [docId, docIdB]);

    const documentCollections = await app.inject({
      method: "GET",
      url: `/api/documents/${docId}/collections`,
    });
    assert.equal(documentCollections.statusCode, 200);
    assert.deepEqual(
      documentCollections.json().items.map((item: { id: string }) => item.id),
      [collectionId],
    );

    const documentCollectionsUpdated = await app.inject({
      method: "PUT",
      url: `/api/documents/${docId}/collections`,
      payload: { collection_ids: [otherId] },
    });
    assert.equal(documentCollectionsUpdated.statusCode, 200);
    assert.deepEqual(
      documentCollectionsUpdated.json().items.map((item: { id: string }) => item.id),
      [otherId],
    );

    const detached = await app.inject({
      method: "DELETE",
      url: `/api/collections/${collectionId}/documents/${docId}`,
    });
    assert.equal(detached.statusCode, 200);
    assert.equal(detached.json().removed, false);

    const deleted = await app.inject({ method: "DELETE", url: `/api/collections/${collectionId}` });
    assert.equal(deleted.statusCode, 200);
    const recreated = await app.inject({
      method: "POST",
      url: "/api/collections",
      payload: { name: "Deals" },
    });
    assert.equal(recreated.statusCode, 201);
    await app.close();
    storage.close();
  });

  it("supports traditional rag answers plus chunk and image asset routes", async () => {
    const { app, storage, blobStore } = await createAppFixture();
    const previousServerBaseUrl = process.env.FS_EXPLORER_SERVER_BASE_URL;
    const previousPort = process.env.FS_EXPLORER_PORT;
    delete process.env.FS_EXPLORER_SERVER_BASE_URL;
    process.env.FS_EXPLORER_PORT = "8000";
    try {
      const corpusId = storage.getOrCreateCorpus("test-root");
      const docId = "manual-doc";
      storage.upsertDocumentStub({
        id: docId,
        corpusId,
        relativePath: "alpha.pdf",
        absolutePath: join(tmpdir(), "alpha.pdf"),
        content: "",
        metadataJson: "{}",
        fileMtime: 1,
        fileSize: 1,
        contentSha256: "hash-alpha",
        originalFilename: "alpha.pdf",
        objectKey: "documents/alpha.pdf",
        sourceObjectKey: "documents/alpha.pdf",
        pagesPrefix: "documents/alpha/pages",
        storageUri: "storage://alpha.pdf",
        contentType: "application/pdf",
        uploadStatus: "completed",
        pageCount: 1,
        parsedContentSha256: "parsed-alpha",
        parsedIsComplete: true,
        embeddingEnabled: false,
        hasEmbeddings: false,
        imageSemanticEnabled: false,
      });
      storage.replaceDocumentChunks(docId, [
        {
          id: "manual-dchunk-1",
          documentId: docId,
          ordinal: 1,
          referenceRetrievalChunkId: "manual-rchunk-1",
          pageNo: 1,
          documentIndex: 1,
          pageIndex: 1,
          blockType: "paragraph",
          bboxJson: "[0,0,1,1]",
          contentMd: "The purchase price is $45,000,000.",
          sizeClass: "normal",
          summaryText: null,
          isSplitFromOversized: false,
          splitIndex: 0,
          splitCount: 1,
          mergedPageNosJson: "[1]",
          mergedBboxesJson: "[[0,0,1,1]]",
        },
      ]);
      storage.replaceRetrievalChunks(docId, [
        {
          id: "manual-rchunk-1",
          documentId: docId,
          ordinal: 1,
          contentMd: "The purchase price is $45,000,000.",
          sizeClass: "normal",
          sourceDocumentChunkIdsJson: JSON.stringify(["manual-dchunk-1"]),
          pageNosJson: JSON.stringify([1]),
          sourceLocator: "page-1",
          bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
        },
      ]);
      const chunks = storage.listDocumentChunks(docId);
      assert.equal(chunks.length, 1);

      const rag = await app.inject({
        method: "POST",
        url: "/api/rag/query",
        payload: {
          question: "What is the purchase price?",
          mode: "keyword",
          document_ids: [docId],
        },
      });
      assert.equal(rag.statusCode, 200);
      assert.equal(rag.json().used_chunks[0].reference_id, "manual-rchunk-1");
      assert.equal(rag.json().used_chunks[0].reference_kind, "retrieval_chunk");
      assert.equal("content" in rag.json().used_chunks[0], false);
      assert.equal(rag.json().used_chunks[0].citation_no, 1);
      assert.equal(rag.json().used_chunks[0].document_name, "alpha.pdf");
      assert.match(
        rag.json().used_chunks[0].source_link,
        /^http:\/\/localhost:8000\/api\/retrieval-chunks\/manual-rchunk-1\/content$/,
      );

      const retrieve = await app.inject({
        method: "POST",
        url: "/api/rag/retrieve",
        payload: {
          question: "What is the purchase price?",
          mode: "keyword",
          document_ids: [docId],
        },
      });
      assert.equal(retrieve.statusCode, 200);
      assert.equal(retrieve.json().retrieved_chunks[0].reference_id, "manual-rchunk-1");
      assert.equal(retrieve.json().retrieved_chunks[0].reference_kind, "retrieval_chunk");
      assert.equal("content" in retrieve.json().retrieved_chunks[0], false);
      assert.match(String(retrieve.json().retrieved_chunks[0].debug_enriched_content || ""), /purchase price/i);
      assert.doesNotMatch(
        String(retrieve.json().retrieved_chunks[0].debug_enriched_content || ""),
        /\/api\/retrieval-chunks\/manual-rchunk-1\/content/,
      );

      const chunkContent = await app.inject({
        method: "GET",
        url: "/api/retrieval-chunks/manual-rchunk-1/content",
      });
      assert.equal(chunkContent.statusCode, 200);
      assert.match(chunkContent.json().chunk.content_md, /purchase price/i);

      const documentChunkContent = await app.inject({
        method: "GET",
        url: `/api/document-chunks/${chunks[0]?.id}/content`,
      });
      assert.equal(documentChunkContent.statusCode, 200);
      assert.match(documentChunkContent.json().chunk.content_md, /purchase price/i);

      await blobStore.put({
        objectKey: "documents/alpha.pdf/images/img-1.png",
        data: Buffer.from("png-bytes", "utf8"),
      });
      storage.upsertImageSemantics([
        {
          imageHash: "img-1",
          sourceDocumentId: docId,
          sourcePageNo: 1,
          sourceImageIndex: 0,
          mimeType: "image/png",
          objectKey: "documents/alpha.pdf/images/img-1.png",
          accessibleUrl: "/api/assets/images/img-1",
        },
      ]);
      const image = await app.inject({
        method: "GET",
        url: "/api/assets/images/img-1",
      });
      assert.equal(image.statusCode, 200);
      assert.equal(image.headers["content-type"], "image/png");
      assert.equal(image.body, "png-bytes");
    } finally {
      process.env.FS_EXPLORER_SERVER_BASE_URL = previousServerBaseUrl;
      process.env.FS_EXPLORER_PORT = previousPort;
      await app.close();
      storage.close();
    }
  });

  it("returns the final traditional rag prompt preview before the model request", async () => {
    const { app, storage } = await createAppFixture();
    const previousServerBaseUrl = process.env.FS_EXPLORER_SERVER_BASE_URL;
    const previousPort = process.env.FS_EXPLORER_PORT;
    process.env.FS_EXPLORER_SERVER_BASE_URL = "https://rag.example.com";
    process.env.FS_EXPLORER_PORT = "8000";
    try {
      const corpusId = storage.getOrCreateCorpus("test-root");
      const docId = "prompt-preview-doc";
      storage.upsertDocumentStub({
        id: docId,
        corpusId,
        relativePath: "prompt-preview.pdf",
        absolutePath: join(tmpdir(), "prompt-preview.pdf"),
        content: "",
        metadataJson: "{}",
        fileMtime: 1,
        fileSize: 1,
        contentSha256: "hash-prompt-preview",
        originalFilename: "prompt-preview.pdf",
        objectKey: "documents/prompt-preview.pdf",
        sourceObjectKey: "documents/prompt-preview.pdf",
        pagesPrefix: "documents/prompt-preview/pages",
        storageUri: "storage://prompt-preview.pdf",
        contentType: "application/pdf",
        uploadStatus: "completed",
        pageCount: 1,
        parsedContentSha256: "parsed-prompt-preview",
        parsedIsComplete: true,
        embeddingEnabled: false,
        hasEmbeddings: false,
        imageSemanticEnabled: false,
      });
      storage.replaceDocumentChunks(docId, [
        {
          id: "prompt-dchunk-1",
          documentId: docId,
          ordinal: 1,
          referenceRetrievalChunkId: "prompt-rchunk-1",
          pageNo: 1,
          documentIndex: 1,
          pageIndex: 1,
          blockType: "paragraph",
          bboxJson: "[0,0,1,1]",
          contentMd: "The purchase price is $45,000,000.",
          sizeClass: "normal",
          summaryText: "Purchase price appendix summary",
          isSplitFromOversized: true,
          splitIndex: 0,
          splitCount: 1,
          mergedPageNosJson: "[1]",
          mergedBboxesJson: "[[0,0,1,1]]",
        },
      ]);
      storage.replaceRetrievalChunks(docId, [
        {
          id: "prompt-rchunk-1",
          documentId: docId,
          ordinal: 1,
          contentMd: "The purchase price is $45,000,000.",
          sizeClass: "oversized",
          summaryText: "Purchase price appendix summary",
          sourceDocumentChunkIdsJson: JSON.stringify(["prompt-dchunk-1"]),
          pageNosJson: JSON.stringify([1]),
          sourceLocator: "page-1",
          bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
        },
      ]);
      const preview = await app.inject({
        method: "POST",
        url: "/api/rag/prompt-preview",
        payload: {
          question: "What is the purchase price?",
          mode: "keyword",
          document_ids: [docId],
        },
      });
      assert.equal(preview.statusCode, 200);
      const payload = preview.json();
      assert.equal(payload.prompt_variant, "stream");
      assert.equal(payload.messages[0]?.role, "system");
      assert.equal(payload.messages[1]?.role, "user");
      assert.match(String(payload.system_prompt || ""), /direct link/i);
      assert.match(String(payload.user_prompt || ""), /Question:/);
      assert.match(String(payload.user_prompt || ""), /Evidence:/);
      assert.match(String(payload.user_prompt || ""), /chunk\[1\] summary:/i);
      assert.match(String(payload.user_prompt || ""), /chunk\[1\] compressed excerpt:/i);
      assert.match(String(payload.user_prompt || ""), /compressed excerpt/i);
      assert.match(
        String(payload.user_prompt || ""),
        /Full content is available at https:\/\/rag\.example\.com\/api\/retrieval-chunks\/prompt-rchunk-1\/content, and you may provide this link directly to the user when it is helpful\./i,
      );
      assert.match(
        String(payload.user_prompt || ""),
        /https:\/\/rag\.example\.com\/api\/retrieval-chunks\/prompt-rchunk-1\/content/,
      );
      assert.doesNotMatch(String(payload.user_prompt || ""), /Source:/);
      assert.match(String(payload.user_prompt || ""), /purchase price/i);
      assert.match(String(payload.request_body_json || ""), /"stream": true/);
    } finally {
      process.env.FS_EXPLORER_SERVER_BASE_URL = previousServerBaseUrl;
      process.env.FS_EXPLORER_PORT = previousPort;
      await app.close();
      storage.close();
    }
  });

  it("maps model citation numbers back to lightweight retrieved chunks", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-rag-"));
    const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
    storage.initialize();
    const blobStore = new LocalBlobStore(join(root, "object-store"));
    const rag = new TraditionalRagService(storage, blobStore);
    const corpusId = storage.getOrCreateCorpus(root);
    const docId = "doc-semantic";

    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "semantic.pdf",
      absolutePath: join(root, "semantic.pdf"),
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 1,
      contentSha256: "hash-semantic",
      originalFilename: "semantic.pdf",
      objectKey: "documents/semantic.pdf",
      sourceObjectKey: "documents/semantic.pdf",
      pagesPrefix: "documents/semantic/pages",
      storageUri: "storage://semantic.pdf",
      contentType: "application/pdf",
      uploadStatus: "completed",
      pageCount: 5,
      parsedContentSha256: "parsed-semantic",
      parsedIsComplete: true,
      embeddingEnabled: true,
      hasEmbeddings: true,
      imageSemanticEnabled: false,
    });

    storage.replaceDocumentChunks(
      docId,
      Array.from({ length: 5 }, (_, index) => ({
        id: `dchunk-${index + 1}`,
        documentId: docId,
        ordinal: index + 1,
        referenceRetrievalChunkId: index === 4 ? null : `rchunk-${index + 1}`,
        pageNo: index + 1,
        documentIndex: index + 1,
        pageIndex: 1,
        blockType: "paragraph",
        bboxJson: "[0,0,1,1]",
        contentMd: `purchase price evidence ${index + 1}`,
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: index === 4,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: JSON.stringify([index + 1]),
        mergedBboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      })),
    );
    storage.replaceRetrievalChunks(
      docId,
      Array.from({ length: 4 }, (_, index) => ({
        id: `rchunk-${index + 1}`,
        documentId: docId,
        ordinal: index + 1,
        contentMd: `purchase price evidence ${index + 1}`,
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify([`dchunk-${index + 1}`]),
        pageNosJson: JSON.stringify([index + 1]),
        sourceLocator: `page-${index + 1}`,
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      })),
    );
    const originalFetch = globalThis.fetch;
    const previousEmbeddingUrl = process.env.EMBEDDING_BASE_URL;
    const previousEmbeddingKey = process.env.EMBEDDING_API_KEY;
    const previousTextUrl = process.env.TEXT_BASE_URL;
    const previousTextKey = process.env.TEXT_API_KEY;
    const previousQdrantUrl = process.env.QDRANT_URL;

    process.env.EMBEDDING_BASE_URL = "https://example.test/v1";
    process.env.EMBEDDING_API_KEY = "embed-test-key";
    process.env.TEXT_BASE_URL = "https://example.test/v1";
    process.env.TEXT_API_KEY = "text-test-key";
    process.env.QDRANT_URL = "https://qdrant.test";

    globalThis.fetch = (async (input, init) => {
      const url = String(input);
      if (url === "https://example.test/v1/embeddings") {
        return new Response(
          JSON.stringify({
            data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://qdrant.test/collections/doc_doc-semantic/points/search") {
        return new Response(
          JSON.stringify({
            result: Array.from({ length: 5 }, (_, index) => ({
              id: `qdrant-${index + 1}`,
              score: 10 - index,
              payload: { retrieval_unit_id: `dchunk-${index + 1}` },
            })),
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://example.test/v1/chat/completions") {
        const body = String(init?.body ?? "");
        if (body.includes('"stream":true')) {
          return new Response("stream unsupported", { status: 400 });
        }
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    answer: "The strongest evidence is [2] and [5].",
                    citations: [2, 5],
                  }),
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    try {
      const result = await rag.query({
        question: "purchase price",
        mode: "semantic",
        documentIds: [docId],
      });

      assert.deepEqual(
        result.used_chunks.map((chunk) => chunk.citation_no),
        [2, 5],
      );
      assert.deepEqual(
        result.used_chunks.map((chunk) => chunk.reference_id),
        ["rchunk-2", "dchunk-5"],
      );
      assert.deepEqual(
        result.used_chunks.map((chunk) => chunk.reference_kind),
        ["retrieval_chunk", "document_chunk"],
      );
      assert.deepEqual(result.detail_chunks?.map((chunk) => chunk.reference_id), ["dchunk-5"]);
      assert.equal(result.used_chunks[0]?.document_name, "semantic.pdf");
      assert.equal("content" in result.used_chunks[0], false);
      assert.equal("debug_enriched_content" in result.used_chunks[0], false);
      assert.match(result.answer, /\/api\/document-chunks\/dchunk-5\/content/);

      const streamEvents = [];
      for await (const event of rag.streamQuery({
        question: "purchase price",
        mode: "semantic",
        documentIds: [docId],
      })) {
        streamEvents.push(event);
      }
      assert.equal(streamEvents[0]?.type, "start");
      assert.equal(streamEvents.some((event) => event.type === "answer_delta"), true);
      const completeEvent = streamEvents.find((event) => event.type === "complete");
      assert.deepEqual(
        completeEvent?.used_chunks.map((chunk) => chunk.citation_no),
        [2, 5],
      );
      assert.deepEqual(completeEvent?.detail_chunks.map((chunk) => chunk.reference_id), ["dchunk-5"]);
      assert.match(String(completeEvent?.answer || ""), /\/api\/document-chunks\/dchunk-5\/content/);
    } finally {
      globalThis.fetch = originalFetch;
      process.env.EMBEDDING_BASE_URL = previousEmbeddingUrl;
      process.env.EMBEDDING_API_KEY = previousEmbeddingKey;
      process.env.TEXT_BASE_URL = previousTextUrl;
      process.env.TEXT_API_KEY = previousTextKey;
      process.env.QDRANT_URL = previousQdrantUrl;
      storage.close();
    }
  });

  it("enriches retrieval evidence with adjacent retrieval and oversized context segments", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-cer-"));
    const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
    storage.initialize();
    const blobStore = new LocalBlobStore(join(root, "object-store"));
    const rag = new TraditionalRagService(storage, blobStore);
    const previousServerBaseUrl = process.env.FS_EXPLORER_SERVER_BASE_URL;
    const previousPort = process.env.FS_EXPLORER_PORT;
    delete process.env.FS_EXPLORER_SERVER_BASE_URL;
    process.env.FS_EXPLORER_PORT = "8000";
    const corpusId = storage.getOrCreateCorpus(root);
    const docId = "doc-cer";
    const oversizedContent =
      "图片链接：![image](/api/assets/images/demo)\n\n" +
      `${"leading context ".repeat(120)}purchase price appears inside the oversized appendix. ` +
      `${"trailing context ".repeat(120)}`;

    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "cer.pdf",
      absolutePath: join(root, "cer.pdf"),
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 1,
      contentSha256: "hash-cer",
      originalFilename: "cer.pdf",
      objectKey: "documents/cer.pdf",
      sourceObjectKey: "documents/cer.pdf",
      pagesPrefix: "documents/cer/pages",
      storageUri: "storage://cer.pdf",
      contentType: "application/pdf",
      uploadStatus: "completed",
      pageCount: 4,
      parsedContentSha256: "parsed-cer",
      parsedIsComplete: true,
      embeddingEnabled: true,
      hasEmbeddings: true,
      imageSemanticEnabled: false,
    });

    storage.replaceDocumentChunks(docId, [
      {
        id: "dchunk-ov-1",
        documentId: docId,
        ordinal: 1,
        referenceRetrievalChunkId: "rchunk-1",
        pageNo: 1,
        documentIndex: 1,
        pageIndex: 1,
        blockType: "paragraph",
        bboxJson: "[0,0,1,1]",
        contentMd: oversizedContent.slice(0, Math.ceil(oversizedContent.length / 2)),
        sizeClass: "normal",
        summaryText: "Oversized appendix summary",
        isSplitFromOversized: true,
        splitIndex: 0,
        splitCount: 2,
        mergedPageNosJson: "[1]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "dchunk-2",
        documentId: docId,
        ordinal: 2,
        referenceRetrievalChunkId: "rchunk-2",
        pageNo: 2,
        documentIndex: 2,
        pageIndex: 1,
        blockType: "paragraph",
        bboxJson: "[0,0,1,1]",
        contentMd: "Context before the purchase discussion.",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[2]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "dchunk-3",
        documentId: docId,
        ordinal: 3,
        referenceRetrievalChunkId: "rchunk-3",
        pageNo: 3,
        documentIndex: 3,
        pageIndex: 1,
        blockType: "paragraph",
        bboxJson: "[0,0,1,1]",
        contentMd: "Purchase price evidence in the main context block.",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[3]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "dchunk-ov-2",
        documentId: docId,
        ordinal: 4,
        referenceRetrievalChunkId: "rchunk-1",
        pageNo: 1,
        documentIndex: 1,
        pageIndex: 1,
        blockType: "paragraph",
        bboxJson: "[0,0,1,1]",
        contentMd: oversizedContent.slice(Math.ceil(oversizedContent.length / 2)),
        sizeClass: "normal",
        summaryText: "Oversized appendix summary",
        isSplitFromOversized: true,
        splitIndex: 1,
        splitCount: 2,
        mergedPageNosJson: "[1]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "dchunk-4",
        documentId: docId,
        ordinal: 5,
        referenceRetrievalChunkId: "rchunk-4",
        pageNo: 4,
        documentIndex: 4,
        pageIndex: 1,
        blockType: "paragraph",
        bboxJson: "[0,0,1,1]",
        contentMd: "Context after the oversized appendix.",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[4]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
    ]);

    storage.replaceRetrievalChunks(docId, [
      {
        id: "rchunk-1",
        documentId: docId,
        ordinal: 1,
        contentMd: oversizedContent,
        sizeClass: "oversized",
        summaryText: "Oversized appendix summary",
        sourceDocumentChunkIdsJson: JSON.stringify(["dchunk-ov-1", "dchunk-ov-2"]),
        pageNosJson: JSON.stringify([1]),
        sourceLocator: "page-1",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
      {
        id: "rchunk-2",
        documentId: docId,
        ordinal: 2,
        contentMd: "Context before the purchase discussion.",
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["dchunk-2"]),
        pageNosJson: JSON.stringify([2]),
        sourceLocator: "page-2",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
      {
        id: "rchunk-3",
        documentId: docId,
        ordinal: 3,
        contentMd: "Purchase price evidence in the main context block.",
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["dchunk-3"]),
        pageNosJson: JSON.stringify([3]),
        sourceLocator: "page-3",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
      {
        id: "rchunk-4",
        documentId: docId,
        ordinal: 4,
        contentMd: "Context after the oversized appendix.",
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["dchunk-4"]),
        pageNosJson: JSON.stringify([4]),
        sourceLocator: "page-4",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
    ]);

    const originalFetch = globalThis.fetch;
    const previousEmbeddingUrl = process.env.EMBEDDING_BASE_URL;
    const previousEmbeddingKey = process.env.EMBEDDING_API_KEY;
    const previousTextUrl = process.env.TEXT_BASE_URL;
    const previousTextKey = process.env.TEXT_API_KEY;
    const previousQdrantUrl = process.env.QDRANT_URL;

    process.env.EMBEDDING_BASE_URL = "https://example.test/v1";
    process.env.EMBEDDING_API_KEY = "embed-test-key";
    process.env.TEXT_BASE_URL = "https://example.test/v1";
    process.env.TEXT_API_KEY = "text-test-key";
    process.env.QDRANT_URL = "https://qdrant.test";

    globalThis.fetch = (async (input, init) => {
      const url = String(input);
      if (url === "https://example.test/v1/embeddings") {
        return new Response(
          JSON.stringify({
            data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://qdrant.test/collections/doc_doc-cer/points/search") {
        return new Response(
          JSON.stringify({
            result: [
              { id: "qdrant-1", score: 10, payload: { retrieval_unit_id: "dchunk-3" } },
              { id: "qdrant-2", score: 9, payload: { retrieval_unit_id: "dchunk-ov-1" } },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://example.test/v1/chat/completions") {
        const body = String(init?.body ?? "");
        if (body.includes('"stream":true')) {
          return new Response("stream unsupported", { status: 400 });
        }
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    answer:
                      "The answer is supported by [1] and [2]. Full detail is available at [Citation 2 full chunk](http://localhost:8000/api/retrieval-chunks/rchunk-1/content).",
                    citations: [1, 2],
                  }),
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    try {
      const retrieveResult = await rag.retrieve({
        question: "purchase price",
        mode: "semantic",
        documentIds: [docId],
      });

      const retrievalChunk = retrieveResult.retrieved_chunks.find((chunk) => chunk.reference_id === "rchunk-3");
      assert.equal(retrievalChunk?.debug_cer_applied, true);
      assert.deepEqual(
        retrievalChunk?.debug_cer_context_refs?.map((item) => item.reference_id),
        ["rchunk-1", "rchunk-4"],
      );
      assert.match(String(retrievalChunk?.debug_enriched_content || ""), /Oversized appendix summary/);
      assert.match(String(retrievalChunk?.debug_enriched_content || ""), /chunk\[1-p-1\] summary:/i);
      assert.match(String(retrievalChunk?.debug_enriched_content || ""), /chunk\[1\] compressed excerpt:/i);
      assert.match(String(retrievalChunk?.debug_enriched_content || ""), /chunk\[1-n-1\] compressed excerpt:/i);
      assert.match(String(retrievalChunk?.debug_enriched_content || ""), /compressed excerpt/i);
      assert.match(String(retrievalChunk?.debug_enriched_content || ""), /!\[image\]\(\/api\/assets\/images\/demo\)/);
      assert.match(String(retrievalChunk?.debug_enriched_content || ""), /\/api\/retrieval-chunks\/rchunk-1\/content/);
      assert.doesNotMatch(String(retrievalChunk?.debug_enriched_content || ""), /!\[image\]\(\/api\/assets\/images\/dem(?:\)|$)/);
      assert.doesNotMatch(String(retrievalChunk?.debug_enriched_content || ""), /\/api\/retrieval-chunks\/rchunk-2\/content/);
      assert.doesNotMatch(String(retrievalChunk?.debug_enriched_content || ""), /\/api\/retrieval-chunks\/rchunk-3\/content/);
      assert.doesNotMatch(String(retrievalChunk?.debug_enriched_content || ""), /\/api\/retrieval-chunks\/rchunk-4\/content/);

      const oversizedChunk = retrieveResult.retrieved_chunks.find((chunk) => chunk.reference_id === "rchunk-1");
      assert.equal(oversizedChunk?.reference_kind, "retrieval_chunk");
      assert.equal(oversizedChunk?.debug_cer_applied, true);
      assert.deepEqual(
        oversizedChunk?.debug_cer_context_refs?.map((item) => item.reference_id),
        ["rchunk-2"],
      );
      assert.match(String(oversizedChunk?.debug_enriched_content || ""), /Oversized appendix summary/);
      assert.match(String(oversizedChunk?.debug_enriched_content || ""), /chunk\[2\] summary:/i);
      assert.match(String(oversizedChunk?.debug_enriched_content || ""), /chunk\[2-n-1\] compressed excerpt:/i);
      assert.match(String(oversizedChunk?.debug_enriched_content || ""), /compressed excerpt/i);
      assert.match(String(oversizedChunk?.debug_enriched_content || ""), /!\[image\]\(\/api\/assets\/images\/demo\)/);
      assert.doesNotMatch(String(oversizedChunk?.debug_enriched_content || ""), /\/api\/retrieval-chunks\/rchunk-2\/content/);
      assert.doesNotMatch(String(oversizedChunk?.debug_enriched_content || ""), /\/api\/retrieval-chunks\/rchunk-4\/content/);
      assert.match(String(oversizedChunk?.debug_enriched_content || ""), /\/api\/retrieval-chunks\/rchunk-1\/content/);
      assert.doesNotMatch(String(oversizedChunk?.debug_enriched_content || ""), /!\[image\]\(\/api\/assets\/images\/dem(?:\)|$)/);

      const queryResult = await rag.query({
        question: "purchase price",
        mode: "semantic",
        documentIds: [docId],
      });
      assert.deepEqual(queryResult.detail_chunks?.map((chunk) => chunk.reference_id), ["rchunk-1"]);
      assert.equal("debug_enriched_content" in queryResult.used_chunks[0], false);
    } finally {
      globalThis.fetch = originalFetch;
      process.env.EMBEDDING_BASE_URL = previousEmbeddingUrl;
      process.env.EMBEDDING_API_KEY = previousEmbeddingKey;
      process.env.TEXT_BASE_URL = previousTextUrl;
      process.env.TEXT_API_KEY = previousTextKey;
      process.env.QDRANT_URL = previousQdrantUrl;
      process.env.FS_EXPLORER_SERVER_BASE_URL = previousServerBaseUrl;
      process.env.FS_EXPLORER_PORT = previousPort;
      storage.close();
    }
  });

  it("drops CER context before compressing the center retrieved chunk", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-cer-priority-"));
    const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
    storage.initialize();
    const blobStore = new LocalBlobStore(join(root, "object-store"));
    const rag = new TraditionalRagService(storage, blobStore);
    const previousServerBaseUrl = process.env.FS_EXPLORER_SERVER_BASE_URL;
    const previousPort = process.env.FS_EXPLORER_PORT;
    const previousBudget = process.env.TRADITIONAL_RAG_EVIDENCE_CHAR_BUDGET;
    delete process.env.FS_EXPLORER_SERVER_BASE_URL;
    process.env.FS_EXPLORER_PORT = "8000";
    process.env.TRADITIONAL_RAG_EVIDENCE_CHAR_BUDGET = "700";
    const corpusId = storage.getOrCreateCorpus(root);
    const docId = "doc-cer-priority";
    const previousContent = `Context before the diagram. ${"shareholder context ".repeat(120)}`;
    const centerContent = [
      "## 公司与实际控制人之间的产权及控制关系的方框图",
      "图片链接：![image](/api/assets/images/demo)",
      "曾毓群持股香港瑞华投资有限公司 100.00%。",
      "香港瑞华投资有限公司持股厦门瑞庭 45.00%。",
      "厦门瑞庭持股宁德时代 23.27%。",
    ].join("\n\n");
    const nextContent = `Context after the diagram. ${"financial statement context ".repeat(140)}`;

    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "cer-priority.pdf",
      absolutePath: join(root, "cer-priority.pdf"),
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 1,
      contentSha256: "hash-cer-priority",
      originalFilename: "cer-priority.pdf",
      objectKey: "documents/cer-priority.pdf",
      sourceObjectKey: "documents/cer-priority.pdf",
      pagesPrefix: "documents/cer-priority/pages",
      storageUri: "storage://cer-priority.pdf",
      contentType: "application/pdf",
      uploadStatus: "completed",
      pageCount: 3,
      parsedContentSha256: "parsed-cer-priority",
      parsedIsComplete: true,
      embeddingEnabled: true,
      hasEmbeddings: true,
      imageSemanticEnabled: true,
    });

    storage.replaceDocumentChunks(docId, [
      {
        id: "priority-dchunk-1",
        documentId: docId,
        ordinal: 1,
        referenceRetrievalChunkId: "priority-rchunk-1",
        pageNo: 1,
        documentIndex: 1,
        pageIndex: 1,
        blockType: "paragraph",
        bboxJson: "[0,0,1,1]",
        contentMd: previousContent,
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[1]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "priority-dchunk-2",
        documentId: docId,
        ordinal: 2,
        referenceRetrievalChunkId: "priority-rchunk-2",
        pageNo: 2,
        documentIndex: 2,
        pageIndex: 1,
        blockType: "picture",
        bboxJson: "[0,0,1,1]",
        contentMd: centerContent,
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[2]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "priority-dchunk-3",
        documentId: docId,
        ordinal: 3,
        referenceRetrievalChunkId: "priority-rchunk-3",
        pageNo: 3,
        documentIndex: 3,
        pageIndex: 1,
        blockType: "paragraph",
        bboxJson: "[0,0,1,1]",
        contentMd: nextContent,
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[3]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
    ]);

    storage.replaceRetrievalChunks(docId, [
      {
        id: "priority-rchunk-1",
        documentId: docId,
        ordinal: 1,
        contentMd: previousContent,
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["priority-dchunk-1"]),
        pageNosJson: JSON.stringify([1]),
        sourceLocator: "page-1",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
      {
        id: "priority-rchunk-2",
        documentId: docId,
        ordinal: 2,
        contentMd: centerContent,
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["priority-dchunk-2"]),
        pageNosJson: JSON.stringify([2]),
        sourceLocator: "page-2",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
      {
        id: "priority-rchunk-3",
        documentId: docId,
        ordinal: 3,
        contentMd: nextContent,
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["priority-dchunk-3"]),
        pageNosJson: JSON.stringify([3]),
        sourceLocator: "page-3",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
    ]);

    const originalFetch = globalThis.fetch;
    const previousEmbeddingUrl = process.env.EMBEDDING_BASE_URL;
    const previousEmbeddingKey = process.env.EMBEDDING_API_KEY;
    const previousQdrantUrl = process.env.QDRANT_URL;

    process.env.EMBEDDING_BASE_URL = "https://example.test/v1";
    process.env.EMBEDDING_API_KEY = "embed-test-key";
    process.env.QDRANT_URL = "https://qdrant.test";

    globalThis.fetch = (async (input) => {
      const url = String(input);
      if (url === "https://example.test/v1/embeddings") {
        return new Response(
          JSON.stringify({
            data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://qdrant.test/collections/doc_doc-cer-priority/points/search") {
        return new Response(
          JSON.stringify({
            result: [{ id: "qdrant-priority", score: 10, payload: { retrieval_unit_id: "priority-dchunk-2" } }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    try {
      const retrieveResult = await rag.retrieve({
        question: "方框图",
        mode: "semantic",
        documentIds: [docId],
      });

      const centerChunk = retrieveResult.retrieved_chunks.find((chunk) => chunk.reference_id === "priority-rchunk-2");
      assert.equal(centerChunk?.debug_cer_applied, true);
      assert.deepEqual(
        centerChunk?.debug_cer_context_refs?.map((item) => item.reference_id),
        ["priority-rchunk-3"],
      );
      assert.match(String(centerChunk?.debug_enriched_content || ""), /公司与实际控制人之间的产权及控制关系的方框图/);
      assert.match(String(centerChunk?.debug_enriched_content || ""), /厦门瑞庭持股宁德时代 23\.27%/);
      assert.match(String(centerChunk?.debug_enriched_content || ""), /!\[image\]\(\/api\/assets\/images\/demo\)/);
      assert.doesNotMatch(String(centerChunk?.debug_enriched_content || ""), /chunk\[1-p-1\]/i);
      assert.match(String(centerChunk?.debug_enriched_content || ""), /chunk\[1-n-1\] compressed excerpt:/i);
      assert.doesNotMatch(String(centerChunk?.debug_enriched_content || ""), /香港瑞华投$/m);
    } finally {
      globalThis.fetch = originalFetch;
      process.env.EMBEDDING_BASE_URL = previousEmbeddingUrl;
      process.env.EMBEDDING_API_KEY = previousEmbeddingKey;
      process.env.QDRANT_URL = previousQdrantUrl;
      process.env.FS_EXPLORER_SERVER_BASE_URL = previousServerBaseUrl;
      process.env.FS_EXPLORER_PORT = previousPort;
      if (previousBudget === undefined) {
        delete process.env.TRADITIONAL_RAG_EVIDENCE_CHAR_BUDGET;
      } else {
        process.env.TRADITIONAL_RAG_EVIDENCE_CHAR_BUDGET = previousBudget;
      }
      storage.close();
    }
  });

  it("keeps only the next CER chunk when the center chunk is a short heading block", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-cer-heading-"));
    const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
    storage.initialize();
    const blobStore = new LocalBlobStore(join(root, "object-store"));
    const rag = new TraditionalRagService(storage, blobStore);
    const previousServerBaseUrl = process.env.FS_EXPLORER_SERVER_BASE_URL;
    const previousPort = process.env.FS_EXPLORER_PORT;
    delete process.env.FS_EXPLORER_SERVER_BASE_URL;
    process.env.FS_EXPLORER_PORT = "8000";
    const corpusId = storage.getOrCreateCorpus(root);
    const docId = "doc-cer-heading";

    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "heading.pdf",
      absolutePath: join(root, "heading.pdf"),
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 1,
      contentSha256: "hash-heading",
      originalFilename: "heading.pdf",
      objectKey: "documents/heading.pdf",
      sourceObjectKey: "documents/heading.pdf",
      pagesPrefix: "documents/heading/pages",
      storageUri: "storage://heading.pdf",
      contentType: "application/pdf",
      uploadStatus: "completed",
      pageCount: 3,
      parsedContentSha256: "parsed-heading",
      parsedIsComplete: true,
      embeddingEnabled: true,
      hasEmbeddings: true,
      imageSemanticEnabled: false,
    });

    storage.replaceDocumentChunks(docId, [
      {
        id: "heading-dchunk-1",
        documentId: docId,
        ordinal: 1,
        referenceRetrievalChunkId: "heading-rchunk-1",
        pageNo: 108,
        documentIndex: 1,
        pageIndex: 1,
        blockType: "text",
        bboxJson: "[0,0,1,1]",
        contentMd: "母公司资产负债表尾部：流动资产、货币资金、应收账款。",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[108]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "heading-dchunk-2",
        documentId: docId,
        ordinal: 2,
        referenceRetrievalChunkId: "heading-rchunk-2",
        pageNo: 109,
        documentIndex: 2,
        pageIndex: 1,
        blockType: "text",
        bboxJson: "[0,0,1,1]",
        contentMd: "## **4** 、母公司利润表\n\n单位：元",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[109]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "heading-dchunk-3",
        documentId: docId,
        ordinal: 3,
        referenceRetrievalChunkId: "heading-rchunk-3",
        pageNo: 110,
        documentIndex: 3,
        pageIndex: 1,
        blockType: "text",
        bboxJson: "[0,0,1,1]",
        contentMd: "营业收入 1,000.00 营业成本 600.00 利润总额 200.00 净利润 180.00",
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[110]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
    ]);

    storage.replaceRetrievalChunks(docId, [
      {
        id: "heading-rchunk-1",
        documentId: docId,
        ordinal: 1,
        contentMd: "母公司资产负债表尾部：流动资产、货币资金、应收账款。",
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["heading-dchunk-1"]),
        pageNosJson: JSON.stringify([108]),
        sourceLocator: "page-108",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
      {
        id: "heading-rchunk-2",
        documentId: docId,
        ordinal: 2,
        contentMd: "## **4** 、母公司利润表\n\n单位：元",
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["heading-dchunk-2"]),
        pageNosJson: JSON.stringify([109]),
        sourceLocator: "page-109",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
      {
        id: "heading-rchunk-3",
        documentId: docId,
        ordinal: 3,
        contentMd: "营业收入 1,000.00 营业成本 600.00 利润总额 200.00 净利润 180.00",
        sizeClass: "normal",
        sourceDocumentChunkIdsJson: JSON.stringify(["heading-dchunk-3"]),
        pageNosJson: JSON.stringify([110]),
        sourceLocator: "page-110",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
    ]);

    const originalFetch = globalThis.fetch;
    const previousEmbeddingUrl = process.env.EMBEDDING_BASE_URL;
    const previousEmbeddingKey = process.env.EMBEDDING_API_KEY;
    const previousQdrantUrl = process.env.QDRANT_URL;

    process.env.EMBEDDING_BASE_URL = "https://example.test/v1";
    process.env.EMBEDDING_API_KEY = "embed-test-key";
    process.env.QDRANT_URL = "https://qdrant.test";

    globalThis.fetch = (async (input) => {
      const url = String(input);
      if (url === "https://example.test/v1/embeddings") {
        return new Response(
          JSON.stringify({
            data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://qdrant.test/collections/doc_doc-cer-heading/points/search") {
        return new Response(
          JSON.stringify({
            result: [{ id: "qdrant-heading", score: 10, payload: { retrieval_unit_id: "heading-dchunk-2" } }],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    try {
      const retrieveResult = await rag.retrieve({
        question: "请输出母公司利润表的内容",
        mode: "semantic",
        documentIds: [docId],
      });

      const centerChunk = retrieveResult.retrieved_chunks.find((chunk) => chunk.reference_id === "heading-rchunk-2");
      assert.equal(centerChunk?.debug_cer_applied, true);
      assert.deepEqual(
        centerChunk?.debug_cer_context_refs?.map((item) => item.reference_id),
        ["heading-rchunk-3"],
      );
      assert.match(String(centerChunk?.debug_enriched_content || ""), /chunk\[1\] compressed excerpt:/i);
      assert.match(String(centerChunk?.debug_enriched_content || ""), /chunk\[1-n-1\] compressed excerpt:/i);
      assert.doesNotMatch(String(centerChunk?.debug_enriched_content || ""), /chunk\[1-p-1\]/i);
    } finally {
      globalThis.fetch = originalFetch;
      process.env.EMBEDDING_BASE_URL = previousEmbeddingUrl;
      process.env.EMBEDDING_API_KEY = previousEmbeddingKey;
      process.env.QDRANT_URL = previousQdrantUrl;
      process.env.FS_EXPLORER_SERVER_BASE_URL = previousServerBaseUrl;
      process.env.FS_EXPLORER_PORT = previousPort;
      storage.close();
    }
  });

  it("expands Chinese keyword queries into subwords during retrieval", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-keyword-chinese-"));
    const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
    storage.initialize();
    const blobStore = new LocalBlobStore(join(root, "object-store"));
    const rag = new TraditionalRagService(storage, blobStore);
    const corpusId = storage.getOrCreateCorpus(root);
    const docId = "doc-keyword-chinese";
    const heading = "## 公司与实际控制人之间的产权及控制关系的方框图";

    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "keyword.pdf",
      absolutePath: join(root, "keyword.pdf"),
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 1,
      contentSha256: "hash-keyword-chinese",
      originalFilename: "keyword.pdf",
      objectKey: "documents/keyword.pdf",
      sourceObjectKey: "documents/keyword.pdf",
      pagesPrefix: "documents/keyword/pages",
      storageUri: "storage://keyword.pdf",
      contentType: "application/pdf",
      uploadStatus: "completed",
      pageCount: 1,
      parsedContentSha256: "parsed-keyword-chinese",
      parsedIsComplete: true,
      retrievalChunkingStrategy: "small_to_big",
    });
    storage.replaceDocumentChunks(docId, [
      {
        id: "keyword-dchunk-1",
        documentId: docId,
        ordinal: 0,
        referenceRetrievalChunkId: "keyword-rchunk-1",
        pageNo: 1,
        documentIndex: 0,
        pageIndex: 0,
        blockType: "section-header",
        bboxJson: "[0,0,1,1]",
        contentMd: heading,
        sizeClass: "small",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[1]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
    ]);
    storage.replaceRetrievalChunks(docId, [
      {
        id: "keyword-rchunk-1",
        documentId: docId,
        ordinal: 0,
        contentMd: heading,
        sizeClass: "small",
        sourceDocumentChunkIdsJson: JSON.stringify(["keyword-dchunk-1"]),
        pageNosJson: JSON.stringify([1]),
        sourceLocator: "page-1",
        bboxesJson: JSON.stringify([[0, 0, 1, 1]]),
      },
    ]);

    try {
      const result = await rag.retrieve({
        question: "输出方框图",
        mode: "keyword",
        documentIds: [docId],
      });
      assert.equal(result.retrieved_chunks.length > 0, true);
      assert.equal(result.retrieved_chunks[0]?.reference_id, "keyword-rchunk-1");
    } finally {
      storage.close();
    }
  });

  it("builds embeddings with controlled batch concurrency", async () => {
    const rag = new TraditionalRagService({} as never, {} as never);
    const originalFetch = globalThis.fetch;
    const previousEmbeddingUrl = process.env.EMBEDDING_BASE_URL;
    const previousEmbeddingKey = process.env.EMBEDDING_API_KEY;
    const previousBatchSize = process.env.EMBEDDING_BATCH_SIZE;
    const previousConcurrency = process.env.EMBEDDING_CONCURRENCY;
    const previousQdrantUrl = process.env.QDRANT_URL;

    process.env.EMBEDDING_BASE_URL = "https://example.test/v1";
    process.env.EMBEDDING_API_KEY = "embed-test-key";
    process.env.EMBEDDING_BATCH_SIZE = "2";
    process.env.EMBEDDING_CONCURRENCY = "2";
    process.env.QDRANT_URL = "https://qdrant.test";

    let activeEmbeddingRequests = 0;
    let maxConcurrentEmbeddingRequests = 0;
    globalThis.fetch = (async (input) => {
      const url = String(input);
      if (url === "https://example.test/v1/embeddings") {
        activeEmbeddingRequests += 1;
        maxConcurrentEmbeddingRequests = Math.max(maxConcurrentEmbeddingRequests, activeEmbeddingRequests);
        await new Promise((resolve) => setTimeout(resolve, 25));
        activeEmbeddingRequests -= 1;
        return new Response(
          JSON.stringify({
            data: [
              { embedding: [0.1, 0.2, 0.3], index: 0 },
              { embedding: [0.4, 0.5, 0.6], index: 1 },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://qdrant.test/collections/doc_doc-concurrency/points?wait=true") {
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://qdrant.test/collections/doc_doc-concurrency") {
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    try {
      const written = await rag.buildEmbeddingsForChunks(
        "doc-concurrency",
        Array.from({ length: 6 }, (_, index) => ({
          id: `unit-${index + 1}`,
          unitText: `embedding payload ${index + 1}`,
        })),
      );
      assert.equal(written, 6);
      assert.equal(maxConcurrentEmbeddingRequests, 2);
    } finally {
      globalThis.fetch = originalFetch;
      process.env.EMBEDDING_BASE_URL = previousEmbeddingUrl;
      process.env.EMBEDDING_API_KEY = previousEmbeddingKey;
      process.env.EMBEDDING_BATCH_SIZE = previousBatchSize;
      process.env.EMBEDDING_CONCURRENCY = previousConcurrency;
      process.env.QDRANT_URL = previousQdrantUrl;
    }
  });

  it("surfaces endpoint and batch details when embedding fetch fails", async () => {
    const rag = new TraditionalRagService({} as never, {} as never);
    const originalFetch = globalThis.fetch;
    const previousEmbeddingUrl = process.env.EMBEDDING_BASE_URL;
    const previousEmbeddingKey = process.env.EMBEDDING_API_KEY;
    const previousBatchSize = process.env.EMBEDDING_BATCH_SIZE;
    const previousConcurrency = process.env.EMBEDDING_CONCURRENCY;

    process.env.EMBEDDING_BASE_URL = "https://example.test/v1";
    process.env.EMBEDDING_API_KEY = "embed-test-key";
    process.env.EMBEDDING_BATCH_SIZE = "2";
    process.env.EMBEDDING_CONCURRENCY = "1";

    globalThis.fetch = (async (input) => {
      const url = String(input);
      if (url === "https://example.test/v1/embeddings") {
        throw new Error("fetch failed");
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    try {
      await assert.rejects(
        () =>
          rag.buildEmbeddingsForChunks("doc-failure", [
            { id: "unit-1", unitText: "a" },
            { id: "unit-2", unitText: "b" },
          ]),
        /Embedding request failed for batch 1\/1 \(2 texts, https:\/\/example\.test\/v1\/embeddings\): fetch failed/,
      );
    } finally {
      globalThis.fetch = originalFetch;
      process.env.EMBEDDING_BASE_URL = previousEmbeddingUrl;
      process.env.EMBEDDING_API_KEY = previousEmbeddingKey;
      process.env.EMBEDDING_BATCH_SIZE = previousBatchSize;
      process.env.EMBEDDING_CONCURRENCY = previousConcurrency;
    }
  });

  it("splits Qdrant point upserts into smaller batches", async () => {
    const rag = new TraditionalRagService({} as never, {} as never);
    const originalFetch = globalThis.fetch;
    const previousEmbeddingUrl = process.env.EMBEDDING_BASE_URL;
    const previousEmbeddingKey = process.env.EMBEDDING_API_KEY;
    const previousBatchSize = process.env.EMBEDDING_BATCH_SIZE;
    const previousConcurrency = process.env.EMBEDDING_CONCURRENCY;
    const previousQdrantUrl = process.env.QDRANT_URL;
    const previousQdrantUpsertBatchSize = process.env.QDRANT_UPSERT_BATCH_SIZE;

    process.env.EMBEDDING_BASE_URL = "https://example.test/v1";
    process.env.EMBEDDING_API_KEY = "embed-test-key";
    process.env.EMBEDDING_BATCH_SIZE = "3";
    process.env.EMBEDDING_CONCURRENCY = "1";
    process.env.QDRANT_URL = "https://qdrant.test";
    process.env.QDRANT_UPSERT_BATCH_SIZE = "2";

    const upsertBatchSizes: number[] = [];
    globalThis.fetch = (async (input, init) => {
      const url = String(input);
      if (url === "https://example.test/v1/embeddings") {
        const body = JSON.parse(String(init?.body ?? "{}"));
        return new Response(
          JSON.stringify({
            data: (body.input || []).map((_: unknown, index: number) => ({
              embedding: [index + 0.1, index + 0.2, index + 0.3],
              index,
            })),
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://qdrant.test/collections/doc_doc-upsert-batches") {
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://qdrant.test/collections/doc_doc-upsert-batches/points?wait=true") {
        const body = JSON.parse(String(init?.body ?? "{}"));
        upsertBatchSizes.push((body.points || []).length);
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    try {
      const written = await rag.buildEmbeddingsForChunks(
        "doc-upsert-batches",
        Array.from({ length: 5 }, (_, index) => ({
          id: `unit-${index + 1}`,
          unitText: `payload ${index + 1}`,
        })),
      );
      assert.equal(written, 5);
      assert.deepEqual(upsertBatchSizes, [2, 2, 1]);
    } finally {
      globalThis.fetch = originalFetch;
      process.env.EMBEDDING_BASE_URL = previousEmbeddingUrl;
      process.env.EMBEDDING_API_KEY = previousEmbeddingKey;
      process.env.EMBEDDING_BATCH_SIZE = previousBatchSize;
      process.env.EMBEDDING_CONCURRENCY = previousConcurrency;
      process.env.QDRANT_URL = previousQdrantUrl;
      process.env.QDRANT_UPSERT_BATCH_SIZE = previousQdrantUpsertBatchSize;
    }
  });

  it("includes previous non-picture context when building image semantics for small_to_big only", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-image-context-"));
    const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
    storage.initialize();
    const blobStore = new LocalBlobStore(join(root, "object-store"));
    const rag = new TraditionalRagService(storage, blobStore);
    const corpusId = storage.getOrCreateCorpus(root);
    const originalFetch = globalThis.fetch;
    const previousVisionUrl = process.env.VISION_BASE_URL;
    const previousVisionKey = process.env.VISION_API_KEY;
    const previousMaxPerPage = process.env.VISION_MAX_SEMANTIC_PER_PAGE;
    const previousMaxPerDoc = process.env.VISION_MAX_SEMANTIC_PER_DOC;
    process.env.VISION_BASE_URL = "https://vision.test/v1";
    process.env.VISION_API_KEY = "vision-test-key";
    process.env.VISION_MAX_SEMANTIC_PER_PAGE = "5";
    process.env.VISION_MAX_SEMANTIC_PER_DOC = "5";

    const imageBytes = Buffer.from("fake-image");
    Object.defineProperty(rag, "pythonAssets", {
      configurable: true,
      writable: true,
      value: {
        async extractPdfImages() {
          return [
            {
              image_hash: "img-context",
              page_no: 1,
              image_index: 0,
              mime_type: "image/png",
              width: 100,
              height: 100,
              bytes_base64: imageBytes.toString("base64"),
              byte_size: imageBytes.length,
            },
          ];
        },
        async inspectImage(bytesBase64: string) {
          return {
            supported: true,
            has_text: true,
            interference_score: 0,
            compressed_bytes_base64: bytesBase64,
            compressed_byte_size: Buffer.from(bytesBase64, "base64").length,
            output_mime_type: "image/png",
          };
        },
      },
    });

    const promptBodies: string[] = [];
    globalThis.fetch = (async (input, init) => {
      const url = String(input);
      if (url === "https://vision.test/v1/chat/completions") {
        const body = JSON.parse(String(init?.body ?? "{}"));
        promptBodies.push(String(body.messages?.[1]?.content?.[0]?.text ?? ""));
        return new Response(
          JSON.stringify({
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    recognizable: true,
                    image_kind: "ownership diagram",
                    contains_text: true,
                    retrieval_summary: "曾毓群持股香港瑞华投资有限公司 100.00%。",
                    detail_markdown: "## 逐项关系\n- 曾毓群持股香港瑞华投资有限公司 100.00%。",
                    visible_text: "曾毓群 100.00%",
                    keywords: ["方框图", "控制关系"],
                    entities: ["曾毓群"],
                    qa_hints: ["问图中控制关系"],
                    detail_truncated: false,
                    drop_reason: null,
                  }),
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    const parsedDocument = {
      parser_name: "test",
      parser_version: "1",
      units: [
        {
          unit_no: 1,
          markdown: "## 公司与实际控制人之间的产权及控制关系的方框图",
          content_hash: "page-1",
          heading: null,
          source_locator: "page-1",
          images: [{ image_hash: "img-context", page_no: 1, image_index: 0, bbox: [0, 0, 1, 1] }],
          blocks: [
            {
              index: 0,
              block_type: "section-header",
              bbox: [0, 0, 1, 0.2],
              markdown: "## 公司与实际控制人之间的产权及控制关系的方框图",
              char_count: 23,
              image_hash: null,
              source_image_index: null,
            },
            {
              index: 1,
              block_type: "picture",
              bbox: [0, 0.2, 1, 1],
              markdown: "![image-placeholder](context)",
              char_count: 1,
              image_hash: "img-context",
              source_image_index: 0,
            },
          ],
        },
      ],
    };

    try {
      storage.upsertDocumentStub({
        id: "doc-image-context-small",
        corpusId,
        relativePath: "context.pdf",
        absolutePath: join(root, "context.pdf"),
        content: "",
        metadataJson: "{}",
        fileMtime: 1,
        fileSize: 1,
        contentSha256: "hash-image-context-small",
        originalFilename: "context.pdf",
        retrievalChunkingStrategy: "small_to_big",
      });
      storage.upsertDocumentStub({
        id: "doc-image-context-fixed",
        corpusId,
        relativePath: "context-fixed.pdf",
        absolutePath: join(root, "context-fixed.pdf"),
        content: "",
        metadataJson: "{}",
        fileMtime: 1,
        fileSize: 1,
        contentSha256: "hash-image-context-fixed",
        originalFilename: "context-fixed.pdf",
        retrievalChunkingStrategy: "fixed",
      });
      await rag.renderImages({
        documentId: "doc-image-context-small",
        filePath: join(root, "context.pdf"),
        originalFilename: "context.pdf",
        parsedDocument,
        enableImageSemantic: true,
        retrievalChunkingStrategy: "small_to_big",
      });
      Object.defineProperty(rag, "pythonAssets", {
        configurable: true,
        writable: true,
        value: {
          async extractPdfImages() {
            return [
              {
                image_hash: "img-context-fixed",
                page_no: 1,
                image_index: 0,
                mime_type: "image/png",
                width: 100,
                height: 100,
                bytes_base64: imageBytes.toString("base64"),
                byte_size: imageBytes.length,
              },
            ];
          },
          async inspectImage(bytesBase64: string) {
            return {
              supported: true,
              has_text: true,
              interference_score: 0,
              compressed_bytes_base64: bytesBase64,
              compressed_byte_size: Buffer.from(bytesBase64, "base64").length,
              output_mime_type: "image/png",
            };
          },
        },
      });
      await rag.renderImages({
        documentId: "doc-image-context-fixed",
        filePath: join(root, "context-fixed.pdf"),
        originalFilename: "context-fixed.pdf",
        parsedDocument: {
          ...parsedDocument,
          units: [
            {
              ...parsedDocument.units[0]!,
              images: [{ image_hash: "img-context-fixed", page_no: 1, image_index: 0, bbox: [0, 0, 1, 1] }],
              blocks: [
                parsedDocument.units[0]!.blocks[0]!,
                { ...parsedDocument.units[0]!.blocks[1]!, image_hash: "img-context-fixed" },
              ],
            },
          ],
        },
        enableImageSemantic: true,
        retrievalChunkingStrategy: "fixed",
      });

      assert.equal(promptBodies.length, 2);
      assert.match(promptBodies[0] ?? "", /邻近上下文/);
      assert.match(promptBodies[0] ?? "", /公司与实际控制人之间的产权及控制关系的方框图/);
      assert.doesNotMatch(promptBodies[1] ?? "", /邻近上下文/);
    } finally {
      globalThis.fetch = originalFetch;
      process.env.VISION_BASE_URL = previousVisionUrl;
      process.env.VISION_API_KEY = previousVisionKey;
      if (previousMaxPerPage === undefined) {
        delete process.env.VISION_MAX_SEMANTIC_PER_PAGE;
      } else {
        process.env.VISION_MAX_SEMANTIC_PER_PAGE = previousMaxPerPage;
      }
      if (previousMaxPerDoc === undefined) {
        delete process.env.VISION_MAX_SEMANTIC_PER_DOC;
      } else {
        process.env.VISION_MAX_SEMANTIC_PER_DOC = previousMaxPerDoc;
      }
      storage.close();
    }
  });

  it("adds previous chunk context to picture embeddings without exceeding the embedding-3 budget", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-picture-embedding-budget-"));
    const storage = new SqliteStorage({ dbPath: join(root, "storage.sqlite") });
    storage.initialize();
    const blobStore = new LocalBlobStore(join(root, "object-store"));
    const rag = new TraditionalRagService(storage, blobStore);
    const corpusId = storage.getOrCreateCorpus(root);
    const docId = "doc-picture-embedding-budget";
    storage.upsertDocumentStub({
      id: docId,
      corpusId,
      relativePath: "picture-embedding.pdf",
      absolutePath: join(root, "picture-embedding.pdf"),
      content: "",
      metadataJson: "{}",
      fileMtime: 1,
      fileSize: 1,
      contentSha256: "hash-picture-embedding-budget",
      originalFilename: "picture-embedding.pdf",
      retrievalChunkingStrategy: "small_to_big",
      embeddingEnabled: true,
      hasEmbeddings: false,
    });
    const previousChunkText = `## 公司与实际控制人之间的产权及控制关系的方框图\n${"控制关系说明 ".repeat(40)}`;
    const pictureChunkText = [
      "图片链接：",
      "![image](/api/assets/images/demo)",
      "",
      "曾毓群持股香港瑞华投资有限公司 100.00%。",
      "香港瑞华投资有限公司持股厦门瑞庭 45.00%。",
      "厦门瑞庭持股宁德时代 23.27%。",
    ].join("\n");
    storage.replaceDocumentChunks(docId, [
      {
        id: "embed-dchunk-1",
        documentId: docId,
        ordinal: 0,
        referenceRetrievalChunkId: "embed-rchunk-1",
        pageNo: 1,
        documentIndex: 0,
        pageIndex: 0,
        blockType: "section-header",
        bboxJson: "[0,0,1,1]",
        contentMd: previousChunkText,
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[1]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
      {
        id: "embed-dchunk-2",
        documentId: docId,
        ordinal: 1,
        referenceRetrievalChunkId: "embed-rchunk-2",
        pageNo: 1,
        documentIndex: 1,
        pageIndex: 1,
        blockType: "picture",
        bboxJson: "[0,0,1,1]",
        contentMd: pictureChunkText,
        sizeClass: "normal",
        summaryText: null,
        isSplitFromOversized: false,
        splitIndex: 0,
        splitCount: 1,
        mergedPageNosJson: "[1]",
        mergedBboxesJson: "[[0,0,1,1]]",
      },
    ]);

    const originalFetch = globalThis.fetch;
    const previousEmbeddingModel = process.env.EMBEDDING_MODEL_NAME;
    const previousEmbeddingUrl = process.env.EMBEDDING_BASE_URL;
    const previousEmbeddingKey = process.env.EMBEDDING_API_KEY;
    const previousTokenBudget = process.env.EMBEDDING_MODEL_MAX_TOKENS;
    const previousRatio = process.env.EMBEDDING_MODEL_CHAR_BUDGET_RATIO;
    const previousQdrantUrl = process.env.QDRANT_URL;

    process.env.EMBEDDING_MODEL_NAME = "embedding-3";
    process.env.EMBEDDING_BASE_URL = "https://example.test/v1";
    process.env.EMBEDDING_API_KEY = "embed-test-key";
    process.env.EMBEDDING_MODEL_MAX_TOKENS = "40";
    process.env.EMBEDDING_MODEL_CHAR_BUDGET_RATIO = "2";
    process.env.QDRANT_URL = "https://qdrant.test";

    const embeddingInputs: string[] = [];
    globalThis.fetch = (async (input, init) => {
      const url = String(input);
      if (url === "https://example.test/v1/embeddings") {
        const body = JSON.parse(String(init?.body ?? "{}"));
        embeddingInputs.push(...(body.input ?? []));
        return new Response(
          JSON.stringify({
            data: (body.input || []).map((_: unknown, index: number) => ({
              embedding: [index + 0.1, index + 0.2, index + 0.3],
              index,
            })),
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url === "https://qdrant.test/collections/doc_doc-picture-embedding-budget") {
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://qdrant.test/collections/doc_doc-picture-embedding-budget/points?wait=true") {
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch ${url}`);
    }) as typeof globalThis.fetch;

    try {
      const written = await rag.buildEmbeddingsForChunks(docId, [
        { id: "embed-dchunk-1", unitText: previousChunkText },
        { id: "embed-dchunk-2", unitText: pictureChunkText },
      ]);
      assert.equal(written, 2);
      assert.equal(embeddingInputs.length, 2);
      assert.equal(embeddingInputs[0], previousChunkText);
      assert.match(embeddingInputs[1] ?? "", /!\[image\]\(\/api\/assets\/images\/demo\)/);
      assert.match(embeddingInputs[1] ?? "", /曾毓群持股香港瑞华投资有限公司 100\.00%/);
      assert.match(embeddingInputs[1] ?? "", /公司与实际控制人之间的产权及控制关系的方框图/);
      assert.equal((embeddingInputs[1] ?? "").length < [pictureChunkText, previousChunkText].join("\n\n").length, true);
      assert.doesNotMatch(embeddingInputs[1] ?? "", /鎺у埗鍏崇郴璇存槑.{0,20}鎺у埗鍏崇郴璇存槑/s);
    } finally {
      globalThis.fetch = originalFetch;
      process.env.EMBEDDING_MODEL_NAME = previousEmbeddingModel;
      process.env.EMBEDDING_BASE_URL = previousEmbeddingUrl;
      process.env.EMBEDDING_API_KEY = previousEmbeddingKey;
      process.env.EMBEDDING_MODEL_MAX_TOKENS = previousTokenBudget;
      process.env.EMBEDDING_MODEL_CHAR_BUDGET_RATIO = previousRatio;
      process.env.QDRANT_URL = previousQdrantUrl;
      storage.close();
    }
  });

  it("removes blank seam lines when merging a table continued across pages", () => {
    const rag = new TraditionalRagService({} as never, {} as never);
    const documentChunks = rag.createDocumentChunks({
      documentId: "doc-table",
      imageRenderMap: new Map(),
      parsedDocument: {
        parser_name: "test",
        parser_version: "1",
        units: [
          {
            unit_no: 1,
            markdown: "| Col A | Col B |\n| --- | --- |\n| A1 | B1 |",
            content_hash: "hash-page-1",
            heading: null,
            source_locator: "page-1",
            images: [],
            blocks: [
              {
                index: 0,
                block_type: "table",
                bbox: [0, 0, 100, 100],
                markdown: "| Col A | Col B |\n| --- | --- |\n| A1 | B1 |\n",
                char_count: 44,
                image_hash: null,
                source_image_index: null,
              },
            ],
          },
          {
            unit_no: 2,
            markdown: "| A2 | B2 |\n| A3 | B3 |",
            content_hash: "hash-page-2",
            heading: null,
            source_locator: "page-2",
            images: [],
            blocks: [
              {
                index: 0,
                block_type: "table",
                bbox: [0, 0, 100, 100],
                markdown: "\n| A2 | B2 |\n| A3 | B3 |",
                char_count: 24,
                image_hash: null,
                source_image_index: null,
              },
            ],
          },
        ],
      },
    });

    assert.equal(documentChunks.length, 1);
    assert.equal(
      documentChunks[0]?.contentMd,
      "| Col A | Col B |\n| --- | --- |\n| A1 | B1 |\n| A2 | B2 |\n| A3 | B3 |",
    );
    assert.doesNotMatch(String(documentChunks[0]?.contentMd || ""), /\|\s*A1\s*\|\s*B1\s*\|\n\n\|/);
  });

  it("splits long image semantic markdown in fixed chunk mode without exceeding the configured limit", () => {
    const rag = new TraditionalRagService({} as never, {} as never);
    const longImageSemantic = [
      "![image](/api/assets/images/demo)",
      "## 图片摘要",
      "这是一张流程图，描述采购申请、部门审批、财务复核和归档。",
      "## 关键结构与关系",
      "- 发起人提交申请",
      "- 部门经理审批",
      "- 财务复核预算",
      "- 系统归档记录",
      "## 详细说明",
      "节点说明 ".repeat(60),
    ].join("\n\n");
    const sourceChunks = rag.createDocumentChunks({
      documentId: "doc-picture-fixed",
      imageRenderMap: new Map([["img-fixed-1", { dropped: false, markdown: longImageSemantic }]]),
      parsedDocument: {
        parser_name: "test",
        parser_version: "1",
        units: [
          {
            unit_no: 1,
            markdown: "placeholder",
            content_hash: "hash-picture-fixed",
            heading: "Image heavy page",
            source_locator: "page-1",
            images: [],
            blocks: [
              {
                index: 0,
                block_type: "picture",
                bbox: [0, 0, 100, 100],
                markdown: "placeholder",
                char_count: longImageSemantic.length,
                image_hash: "img-fixed-1",
                source_image_index: 1,
              },
            ],
          },
        ],
      },
    });

    const fixedChunks = rag.createFixedRetrievalChunks({
      documentId: "doc-picture-fixed",
      sourceChunks,
      fixedChunkChars: 120,
    });

    assert.equal(fixedChunks.length > 1, true);
    assert.equal(fixedChunks.every((chunk) => chunk.contentMd.length <= 120), true);
    assert.equal(
      fixedChunks
        .filter((chunk) => chunk.contentMd.includes("![image]"))
        .every((chunk) => chunk.contentMd.includes("![image](/api/assets/images/demo)")),
      true,
    );
    assert.equal(
      fixedChunks.every((chunk) => !/!\[image\]\([^)]*$/.test(chunk.contentMd)),
      true,
    );
    assert.match(fixedChunks[0]?.contentMd ?? "", /图片摘要|!\[image\]/);
  });

  it("supports exploration session create/get/reply and keeps unsupported index routes explicit", async () => {
    const { app, storage } = await createAppFixture();
    const upload = multipartBody({
      filename: "alpha.pdf",
      content: "%PDF fake",
      fields: {
        enable_embedding: "false",
        enable_image_semantic: "false",
      },
    });
    const uploaded = await app.inject({
      method: "POST",
      url: "/api/documents",
      headers: { "content-type": upload.contentType },
      payload: upload.body,
    });
    const docId = uploaded.json().document.id as string;
    await waitForDocumentDetail(app, docId, (document) => document.upload_status === "completed");
    const collection = storage.createCollection("Answers");
    storage.attachDocumentsToCollection(collection.id, [docId]);

    const started = await app.inject({
      method: "POST",
      url: "/api/explore/sessions",
      payload: {
        task: "What is the purchase price?",
        document_ids: [],
        collection_ids: [collection.id],
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
    assert.deepEqual(snapshot.json().collection_ids, [collection.id]);
    assert.deepEqual(snapshot.json().document_ids, [docId]);

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

    const upload = multipartBody({
      filename: "alpha.pdf",
      content: "%PDF fake",
      fields: {
        enable_embedding: "false",
        enable_image_semantic: "false",
      },
    });
    const uploaded = await fetch(`${baseUrl}/api/documents`, {
      method: "POST",
      headers: { "content-type": upload.contentType },
      body: upload.body,
    });
    assert.equal(uploaded.status, 202);
    const uploadedPayload = (await uploaded.json()) as { document: { id: string } };
    await waitForDocumentDetail(app, uploadedPayload.document.id, (document) => document.upload_status === "completed");

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

import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import type { ActionModel, ActionModelRequest } from "../src/agent/agent.js";
import {
  ensureLibraryCorpus,
  ExplorationWorkflowService,
  LocalBlobStore,
  SqliteStorage,
} from "../src/index.js";

class SequenceModel implements ActionModel {
  readonly requests: ActionModelRequest[] = [];

  constructor(private readonly actions: unknown[]) {}

  async generateAction(request: ActionModelRequest): Promise<unknown> {
    this.requests.push(request);
    const next = this.actions.shift();
    if (!next) {
      return {
        action: { final_result: "No more scripted actions." },
        reason: "Stop.",
      };
    }
    return next;
  }
}

class StreamingSequenceModel extends SequenceModel {
  async *streamFinalAnswer(request: { draftAnswer: string }): AsyncIterable<string> {
    const text = request.draftAnswer || "No final answer.";
    yield text.slice(0, Math.ceil(text.length / 2));
    yield text.slice(Math.ceil(text.length / 2));
  }
}

function toolCall(toolName: string, toolInput: Record<string, unknown>) {
  return {
    action: {
      tool_name: toolName,
      tool_input: Object.entries(toolInput).map(([parameter_name, parameter_value]) => ({
        parameter_name,
        parameter_value,
      })),
    },
    reason: `Run ${toolName}`,
  };
}

function stop(finalResult: string) {
  return {
    action: { final_result: finalResult },
    reason: "Enough evidence.",
  };
}

function askHuman(question: string) {
  return {
    action: { question },
    reason: "Need clarification.",
  };
}

async function createWorkflowFixture() {
  const root = await mkdtemp(join(tmpdir(), "afs-workflow-"));
  const dbPath = join(root, "storage.sqlite");
  const objectRoot = join(root, "object-store");
  const storage = new SqliteStorage({ dbPath });
  storage.initialize();
  const blobStore = new LocalBlobStore(objectRoot);
  const corpusId = ensureLibraryCorpus(storage);
  const pagesPrefix = "library/documents/alpha.pdf/pages";
  const pagesDir = join(objectRoot, pagesPrefix);
  await mkdir(pagesDir, { recursive: true });
  const pagePath = join(pagesDir, "page-0001.md");
  const pageTwoPath = join(pagesDir, "page-0002.md");
  await writeFile(
    pagePath,
    [
      "---",
      "document_id: doc-alpha",
      "page_no: 1",
      "original_filename: alpha.pdf",
      "heading: Purchase Price",
      "source_locator: page-1",
      "---",
      "",
      "The purchase price is $45,000,000.",
    ].join("\n"),
  );
  await writeFile(
    pageTwoPath,
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
  storage.upsertDocumentStub({
    id: "doc-alpha",
    corpusId,
    relativePath: "alpha.pdf",
    absolutePath: join(root, "alpha.pdf"),
    content: "",
    metadataJson: "{}",
    fileMtime: 1,
    fileSize: 123,
    contentSha256: "sha-alpha",
    originalFilename: "alpha.pdf",
    pagesPrefix,
    pageCount: 2,
    uploadStatus: "uploaded",
    parsedContentSha256: "sha-alpha",
    parsedIsComplete: true,
  });
  storage.syncDocumentPages("doc-alpha", [
    {
      documentId: "doc-alpha",
      pageNo: 1,
      objectKey: `${pagesPrefix}/page-0001.md`,
      contentHash: "hash-page-1",
      charCount: 39,
      isSyntheticPage: false,
      heading: "Purchase Price",
      sourceLocator: "page-1",
      leadingBlockMarkdown: "The purchase price is $45,000,000.",
      trailingBlockMarkdown: "The purchase price is $45,000,000.",
    },
    {
      documentId: "doc-alpha",
      pageNo: 2,
      objectKey: `${pagesPrefix}/page-0002.md`,
      contentHash: "hash-page-2",
      charCount: 22,
      isSyntheticPage: false,
      heading: "Closing",
      sourceLocator: "page-2",
      leadingBlockMarkdown: "Closing follows later.",
      trailingBlockMarkdown: "Closing follows later.",
    },
  ]);

  return {
    root,
    storage,
    blobStore,
    pagesDir,
    pagePath,
    pageTwoPath,
  };
}

describe("exploration workflow service", () => {
  it("runs the legacy page-first workflow to completion", async () => {
    const fixture = await createWorkflowFixture();
    const model = new SequenceModel([
      toolCall("glob", { directory: fixture.pagesDir, pattern: "page-*.md" }),
      toolCall("grep", { file_path: fixture.pagesDir, pattern: "purchase price" }),
      toolCall("read", { file_path: fixture.pagePath }),
      stop("The purchase price is $45,000,000. [Source: alpha.pdf, page 1]"),
    ]);
    const service = new ExplorationWorkflowService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      model,
      skillsRoot: join(process.cwd(), "skills"),
      rootDirectory: fixture.root,
    });

    const started = await service.startSession({
      task: "What is the purchase price?",
      documentIds: ["doc-alpha"],
    });
    await service.waitForSession(started.session_id);

    const session = service.getSession(started.session_id)!;
    assert.equal(session.status, "completed");
    assert.equal(session.finalResult, "The purchase price is $45,000,000. [Source: alpha.pdf, page 1]");
    assert.deepEqual(session.history.map((event) => event.type), [
      "start",
      "context_scope_updated",
      "tool_call",
      "page_scope_resolved",
      "tool_call",
      "candidate_pages_found",
      "context_scope_updated",
      "tool_call",
      "pages_read",
      "context_scope_updated",
      "answer_started",
      "answer_delta",
      "answer_done",
      "complete",
    ]);
    assert.match(model.requests[0].messages.map((message) => message.content).join("\n"), /Start with `glob\(directory="scope"\)`/);
    assert.match(model.requests[0].systemPrompt, /page_boundary_context/);
    assert.match(model.requests[1].messages.map((message) => message.content).join("\n"), /Given the tool result/);

    const completeEvent = session.history.at(-1)!;
    const trace = completeEvent.data.trace as Record<string, unknown>;
    assert.deepEqual(trace.cited_sources, ["alpha.pdf"]);
    assert.deepEqual(
      ((completeEvent.data.stats as Record<string, unknown>).candidate_pages as Array<Record<string, unknown>>)
        .at(0)
        ?.document_id,
      "doc-alpha",
    );
    assert.ok((trace.step_path as string[]).some((item) => item.includes("tool:read")));
    assert.equal(
      (session.contextStateSnapshot.context_scope as Record<string, unknown>).active_document_id,
      "doc-alpha",
    );

    fixture.storage.close();
  });

  it("publishes page boundary events when the agent requests boundary context", async () => {
    const fixture = await createWorkflowFixture();
    const model = new SequenceModel([
      toolCall("read", { file_path: fixture.pagePath }),
      toolCall("page_boundary_context", { file_path: fixture.pagePath, direction: "next" }),
      stop("The purchase price is $45,000,000. [Source: alpha.pdf, page 1]"),
    ]);
    const service = new ExplorationWorkflowService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      model,
      skillsRoot: join(process.cwd(), "skills"),
      rootDirectory: fixture.root,
    });

    const started = await service.startSession({
      task: "What is the purchase price?",
      documentIds: ["doc-alpha"],
    });
    await service.waitForSession(started.session_id);

    const session = service.getSession(started.session_id)!;
    assert.equal(session.status, "completed");
    assert.ok(session.history.some((event) => event.type === "page_boundary_loaded"));
    const completeEvent = session.history.at(-1)!;
    const trace = completeEvent.data.trace as Record<string, unknown>;
    assert.equal((trace.page_boundary_loads as Array<Record<string, unknown>>).length, 1);
    assert.equal((trace.page_boundary_loads as Array<Record<string, unknown>>)[0]?.direction, "next");
    assert.equal((trace.page_boundary_loads as Array<Record<string, unknown>>)[0]?.mode, "single");
    assert.equal((trace.page_boundary_loads as Array<Record<string, unknown>>)[0]?.anchor_page, 1);

    fixture.storage.close();
  });

  it("pauses for ask_human and resumes with a human reply", async () => {
    const fixture = await createWorkflowFixture();
    const model = new SequenceModel([
      askHuman("Which date range should I use?"),
      stop("Using the provided date range, no additional evidence was needed."),
    ]);
    const service = new ExplorationWorkflowService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      model,
      skillsRoot: join(process.cwd(), "skills"),
      rootDirectory: fixture.root,
    });

    const started = await service.startSession({
      task: "Summarize the document for a date range.",
      documentIds: ["doc-alpha"],
    });
    await service.waitForSession(started.session_id);

    const paused = service.getSession(started.session_id)!;
    assert.equal(paused.status, "awaiting_human");
    assert.equal(paused.pendingQuestion, "Which date range should I use?");
    assert.equal(paused.history.at(-1)!.type, "ask_human");

    await service.replyToSession({
      sessionId: started.session_id,
      response: "Use calendar year 2025.",
    });
    await service.waitForSession(started.session_id);

    const completed = service.getSession(started.session_id)!;
    assert.equal(completed.status, "completed");
    assert.match(
      model.requests.at(-1)!.messages.map((message) => message.content).join("\n"),
      /Human response to your question: Use calendar year 2025\./,
    );

    fixture.storage.close();
  });

  it("streams the final answer before the complete event when the model supports it", async () => {
    const fixture = await createWorkflowFixture();
    const model = new StreamingSequenceModel([
      toolCall("glob", { directory: fixture.pagesDir, pattern: "page-*.md" }),
      toolCall("read", { file_path: fixture.pagePath }),
      stop("The purchase price is $45,000,000. [Source: alpha.pdf, page 1]"),
    ]);
    const service = new ExplorationWorkflowService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      model,
      skillsRoot: join(process.cwd(), "skills"),
      rootDirectory: fixture.root,
    });

    const started = await service.startSession({
      task: "What is the purchase price?",
      documentIds: ["doc-alpha"],
    });
    await service.waitForSession(started.session_id);

    const session = service.getSession(started.session_id)!;
    assert.equal(session.status, "completed");
    assert.deepEqual(session.history.slice(-5).map((event) => event.type), [
      "answer_started",
      "answer_delta",
      "answer_delta",
      "answer_done",
      "complete",
    ]);
    assert.equal(
      session.finalResult,
      "The purchase price is $45,000,000. [Source: alpha.pdf, page 1]",
    );

    fixture.storage.close();
  });

  it("runs forced batch mode and publishes batch cache events", async () => {
    const fixture = await createWorkflowFixture();
    const model = new SequenceModel([
      {
        selected_documents: [
          {
            document_id: "doc-alpha",
            reason: "The retrieval hit contains the purchase price.",
          },
        ],
      },
      stop("Document answer: the purchase price is $45,000,000. [Source: alpha.pdf, page 1]"),
      stop("Final synthesis: the purchase price is $45,000,000. [Source: alpha.pdf, page 1]"),
    ]);
    const service = new ExplorationWorkflowService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      model,
      skillsRoot: join(process.cwd(), "skills"),
      rootDirectory: fixture.root,
    });

    const started = await service.startSession({
      task: "What is the purchase price?",
      documentIds: ["doc-alpha"],
      batchMode: "force",
      batchSize: 1,
      batchThreshold: 10,
    });
    await service.waitForSession(started.session_id);

    const session = service.getSession(started.session_id)!;
    const eventTypes = session.history.map((event) => event.type);
    assert.equal(session.status, "completed");
    assert.ok(eventTypes.includes("candidate_documents_found"));
    assert.ok(eventTypes.includes("document_agent_started"));
    assert.ok(eventTypes.includes("document_answer_done"));
    assert.ok(eventTypes.includes("document_agent_completed"));
    assert.ok(eventTypes.includes("final_synthesis_started"));
    assert.ok(eventTypes.includes("final_synthesis_done"));
    assert.ok(eventTypes.includes("batch_started"));
    assert.ok(eventTypes.includes("batch_answer_done"));
    assert.ok(eventTypes.includes("cumulative_answer_updated"));
    assert.ok(eventTypes.includes("batch_completed"));
    assert.ok(eventTypes.includes("batch_context_released"));
    assert.equal(session.documentSummaries.length, 1);
    assert.equal(session.batchSummaries.length, 1);
    assert.match(session.cumulativeAnswer ?? "", /Document answer/);
    assert.match(session.finalResult ?? "", /Final synthesis/);
    assert.equal(session.snapshot().batch_mode, "force");
    assert.equal(session.snapshot().parallel_document_limit, 3);

    fixture.storage.close();
  });

  it("does not run document agents for invalid coordinator selections", async () => {
    const fixture = await createWorkflowFixture();
    const model = new SequenceModel([
      {
        selected_documents: [
          {
            document_id: "doc-missing",
            reason: "This id is outside the retrieval candidate set.",
          },
        ],
      },
      stop("No selected candidate document supports the answer."),
    ]);
    const service = new ExplorationWorkflowService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      model,
      skillsRoot: join(process.cwd(), "skills"),
      rootDirectory: fixture.root,
    });

    const started = await service.startSession({
      task: "What is the purchase price?",
      documentIds: ["doc-alpha"],
      batchMode: "force",
      batchSize: 1,
      batchThreshold: 10,
    });
    await service.waitForSession(started.session_id);

    const session = service.getSession(started.session_id)!;
    const eventTypes = session.history.map((event) => event.type);
    assert.equal(session.status, "completed");
    assert.ok(eventTypes.includes("candidate_documents_found"));
    assert.equal(eventTypes.includes("document_agent_started"), false);
    assert.ok(eventTypes.includes("final_synthesis_done"));
    assert.equal(session.documentSummaries.length, 0);
    assert.equal(session.batchSummaries.length, 0);
    assert.match(session.finalResult ?? "", /No selected candidate document/);

    fixture.storage.close();
  });
});

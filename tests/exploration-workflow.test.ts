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

class JsonEnvelopeStreamingModel extends SequenceModel {
  async *streamFinalAnswer(): AsyncIterable<string> {
    yield JSON.stringify({
      action: {
        final_result: "## Clean Answer\n\nThis should render as markdown.",
        reason: "Wrapped incorrectly by the model.",
      },
    });
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
    pageCount: 1,
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
    },
  ]);

  return {
    root,
    storage,
    blobStore,
    pagesDir,
    pagePath,
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
    assert.match(model.requests[0].messages.map((message) => message.content).join("\n"), /Start with `glob`/);
    const toolResultPrompt = model.requests[1].messages.map((message) => message.content).join("\n");
    assert.match(toolResultPrompt, /Given the tool result/);
    assert.match(toolResultPrompt, /previous page, next page, or both/);
    assert.match(toolResultPrompt, /read only those needed adjacent pages/);
    assert.ok(!toolResultPrompt.includes("treat the previous and next page as candidate pages"));

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

  it("unwraps JSON-wrapped final answers before completing the session", async () => {
    const fixture = await createWorkflowFixture();
    const model = new JsonEnvelopeStreamingModel([
      toolCall("glob", { directory: fixture.pagesDir, pattern: "page-*.md" }),
      toolCall("read", { file_path: fixture.pagePath }),
      stop("Draft answer that should be replaced by the streamed output."),
    ]);
    const service = new ExplorationWorkflowService({
      storage: fixture.storage,
      blobStore: fixture.blobStore,
      model,
      skillsRoot: join(process.cwd(), "skills"),
      rootDirectory: fixture.root,
    });

    const started = await service.startSession({
      task: "Return the answer as markdown.",
      documentIds: ["doc-alpha"],
    });
    await service.waitForSession(started.session_id);

    const session = service.getSession(started.session_id)!;
    assert.equal(session.status, "completed");
    assert.equal(session.finalResult, "## Clean Answer\n\nThis should render as markdown.");
    assert.equal(
      session.history.find((event) => event.type === "answer_done")?.data.final_result,
      "## Clean Answer\n\nThis should render as markdown.",
    );
    assert.equal(
      session.history.at(-1)?.data.final_result,
      "## Clean Answer\n\nThis should render as markdown.",
    );

    fixture.storage.close();
  });

});

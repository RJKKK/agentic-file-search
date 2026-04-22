import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import { createAgent, type ActionModel } from "../src/agent/agent.js";
import { ContextState } from "../src/agent/context-state.js";
import { loadSkills } from "../src/runtime/load-skills.js";
import { buildToolRegistry } from "../src/runtime/registry.js";
import { toFnArgs } from "../src/types/actions.js";
import type { DocumentCatalog } from "../src/types/skills.js";

class SequenceModel implements ActionModel {
  constructor(private readonly actions: unknown[]) {}

  async generateAction(): Promise<unknown> {
    const next = this.actions.shift();
    if (!next) {
      return {
        action: { final_result: "done" },
        reason: "No more scripted actions.",
      };
    }
    return next;
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

describe("agent behavior", () => {
  it("supports a basic page-first glob -> grep -> read flow", async () => {
    const root = join(tmpdir(), `afs-pages-${Date.now()}`);
    await mkdir(root, { recursive: true });
    const pagePath = join(root, "page-0001.md");
    await writeFile(
      pagePath,
      [
        "---",
        "page_no: 1",
        "original_filename: alpha.md",
        "heading: Purchase Price",
        "source_locator: page-1",
        "---",
        "",
        "The purchase price is $45,000,000.",
      ].join("\n"),
    );

    const skills = await loadSkills(join(process.cwd(), "skills"));
    const registry = buildToolRegistry(skills);
    const agent = createAgent(
      {
        model: new SequenceModel([
          toolCall("glob", { directory: root, pattern: "page-*.md" }),
          toolCall("grep", { file_path: root, pattern: "purchase price" }),
          toolCall("read", { file_path: pagePath }),
        ]),
      },
      registry,
    );
    agent.configureTask("Find the purchase price.");

    const first = await agent.takeAction();
    assert.ok("tool_name" in first.action);
    await agent.callTool(first.action.tool_name, toFnArgs(first.action));
    const second = await agent.takeAction();
    assert.ok("tool_name" in second.action);
    await agent.callTool(second.action.tool_name, toFnArgs(second.action));
    const third = await agent.takeAction();
    assert.ok("tool_name" in third.action);
    await agent.callTool(third.action.tool_name, toFnArgs(third.action));

    const history = agent.getHistory().map((item) => item.content).join("\n");
    const snapshot = agent.getContextState().snapshot();
    assert.match(history, /Glob receipt:/);
    assert.match(history, /Grep receipt:/);
    assert.match(history, /Parse receipt:/);
    assert.equal(snapshot.context_scope.active_file_path, pagePath);
    assert.equal(snapshot.context_scope.active_document_id, "alpha.md");
    assert.ok(snapshot.evidence_units.some((item) => item.file_path === pagePath && item.unit_no === 1));
  });

  it("supports list_indexed_documents -> get_document when a doc id is needed", async () => {
    const catalog: DocumentCatalog = {
      listDocuments() {
        return [
          {
            id: "doc-123",
            absolutePath: "C:/docs/alpha.md",
            label: "alpha.md",
            pageCount: 3,
          },
        ];
      },
      getDocument(docId) {
        if (docId !== "doc-123") {
          return null;
        }
        return {
          id: "doc-123",
          absolutePath: "C:/docs/alpha.md",
          label: "alpha.md",
          pageCount: 3,
          content: "Full document body.",
        };
      },
    };

    const skills = await loadSkills(join(process.cwd(), "skills"));
    const registry = buildToolRegistry(skills);
    const agent = createAgent(
      {
        model: new SequenceModel([
          toolCall("list_indexed_documents", {}),
          toolCall("get_document", { doc_id: "doc-123" }),
        ]),
        documentCatalog: catalog,
      },
      registry,
    );
    agent.configureTask("Read the full indexed document.");

    const first = await agent.takeAction();
    assert.ok("tool_name" in first.action);
    await agent.callTool(first.action.tool_name, toFnArgs(first.action));
    const second = await agent.takeAction();
    assert.ok("tool_name" in second.action);
    await agent.callTool(second.action.tool_name, toFnArgs(second.action));

    const history = agent.getHistory().map((item) => item.content).join("\n");
    const snapshot = agent.getContextState().snapshot();
    assert.match(history, /doc_id=doc-123/);
    assert.equal(snapshot.context_scope.active_document_id, "doc-123");
    assert.ok(snapshot.evidence_units.some((item) => item.document_id === "doc-123" && item.kind === "document_body"));
  });

  it("releases raw evidence for summarized batch documents while preserving snippets and coverage", () => {
    const state = new ContextState();
    state.registerDocuments([
      {
        documentId: "doc-alpha",
        label: "alpha.pdf",
        filePath: "C:/docs/alpha.pdf",
      },
    ]);
    state.ingestParseResult({
      documentId: "doc-alpha",
      filePath: "C:/docs/alpha.pdf",
      label: "alpha.pdf",
      units: [
        {
          unit_no: 1,
          source_locator: "page-1",
          heading: "Purchase Price",
          markdown: "The purchase price is $45,000,000.",
        },
        {
          unit_no: 2,
          source_locator: "page-2",
          heading: "Closing",
          markdown: "Closing follows later.",
        },
      ],
      totalUnits: 2,
      anchor: 1,
      window: 1,
    });

    const released = state.releaseDocumentEvidence({ documentId: "doc-alpha" });
    const snapshot = state.snapshot();
    assert.equal(released.released, true);
    assert.deepEqual(snapshot.context_scope.active_ranges, []);
    assert.deepEqual(snapshot.coverage_by_document["doc-alpha"].retrieved_ranges, [{ start: 1, end: 2 }]);
    assert.deepEqual(snapshot.coverage_by_document["doc-alpha"].summarized_ranges, [{ start: 1, end: 2 }]);
    assert.ok(snapshot.evidence_units.some((item) => item.snippet));
    assert.ok(!state.buildContextPack(5000).text.includes("Active evidence window (raw excerpts)"));
  });

  it("warns when a document range read is too wide and keeps only a narrow active window", async () => {
    const skills = await loadSkills(join(process.cwd(), "skills"));
    const registry = buildToolRegistry(skills);
    const readTool = registry.tools.read.handler;
    const contextState = new ContextState();
    const events: Array<{ type: string; data: Record<string, unknown> }> = [];
    contextState.registerDocuments([
      {
        documentId: "doc-yangtze",
        label: "yangtze.pdf",
        filePath: "C:/docs/yangtze.pdf",
      },
    ]);

    const result = await readTool(
      {
        document_id: "doc-yangtze",
        start_page: 30,
        end_page: 40,
      },
      {
        contextState,
        services: {
          indexSearch: {
            async resolvePageBatch() {
              return [
                {
                  document: {
                    id: "doc-yangtze",
                    original_filename: "yangtze.pdf",
                    relative_path: "yangtze.pdf",
                    absolute_path: "C:/docs/yangtze.pdf",
                    page_count: 80,
                  },
                  pages: [30, 31, 32].map((pageNo) => ({
                    page_no: pageNo,
                    file_path: `C:/docs/pages/page-${String(pageNo).padStart(4, "0")}.md`,
                    heading: `Page ${pageNo}`,
                    source_locator: `page-${pageNo}`,
                    markdown: `董事会成员信息 page ${pageNo}`,
                    char_count: 20,
                    page_label: String(pageNo),
                    is_synthetic_page: false,
                  })),
                  truncated: true,
                  omittedPages: [33, 34, 35, 36, 37, 38, 39, 40],
                },
              ];
            },
            emit(type: string, data: Record<string, unknown>) {
              events.push({ type, data });
            },
          },
        },
      } as never,
    );

    assert.match(result.receipt ?? "", /WARNING wide range requested/);
    assert.match(result.receipt ?? "", /omitted=33, 34, 35, 36, 37, 38, 39, 40/);
    assert.match(result.receipt ?? "", /must not be used as evidence/);
    assert.match(result.receipt ?? "", /returned=30-32/);
    assert.ok(!result.output.includes("page 40"));
    assert.deepEqual(contextState.snapshot().context_scope.active_ranges, [{ start: 30, end: 32 }]);
    assert.equal(events.find((event) => event.type === "pages_read")?.data.truncated, true);
  });

  it("stops after repeated identical tool calls", async () => {
    const root = join(tmpdir(), `afs-repeat-${Date.now()}`);
    await mkdir(root, { recursive: true });

    const skills = await loadSkills(join(process.cwd(), "skills"));
    const registry = buildToolRegistry(skills);
    const agent = createAgent(
      {
        model: new SequenceModel([
          toolCall("glob", { directory: root, pattern: "page-*.md" }),
          toolCall("glob", { directory: root, pattern: "page-*.md" }),
          toolCall("glob", { directory: root, pattern: "page-*.md" }),
        ]),
      },
      registry,
    );
    agent.configureTask("Find anything.");

    const first = await agent.takeAction();
    assert.ok("tool_name" in first.action);
    await agent.callTool(first.action.tool_name, toFnArgs(first.action));
    const second = await agent.takeAction();
    const history = agent.getHistory().map((item) => item.content).join("\n");

    assert.equal(first.action.tool_name, "glob");
    assert.ok("final_result" in second.action);
    assert.match(history, /Loop guard: you are repeating the same tool call/);
    assert.match(history, /Loop guard: you repeated the same tool call multiple times/);
  });

  it("repairs one invalid action response with the repair prompt", async () => {
    const root = join(tmpdir(), `afs-repair-${Date.now()}`);
    await mkdir(root, { recursive: true });

    const skills = await loadSkills(join(process.cwd(), "skills"));
    const registry = buildToolRegistry(skills);
    const agent = createAgent(
      {
        model: new SequenceModel([
          "not valid json",
          toolCall("glob", { directory: root, pattern: "page-*.md" }),
        ]),
      },
      registry,
    );
    agent.configureTask("Inspect pages.");

    const action = await agent.takeAction();
    assert.ok("tool_name" in action.action);
    await agent.callTool(action.action.tool_name, toFnArgs(action.action));
    const history = agent.getHistory().map((item) => item.content).join("\n");

    assert.equal(action.action.tool_name, "glob");
    assert.match(history, /Glob receipt:/);
  });

  it("repairs unknown tool names before they can loop the workflow", async () => {
    const root = join(tmpdir(), `afs-unknown-tool-${Date.now()}`);
    await mkdir(root, { recursive: true });

    const skills = await loadSkills(join(process.cwd(), "skills"));
    const registry = buildToolRegistry(skills);
    const agent = createAgent(
      {
        model: new SequenceModel([
          toolCall("stop", {}),
          {
            action: {
              final_result: "The answer is supported by the collected evidence.",
            },
            reason: "Stop with the final answer.",
          },
        ]),
      },
      registry,
    );
    agent.configureTask("Answer from the evidence.");

    const action = await agent.takeAction();
    assert.ok("final_result" in action.action);
    assert.equal(action.action.final_result, "The answer is supported by the collected evidence.");
  });
});

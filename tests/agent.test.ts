import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import { createAgent, type ActionModel } from "../src/agent/agent.js";
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

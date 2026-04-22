import assert from "node:assert/strict";
import { join } from "node:path";
import { describe, it } from "node:test";

import { createAgent } from "../src/agent/agent.js";
import {
  BEST_EFFORT_FINAL_PROMPT,
  REPEATED_TOOLCALL_PROMPT,
  renderActionRepairPrompt,
} from "../src/agent/prompts.js";
import { loadSkills } from "../src/runtime/load-skills.js";
import { buildToolRegistry } from "../src/runtime/registry.js";

describe("system prompt", () => {
  it("includes only active tools and preserves page-first guidance", async () => {
    const skills = await loadSkills(join(process.cwd(), "skills"));
    const registry = buildToolRegistry(skills);
    const agent = createAgent(
      {
        model: {
          async generateAction() {
            return {
              action: { final_result: "unused" },
              reason: "unused",
            };
          },
        },
      },
      registry,
    );

    const prompt = agent.getSystemPrompt();

    assert.match(prompt, /`glob`/);
    assert.match(prompt, /`grep`/);
    assert.match(prompt, /`read`/);
    assert.match(prompt, /`list_indexed_documents`/);
    assert.match(prompt, /`get_document`/);
    assert.ok(!prompt.includes("`scan_folder`"));
    assert.ok(!prompt.includes("`semantic_search`"));
    assert.match(prompt, /1\. Start with `glob`/);
    assert.match(prompt, /Structured Context Rules/);
    assert.match(prompt, /Three-Phase Page Exploration Strategy/);
    assert.match(prompt, /Providing Detailed Reasoning/);
    assert.match(prompt, /Citation Requirements for Final Answers/i);
  });

  it("renders repair and loop-guard prompts close to the legacy behavior", async () => {
    const skills = await loadSkills(join(process.cwd(), "skills"));
    const registry = buildToolRegistry(skills);
    const repairPrompt = renderActionRepairPrompt(registry);

    assert.match(repairPrompt, /Return exactly one JSON object and nothing else/);
    assert.match(repairPrompt, /tool_name/);
    assert.match(repairPrompt, /`glob`/);
    assert.match(repairPrompt, /`get_document`/);
    assert.match(REPEATED_TOOLCALL_PROMPT, /Do not repeat the exact same tool call again/);
    assert.match(BEST_EFFORT_FINAL_PROMPT, /Stop using tools now/);
  });
});

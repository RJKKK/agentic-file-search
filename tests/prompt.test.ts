import assert from "node:assert/strict";
import { join } from "node:path";
import { describe, it } from "node:test";

import { createAgent } from "../src/agent/agent.js";
import {
  BEST_EFFORT_FINAL_PROMPT,
  REPEATED_TOOLCALL_PROMPT,
  renderActionRepairPrompt,
} from "../src/agent/prompts.js";
import {
  buildVisionPromptMessages,
  renderImageSemantic,
  truncateImageSemanticPreview,
} from "../src/runtime/image-semantic.js";
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
    assert.match(prompt, /choose the previous page, next page, or both based on the evidence gap/);
    assert.match(prompt, /state which side is missing/);
    assert.match(prompt, /do not reread them/i);
    assert.ok(!prompt.includes("include BOTH the previous page and the next page"));
    assert.ok(!prompt.includes("read the previous and next page for that evidence page"));
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

  it("builds a vision prompt that requires detailed diagram semantics and strict JSON", () => {
    const prompt = buildVisionPromptMessages();

    assert.match(prompt.systemPrompt, /Return strict JSON only/i);
    assert.match(prompt.systemPrompt, /retrieval_summary/i);
    assert.match(prompt.systemPrompt, /detail_markdown/i);
    assert.match(prompt.userPrompt, /流程图、业务图、时序图、方框图：务必详尽描述/);
    assert.match(prompt.userPrompt, /结构图、架构图、关系图、网络拓扑图：务必详尽描述/);
    assert.match(prompt.userPrompt, /统计图表：务必详尽描述图表类型、坐标轴、单位、图例/);
    assert.match(prompt.userPrompt, /关系图、股权图、穿透图、组织关系图的硬性要求/);
    assert.match(prompt.userPrompt, /每条关系尽量写成主谓宾结构句子/);
    assert.match(prompt.userPrompt, /A 持股 B 100\.00%/);
    assert.match(prompt.userPrompt, /## 逐项关系/);
    assert.match(prompt.userPrompt, /detail_truncated/);
  });

  it("injects neighboring context into the vision prompt with non-hallucination guardrails", () => {
    const prompt = buildVisionPromptMessages({
      contextText: "## 公司与实际控制人之间的产权及控制关系的方框图\n曾毓群为实际控制人。",
    });

    assert.match(prompt.userPrompt, /邻近上下文/);
    assert.match(prompt.userPrompt, /公司与实际控制人之间的产权及控制关系的方框图/);
    assert.match(prompt.userPrompt, /不得仅凭邻近上下文补造/);
  });

  it("renders layered image semantics with a short retrieval text and a separate detailed text", () => {
    const rendered = renderImageSemantic({
      accessibleUrl: "/api/assets/images/demo",
      payload: {
        recognizable: true,
        retrieval_summary: "流程图描述采购申请到审批完成的主流程。",
        detail_markdown: "## 关键结构与关系\n- 发起 -> 审批 -> 归档\n\n## 详细说明\n" + "节点说明 ".repeat(1200),
        visible_text: "发起审批归档".repeat(120),
        keywords: ["采购申请", "审批流", "归档"],
        detail_truncated: false,
      },
    });

    assert.match(rendered.shortMarkdown ?? "", /流程图描述采购申请到审批完成的主流程/);
    assert.match(rendered.shortMarkdown ?? "", /\[Keywords\]/);
    assert.match(rendered.shortMarkdown ?? "", /!\[image\]\(\/api\/assets\/images\/demo\)/);
    assert.match(rendered.semanticDetailText ?? "", /## 关键结构与关系/);
    assert.match(rendered.semanticDetailText ?? "", /\[detail_truncated=true\]/);
    assert.equal((rendered.semanticText?.length ?? 0) > 0, true);
  });

  it("preserves full image links when truncating image semantic previews", () => {
    const preview = truncateImageSemanticPreview(
      "图片链接：![image](/api/assets/images/demo)\n\n" + "Detailed explanation ".repeat(40),
      80,
    );

    assert.match(preview, /图片链接：!\[image\]\(\/api\/assets\/images\/demo\)/);
    assert.match(preview, /\[truncated\]/);
    assert.doesNotMatch(preview, /!\[image\]\(\/api\/assets\/images\/dem(?:\)|$)/);
  });
});

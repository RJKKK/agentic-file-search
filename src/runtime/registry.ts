/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/workflow.py
*/

import type { LoadedSkill, RegisteredTool, ToolRegistry } from "../types/skills.js";

const PREFERRED_TOOL_ORDER = [
  "glob",
  "grep",
  "read",
  "list_indexed_documents",
  "get_document",
] as const;

export function buildToolRegistry(skills: LoadedSkill[]): ToolRegistry {
  const tools: Record<string, RegisteredTool> = {};
  const discoveredOrder: string[] = [];

  for (const skill of skills) {
    for (const definition of skill.manifest.tools) {
      if (tools[definition.name]) {
        throw new Error(`Duplicate tool name detected: ${definition.name}`);
      }
      tools[definition.name] = {
        skillId: skill.manifest.id,
        definition,
        handler: skill.module.tools[definition.name],
      };
      discoveredOrder.push(definition.name);
    }
  }

  const order = [
    ...PREFERRED_TOOL_ORDER.filter((name) => discoveredOrder.includes(name)),
    ...discoveredOrder.filter((name) => !PREFERRED_TOOL_ORDER.includes(name as never)).sort(),
  ];

  return { tools, order };
}

export function renderToolCatalog(registry: ToolRegistry): string {
  const lines = [
    "## Available Tools",
    "",
    "| Tool | Purpose | Parameters |",
    "|------|---------|------------|",
  ];

  for (const name of registry.order) {
    const tool = registry.tools[name];
    const parameters = tool.definition.parameters.length
      ? tool.definition.parameters.map((parameter) => parameter.name).join(", ")
      : "none";
    lines.push(`| \`${name}\` | ${tool.definition.description} | \`${parameters}\` |`);
  }

  return lines.join("\n");
}

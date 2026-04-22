/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/models.py
*/

import { z } from "zod";

import type { ContextState } from "../agent/context-state.js";

export const ParameterDefinitionSchema = z.object({
  name: z.string().min(1),
  description: z.string().min(1),
  required: z.boolean().default(true),
});

export const ToolDefinitionSchema = z.object({
  name: z.string().min(1),
  description: z.string().min(1),
  parameters: z.array(ParameterDefinitionSchema).default([]),
});

export const SkillManifestSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  status: z.enum(["active", "outdated"]),
  description: z.string().min(1),
  entry: z.string().min(1),
  tools: z.array(ToolDefinitionSchema).min(1),
  enabledByDefault: z.boolean(),
});

export type ParameterDefinition = z.infer<typeof ParameterDefinitionSchema>;
export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;
export type SkillManifest = z.infer<typeof SkillManifestSchema>;

export interface DocumentSummary {
  id: string;
  absolutePath: string;
  label: string;
  pageCount?: number;
  pagesDir?: string;
}

export interface DocumentRecord extends DocumentSummary {
  content: string;
}

export interface DocumentCatalog {
  listDocuments(): Promise<DocumentSummary[]> | DocumentSummary[];
  getDocument(docId: string): Promise<DocumentRecord | null> | DocumentRecord | null;
}

export interface ToolServices {
  documentCatalog?: DocumentCatalog;
}

export interface ToolExecutionContext {
  contextState: ContextState;
  services: ToolServices;
}

export interface ToolResult {
  output: string;
  receipt?: string;
}

export type ToolHandler = (
  input: Record<string, unknown>,
  context: ToolExecutionContext,
) => Promise<ToolResult> | ToolResult;

export interface SkillModule {
  tools: Record<string, ToolHandler>;
}

export interface LoadedSkill {
  manifest: SkillManifest;
  directory: string;
  module: SkillModule;
}

export interface RegisteredTool {
  skillId: string;
  definition: ToolDefinition;
  handler: ToolHandler;
}

export interface ToolRegistry {
  tools: Record<string, RegisteredTool>;
  order: string[];
}

/*
Reference: legacy/python/src/fs_explorer/models.py
Reference: legacy/python/src/fs_explorer/agent.py
*/

import { z } from "zod";

export const LEGACY_TOOL_NAMES = [
  "read",
  "grep",
  "glob",
  "scan_folder",
  "preview_file",
  "semantic_search",
  "get_document",
  "list_indexed_documents",
] as const;

export const ToolCallArgSchema = z.object({
  parameter_name: z.string().min(1),
  parameter_value: z.unknown(),
});

export const ToolCallActionSchema = z.object({
  tool_name: z.string().min(1),
  tool_input: z.array(ToolCallArgSchema).default([]),
});

export const StopActionSchema = z.object({
  final_result: z.string().min(1),
});

export const AskHumanActionSchema = z.object({
  question: z.string().min(1),
});

export const GoDeeperActionSchema = z.object({
  directory: z.string().min(1),
});

export const ContextPlanRangeSchema = z.object({
  start_unit: z.number().int().positive(),
  end_unit: z.number().int().positive().optional(),
});

export const ContextPlanSchema = z.object({
  operation: z.enum([
    "keep_ranges",
    "expand_range",
    "narrow_to_range",
    "switch_document",
    "promote_evidence_units",
    "summarize_stale_ranges",
    "stop_insufficient_evidence",
  ]),
  document_id: z.string().optional(),
  file_path: z.string().optional(),
  ranges: z.array(ContextPlanRangeSchema).default([]),
  evidence_ids: z.array(z.string()).default([]),
  note: z.string().optional(),
});

export const ActionSchema = z.object({
  action: z.union([
    ToolCallActionSchema,
    GoDeeperActionSchema,
    StopActionSchema,
    AskHumanActionSchema,
  ]),
  reason: z.string().min(1),
  context_plan: ContextPlanSchema.optional(),
});

export type ToolCallArg = z.infer<typeof ToolCallArgSchema>;
export type ToolCallAction = z.infer<typeof ToolCallActionSchema>;
export type StopAction = z.infer<typeof StopActionSchema>;
export type AskHumanAction = z.infer<typeof AskHumanActionSchema>;
export type GoDeeperAction = z.infer<typeof GoDeeperActionSchema>;
export type ContextPlan = z.infer<typeof ContextPlanSchema>;
export type Action = z.infer<typeof ActionSchema>;

export type ActionType = "stop" | "godeeper" | "toolcall" | "askhuman";

export function toActionType(action: Action): ActionType {
  if ("tool_name" in action.action) {
    return "toolcall";
  }
  if ("directory" in action.action) {
    return "godeeper";
  }
  if ("question" in action.action) {
    return "askhuman";
  }
  return "stop";
}

export function toFnArgs(action: ToolCallAction): Record<string, unknown> {
  return Object.fromEntries(
    action.tool_input.map((item) => [item.parameter_name, item.parameter_value]),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parametersToToolInput(parameters: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(parameters)) {
    return [...parameters] as Array<Record<string, unknown>>;
  }
  if (!isRecord(parameters)) {
    return [];
  }
  return Object.entries(parameters).map(([name, value]) => ({
    parameter_name: String(name),
    parameter_value: value,
  }));
}

function normalizeContextPlanCandidate(candidate: Record<string, unknown>): Record<string, unknown> {
  if (!("context_plan" in candidate)) {
    return candidate;
  }
  const contextPlan = candidate.context_plan;
  if (contextPlan == null) {
    return candidate;
  }
  if (!isRecord(contextPlan) || !("operation" in contextPlan)) {
    const normalized = { ...candidate };
    delete normalized.context_plan;
    return normalized;
  }
  return candidate;
}

function normalizeActionCandidate(candidate: Record<string, unknown>): Record<string, unknown> {
  const normalized = normalizeContextPlanCandidate({ ...candidate });
  const actionPayload = normalized.action;

  if (typeof actionPayload === "string") {
    const toolName = actionPayload.trim();
    if (LEGACY_TOOL_NAMES.includes(toolName as (typeof LEGACY_TOOL_NAMES)[number])) {
      normalized.action = {
        tool_name: toolName,
        tool_input: parametersToToolInput(normalized.parameters),
      };
    }
    return normalized;
  }

  if (isRecord(actionPayload)) {
    const actionDict = { ...actionPayload };
    const toolName = typeof actionDict.tool_name === "string" ? actionDict.tool_name.trim() : "";
    if (LEGACY_TOOL_NAMES.includes(toolName as (typeof LEGACY_TOOL_NAMES)[number])) {
      if (isRecord(actionDict.tool_input)) {
        actionDict.tool_input = parametersToToolInput(actionDict.tool_input);
      } else if (!("tool_input" in actionDict) && "parameters" in actionDict) {
        actionDict.tool_input = parametersToToolInput(actionDict.parameters);
      }
      normalized.action = actionDict;
    }
    return normalized;
  }

  if (typeof normalized.tool_name === "string") {
    const toolName = normalized.tool_name.trim();
    if (LEGACY_TOOL_NAMES.includes(toolName as (typeof LEGACY_TOOL_NAMES)[number])) {
      return {
        action: {
          tool_name: toolName,
          tool_input: parametersToToolInput(normalized.parameters),
        },
        reason: typeof normalized.reason === "string" ? normalized.reason : "",
        ...("context_plan" in normalized ? { context_plan: normalized.context_plan } : {}),
      };
    }
  }

  return normalized;
}

function escapeInvalidJsonBackslashes(text: string): string {
  const doubledDrivePaths = text.replace(
    /(:\s*")([A-Za-z]:\\[^"\n]*)(")/g,
    (_match, prefix: string, body: string, suffix: string) =>
      `${prefix}${body.replace(/\\+/g, "\\\\")}${suffix}`,
  );
  return doubledDrivePaths.replace(/(?<!\\)\\(?!["\\/bfnrtu])/g, "\\\\");
}

function iterActionJsonCandidates(rawText: string): Array<string | Record<string, unknown>> {
  const stripped = rawText.trim();
  if (!stripped) {
    return [];
  }

  const candidates: Array<string | Record<string, unknown>> = [stripped];
  const seenStrings = new Set<string>([stripped]);
  const escaped = escapeInvalidJsonBackslashes(stripped);
  if (escaped !== stripped) {
    seenStrings.add(escaped);
    candidates.push(escaped);
  }

  const fencedMatches = stripped.matchAll(/```(?:json)?\s*(.*?)```/gis);
  for (const match of fencedMatches) {
    const candidate = (match[1] ?? "").trim();
    if (candidate && !seenStrings.has(candidate)) {
      seenStrings.add(candidate);
      candidates.push(candidate);
      const escapedCandidate = escapeInvalidJsonBackslashes(candidate);
      if (escapedCandidate !== candidate && !seenStrings.has(escapedCandidate)) {
        seenStrings.add(escapedCandidate);
        candidates.push(escapedCandidate);
      }
    }
  }

  for (let index = 0; index < stripped.length; index += 1) {
    if (stripped[index] !== "{") {
      continue;
    }
    try {
      const parsed = JSON.parse(stripped.slice(index));
      if (isRecord(parsed)) {
        candidates.push(parsed);
        break;
      }
    } catch {
      continue;
    }
  }

  return candidates;
}

function extractFinalResultCandidate(value: unknown): string | null {
  if (!isRecord(value)) {
    return null;
  }
  if (typeof value.final_result === "string" && value.final_result.trim()) {
    return value.final_result.trim();
  }
  if (isRecord(value.action) && typeof value.action.final_result === "string" && value.action.final_result.trim()) {
    return value.action.final_result.trim();
  }
  return null;
}

export function parseAction(input: unknown): Action {
  if (typeof input !== "string") {
    const normalized = isRecord(input) ? normalizeActionCandidate(input) : input;
    return ActionSchema.parse(normalized);
  }

  for (const candidate of iterActionJsonCandidates(input)) {
    try {
      if (isRecord(candidate)) {
        return ActionSchema.parse(normalizeActionCandidate(candidate));
      }
      try {
        const parsed = JSON.parse(candidate);
        if (isRecord(parsed)) {
          return ActionSchema.parse(normalizeActionCandidate(parsed));
        }
      } catch {
        // Fall through to schema parse of the candidate string.
      }
      return ActionSchema.parse(JSON.parse(candidate));
    } catch {
      continue;
    }
  }

  throw new Error("Could not parse action response.");
}

export function fallbackStopAction(rawText: string): Action | null {
  const text = rawText.trim();
  if (!text) {
    return null;
  }
  return {
    action: { final_result: text },
    reason:
      "Model returned unstructured text instead of the required JSON action. Treating that text as the final answer.",
  };
}

export function unwrapFinalAnswerEnvelope(rawText: string): string {
  const text = rawText.trim();
  if (!text) {
    return rawText;
  }

  for (const candidate of iterActionJsonCandidates(text)) {
    const extracted = extractFinalResultCandidate(
      typeof candidate === "string"
        ? (() => {
            try {
              return JSON.parse(candidate);
            } catch {
              return null;
            }
          })()
        : candidate,
    );
    if (extracted) {
      return extracted;
    }
  }

  return rawText;
}

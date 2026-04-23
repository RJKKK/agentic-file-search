/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/context_budget.py
Reference: legacy/python/src/fs_explorer/context_state.py
Reference: legacy/python/src/fs_explorer/models.py
*/

import { ContextBudgetManager } from "./context-budget.js";
import {
  ContextState,
  type ContextPlanResult,
  type RegisteredDocument,
} from "./context-state.js";
import {
  fallbackStopAction,
  parseAction,
  toActionType,
  toFnArgs,
  type Action,
} from "../types/actions.js";
import type { ContextBudgetStats } from "./context-budget.js";
import type { AgentMessage } from "../types/messages.js";
import type {
  DocumentCatalog,
  ToolServices,
  ToolDefinition,
  ToolRegistry,
} from "../types/skills.js";
import type { IndexSearchServiceContract, RuntimeEventEmitter } from "../types/search.js";
import {
  BEST_EFFORT_FINAL_PROMPT,
  REPEATED_TOOLCALL_PROMPT,
  renderBoundaryFirstPrompt,
  renderCoveredReadPrompt,
  renderRepeatedCandidateSearchPrompt,
  renderActionRepairPrompt,
  renderFinalAnswerSystemPrompt,
  renderFinalAnswerUserPrompt,
  renderSystemPrompt,
} from "./prompts.js";

export interface ActionModelRequest {
  systemPrompt: string;
  messages: AgentMessage[];
  task: string;
  availableTools: ToolDefinition[];
}

export interface FinalAnswerRequest {
  systemPrompt: string;
  messages: AgentMessage[];
  task: string;
  draftAnswer: string;
}

export interface ActionModel {
  generateAction(request: ActionModelRequest): Promise<unknown> | unknown;
  streamFinalAnswer?(request: FinalAnswerRequest): AsyncIterable<string> | Promise<AsyncIterable<string>>;
}

export interface AgentConfig {
  model: ActionModel;
  documentCatalog?: DocumentCatalog;
  indexSearch?: IndexSearchServiceContract;
  emitRuntimeEvent?: RuntimeEventEmitter;
  contextBudget?: {
    maxInputChars?: number;
    minRecentMessages?: number;
  };
}

export class FsExplorerAgent {
  private readonly contextState = new ContextState();

  private readonly history: AgentMessage[] = [];

  private readonly budgetManager: ContextBudgetManager;

  private readonly services: ToolServices;

  private task = "";

  private lastToolSignature: string | null = null;

  private repeatGuardHits = 0;

  private readonly seenCandidateSearchSignatures = new Set<string>();

  private readonly candidateSearchGuardHits = new Map<string, number>();

  private readonly coveredReadGuardHits = new Map<string, number>();

  private lastContextPlanResult: ContextPlanResult | null = null;

  private lastContextBudgetStats: ContextBudgetStats | null = null;

  private lastContextPackStats: Record<string, unknown> = {};

  constructor(
    private readonly config: AgentConfig,
    private readonly registry: ToolRegistry,
  ) {
    this.budgetManager = new ContextBudgetManager(config.contextBudget);
    this.services = {
      documentCatalog: config.documentCatalog,
      indexSearch: config.indexSearch,
      emitRuntimeEvent: config.emitRuntimeEvent,
    };
  }

  configureTask(
    task: string,
    documents: RegisteredDocument[] = [],
    options: { rawHistory?: boolean } = {},
  ): void {
    this.task = task;
    this.contextState.setTask(task);
    this.contextState.registerDocuments(documents);
    this.history.push({
      role: "user",
      content: options.rawHistory ? task : `User task: ${task}`,
    });
  }

  getSystemPrompt(): string {
    return renderSystemPrompt(this.registry);
  }

  getContextState(): ContextState {
    return this.contextState;
  }

  getHistory(): AgentMessage[] {
    return [...this.history];
  }

  getLastContextPlanResult(): ContextPlanResult | null {
    return this.lastContextPlanResult;
  }

  consumeLastContextPlanResult(): ContextPlanResult | null {
    const result = this.lastContextPlanResult;
    this.lastContextPlanResult = null;
    return result;
  }

  getContextBudgetStats(): Record<string, unknown> {
    return {
      history: this.lastContextBudgetStats ?? {},
      structured_context: this.lastContextPackStats,
    };
  }

  async takeAction(finalOnly = false): Promise<Action> {
    const requestMessages = this.buildModelMessages(4_000);
    const rawAction = await this.config.model.generateAction({
      systemPrompt: this.getSystemPrompt(),
      messages: requestMessages,
      task: this.task,
      availableTools: this.registry.order.map((name) => this.registry.tools[name].definition),
    });

    let action: Action | null = null;
    let parseError: unknown = null;
    try {
      action = parseAction(rawAction);
    } catch (error) {
      parseError = error;
    }

    if (
      action &&
      toActionType(action) === "toolcall" &&
      "tool_name" in action.action &&
      !this.registry.tools[action.action.tool_name]
    ) {
      parseError = new Error(`Unknown tool in action response: ${action.action.tool_name}`);
      action = null;
    }

    if (!action) {
      const repaired = await this.repairActionResponse(requestMessages, rawAction);
      if (
        repaired &&
        !(
          toActionType(repaired) === "toolcall" &&
          "tool_name" in repaired.action &&
          !this.registry.tools[repaired.action.tool_name]
        )
      ) {
        action = repaired;
      } else {
        action =
          (typeof rawAction === "string" ? fallbackStopAction(rawAction) : null) ?? {
            action: {
              final_result: `Could not parse action response: ${String(parseError)}`,
            },
            reason: "Stopped after receiving an invalid action payload.",
          };
        this.history.push({
          role: "assistant",
          content: JSON.stringify(action),
        });
        return action;
      }
    }

    if (finalOnly && toActionType(action) !== "stop") {
      return this.bestEffortStopAction();
    }

    const boundaryRedirectPrompt = this.boundaryRedirectPrompt(action);
    if (boundaryRedirectPrompt) {
      this.history.push({
        role: "user",
        content: boundaryRedirectPrompt,
      });
      return this.takeAction(finalOnly);
    }

    const coverageGuard = this.coverageLoopGuard(action);
    if (coverageGuard === "stop") {
      return this.bestEffortStopAction();
    }
    if (coverageGuard) {
      this.history.push({
        role: "user",
        content: coverageGuard,
      });
      return this.takeAction(finalOnly);
    }

    this.history.push({
      role: "assistant",
      content: JSON.stringify(action),
    });
    // Reference: legacy/python/src/fs_explorer/agent.py _maybe_apply_context_plan.
    this.lastContextPlanResult = this.contextState.applyContextPlan(action.context_plan);

    if (toActionType(action) !== "toolcall" || !("tool_name" in action.action)) {
      return action;
    }

    const toolAction = action.action;
    const toolInput = toFnArgs(toolAction);
    const signature = this.toolSignature(toolAction.tool_name, toolInput);
    if (signature === this.lastToolSignature) {
      this.repeatGuardHits += 1;
      // Reference: legacy/python/src/fs_explorer/agent.py repeated tool-call loop guard.
      if (this.repeatGuardHits >= 2) {
        this.history.push({
          role: "user",
          content: BEST_EFFORT_FINAL_PROMPT,
        });
        const finalResult = await this.takeAction(true);
        if ("final_result" in finalResult.action) {
          return finalResult;
        }
        return this.bestEffortStopAction();
      }
      this.history.push({
        role: "user",
        content: REPEATED_TOOLCALL_PROMPT,
      });
      return this.takeAction();
    } else {
      this.repeatGuardHits = 0;
    }

    return action;
  }

  async *streamFinalAnswer(draftAnswer: string): AsyncIterable<string> {
    const trimmedDraft = draftAnswer.trim();
    const streamFactory = this.config.model.streamFinalAnswer;
    if (!streamFactory) {
      if (trimmedDraft) {
        yield trimmedDraft;
      }
      return;
    }

    const requestMessages = this.buildModelMessages(7_000);
    requestMessages.push({
      role: "user",
      content: renderFinalAnswerUserPrompt({
        task: this.task,
        draftAnswer: trimmedDraft,
      }),
    });

    const stream = await streamFactory({
      systemPrompt: renderFinalAnswerSystemPrompt(),
      messages: requestMessages,
      task: this.task,
      draftAnswer: trimmedDraft,
    });

    let streamedChars = 0;
    for await (const chunk of stream) {
      const text = String(chunk ?? "");
      if (!text) {
        continue;
      }
      streamedChars += text.length;
      yield text;
    }

    if (streamedChars === 0 && trimmedDraft) {
      yield trimmedDraft;
    }
  }

  private buildModelMessages(contextPackChars: number): AgentMessage[] {
    const compacted = this.budgetManager.compactHistory(this.history);
    const contextPack = this.contextState.buildContextPack(contextPackChars);
    this.lastContextBudgetStats = compacted.stats;
    this.lastContextPackStats = contextPack.stats;
    const requestMessages = [...compacted.messages];
    if (contextPack.text.trim()) {
      requestMessages.push({
        role: "user",
        content: contextPack.text,
      });
    }
    return requestMessages;
  }

  async callTool(toolName: string, toolInput: Record<string, unknown>): Promise<void> {
    const registered = this.registry.tools[toolName];
    if (!registered) {
      this.appendHistoryReceipt(`Tool result for ${toolName}:\n\nUnknown tool: ${toolName}`);
      return;
    }

    let resultText = "";
    let historyReceipt: string | undefined;
    try {
      const result = await registered.handler(toolInput, {
        contextState: this.contextState,
        services: this.services,
      });
      resultText = result.output;
      historyReceipt = result.receipt;
    } catch (error) {
      resultText = `An error occurred while calling tool ${toolName} with ${JSON.stringify(toolInput)}: ${String(error)}`;
    }
    this.lastToolSignature = this.toolSignature(toolName, toolInput);
    if (!historyReceipt) {
      historyReceipt = `Tool result for ${toolName}:\n\n${resultText}`;
    }
    this.appendHistoryReceipt(historyReceipt);
  }

  private bestEffortStopAction(): Action {
    const finalResult = this.contextState.bestEffortAnswer();
    const action: Action = {
      action: {
        final_result: finalResult,
      },
      reason: "Stopped repeated tool calls and returned the best available evidence.",
    };
    this.history.push({
      role: "assistant",
      content: JSON.stringify(action),
    });
    return action;
  }

  private async repairActionResponse(
    requestMessages: AgentMessage[],
    invalidAction: unknown,
  ): Promise<Action | null> {
    const invalidText =
      typeof invalidAction === "string" ? invalidAction : JSON.stringify(invalidAction, null, 2);
    const repaired = await this.config.model.generateAction({
      systemPrompt: this.getSystemPrompt(),
      messages: [
        ...requestMessages,
        {
          role: "user",
          content: `${renderActionRepairPrompt(this.registry)}\n\nPrevious invalid reply:\n\`\`\`text\n${invalidText}\n\`\`\``,
        },
      ],
      task: this.task,
      availableTools: this.registry.order.map((name) => this.registry.tools[name].definition),
    });
    try {
      return parseAction(repaired);
    } catch {
      return typeof repaired === "string" ? fallbackStopAction(repaired) : null;
    }
  }

  private appendHistoryReceipt(text: string): void {
    this.history.push({
      role: "user",
      content: text,
    });
  }

  private boundaryRedirectPrompt(action: Action): string | null {
    if (toActionType(action) !== "toolcall" || !("tool_name" in action.action)) {
      return null;
    }
    if (action.action.tool_name === "page_boundary_context") {
      return null;
    }
    if (action.action.tool_name !== "read") {
      return null;
    }

    const pendingBoundaryHintIndex = this.findLastHistoryIndex((item) =>
      item.role === "user" && /boundary_hint=/i.test(item.content),
    );
    if (pendingBoundaryHintIndex < 0) {
      return null;
    }
    const resolvedBoundaryIndex = this.findLastHistoryIndex((item) =>
      item.role === "user" && /Page boundary receipt:/i.test(item.content),
    );
    if (resolvedBoundaryIndex > pendingBoundaryHintIndex) {
      return null;
    }

    const redirect = this.classifyBoundaryRedirect(toFnArgs(action.action));
    if (!redirect) {
      return null;
    }

    return renderBoundaryFirstPrompt({
      suggestedDirection: redirect.direction,
      activeDocumentId: redirect.activeDocumentId,
      activeRangeLabel: redirect.activeRangeLabel,
      requestedRangeLabel: redirect.requestedRangeLabel,
      suggestedToolInput: redirect.suggestedToolInput,
    });
  }

  private coverageLoopGuard(action: Action): string | "stop" | null {
    if (toActionType(action) !== "toolcall" || !("tool_name" in action.action)) {
      return null;
    }
    const toolInput = toFnArgs(action.action);
    if (action.action.tool_name === "search_candidates") {
      return this.candidateSearchLoopGuard(toolInput);
    }
    if (action.action.tool_name === "read") {
      return this.coveredReadLoopGuard(toolInput);
    }
    return null;
  }

  private candidateSearchLoopGuard(toolInput: Record<string, unknown>): string | "stop" | null {
    const query = String(toolInput.query ?? "").trim();
    if (!query) {
      return null;
    }
    const signature = `search_candidates:${query.toLowerCase().replace(/\s+/g, " ")}`;
    const hasPageEvidence = this.hasCollectedPageEvidence();
    if (!this.seenCandidateSearchSignatures.has(signature)) {
      this.seenCandidateSearchSignatures.add(signature);
      return null;
    }
    if (!hasPageEvidence) {
      return null;
    }

    const hits = (this.candidateSearchGuardHits.get(signature) ?? 0) + 1;
    this.candidateSearchGuardHits.set(signature, hits);
    if (hits >= 2) {
      this.history.push({
        role: "user",
        content: BEST_EFFORT_FINAL_PROMPT,
      });
      return "stop";
    }
    return renderRepeatedCandidateSearchPrompt({ query });
  }

  private coveredReadLoopGuard(toolInput: Record<string, unknown>): string | "stop" | null {
    const duplicate = this.coveredReadSignature(toolInput);
    if (!duplicate) {
      return null;
    }
    const hits = (this.coveredReadGuardHits.get(duplicate.signature) ?? 0) + 1;
    this.coveredReadGuardHits.set(duplicate.signature, hits);
    if (hits >= 2) {
      this.history.push({
        role: "user",
        content: BEST_EFFORT_FINAL_PROMPT,
      });
      return "stop";
    }
    return renderCoveredReadPrompt({ requestedLabel: duplicate.label });
  }

  private hasCollectedPageEvidence(): boolean {
    return this.contextState.snapshot().evidence_units.some((item) =>
      ["parsed_unit", "page_boundary_context", "document_body"].includes(String(item.kind ?? "")),
    );
  }

  private coveredReadSignature(
    toolInput: Record<string, unknown>,
  ): { signature: string; label: string } | null {
    const documentId = typeof toolInput.document_id === "string" ? toolInput.document_id.trim() : "";
    if (documentId) {
      const startPage = this.toPositiveInt(toolInput.start_page);
      const endPage = this.toPositiveInt(toolInput.end_page) ?? startPage;
      if (startPage != null && endPage != null) {
        const start = Math.min(startPage, endPage);
        const end = Math.max(startPage, endPage);
        if (this.isDocumentRangeCovered(documentId, start, end)) {
          return {
            signature: `read:document:${documentId}:${start}-${end}`,
            label: `${documentId} pages ${start === end ? String(start) : `${start}-${end}`}`,
          };
        }
      }
    }

    const filePath = typeof toolInput.file_path === "string" ? toolInput.file_path.trim() : "";
    if (filePath && this.isFilePageCovered(filePath)) {
      const pageNo = this.parsePageNoFromPath(filePath);
      return {
        signature: `read:file:${filePath.toLowerCase()}`,
        label: pageNo != null ? `${filePath} (page ${pageNo})` : filePath,
      };
    }

    const filePaths = Array.isArray(toolInput.file_paths)
      ? toolInput.file_paths.map((item) => String(item).trim()).filter(Boolean)
      : [];
    if (filePaths.length > 0 && filePaths.every((item) => this.isFilePageCovered(item))) {
      return {
        signature: `read:files:${filePaths.map((item) => item.toLowerCase()).sort().join("|")}`,
        label: filePaths.join(", "),
      };
    }

    return null;
  }

  private isDocumentRangeCovered(documentId: string, start: number, end: number): boolean {
    const coverage = this.contextState.snapshot().coverage_by_document[documentId];
    const ranges = Array.isArray(coverage?.retrieved_ranges) ? coverage.retrieved_ranges : [];
    for (let pageNo = start; pageNo <= end; pageNo += 1) {
      const covered = ranges.some((range) => {
        const record = range as Record<string, unknown>;
        return Number(record.start) <= pageNo && pageNo <= Number(record.end);
      });
      if (!covered) {
        return false;
      }
    }
    return true;
  }

  private isFilePageCovered(filePath: string): boolean {
    const pageNo = this.parsePageNoFromPath(filePath);
    if (pageNo == null) {
      return false;
    }
    const normalizedPath = filePath.toLowerCase();
    return this.contextState.snapshot().evidence_units.some(
      (item) =>
        String(item.kind ?? "") === "parsed_unit" &&
        Number(item.unit_no) === pageNo &&
        String(item.file_path ?? "").toLowerCase() === normalizedPath,
    );
  }

  private toolSignature(toolName: string, toolInput: Record<string, unknown>): string {
    return JSON.stringify({
      tool_name: toolName,
      tool_input: toolInput,
    });
  }

  private findLastHistoryIndex(predicate: (message: AgentMessage) => boolean): number {
    for (let index = this.history.length - 1; index >= 0; index -= 1) {
      if (predicate(this.history[index]!)) {
        return index;
      }
    }
    return -1;
  }

  private classifyBoundaryRedirect(
    toolInput: Record<string, unknown>,
  ): {
    direction: "previous" | "next" | "both";
    activeDocumentId: string | null;
    activeRangeLabel: string | null;
    requestedRangeLabel: string | null;
    suggestedToolInput: Record<string, unknown>;
  } | null {
    const snapshot = this.contextState.snapshot();
    const activeDocumentId = snapshot.context_scope.active_document_id || null;
    const activeFilePath = snapshot.context_scope.active_file_path || null;
    const activeRanges = [...(snapshot.context_scope.active_ranges ?? [])]
      .filter(
        (item): item is { start: number; end: number } =>
          typeof item?.start === "number" && typeof item?.end === "number",
      )
      .sort((left, right) => left.start - right.start);
    if (activeRanges.length === 0) {
      return null;
    }
    const activeStart = activeRanges[0]!.start;
    const activeEnd = activeRanges.at(-1)!.end;
    const activeRangeLabel = this.renderUnitRange(activeStart, activeEnd);

    const documentId = typeof toolInput.document_id === "string" ? toolInput.document_id.trim() : "";
    if (documentId && activeDocumentId && documentId === activeDocumentId) {
      const startPage = this.toPositiveInt(toolInput.start_page);
      const endPage = this.toPositiveInt(toolInput.end_page) ?? startPage;
      if (startPage != null && endPage != null) {
        const direction = this.classifyRequestedRangeDirection({
          activeStart,
          activeEnd,
          requestedStart: Math.min(startPage, endPage),
          requestedEnd: Math.max(startPage, endPage),
        });
        if (direction) {
          return {
            direction,
            activeDocumentId,
            activeRangeLabel,
            requestedRangeLabel: this.renderUnitRange(Math.min(startPage, endPage), Math.max(startPage, endPage)),
            suggestedToolInput: {
              document_id: activeDocumentId,
              start_page: activeStart,
              end_page: activeEnd,
              direction,
            },
          };
        }
      }
    }

    const filePath = typeof toolInput.file_path === "string" ? toolInput.file_path.trim() : "";
    if (filePath) {
      const requestedPage = this.parsePageNoFromPath(filePath);
      if (
        requestedPage != null &&
        activeFilePath &&
        this.pageDirectory(filePath) === this.pageDirectory(activeFilePath)
      ) {
        const direction = this.classifyRequestedRangeDirection({
          activeStart,
          activeEnd,
          requestedStart: requestedPage,
          requestedEnd: requestedPage,
        });
        if (direction) {
          const suggestedToolInput =
            activeStart === activeEnd
              ? { file_path: activeFilePath, direction }
              : activeDocumentId
                ? {
                    document_id: activeDocumentId,
                    start_page: activeStart,
                    end_page: activeEnd,
                    direction,
                  }
                : { file_path: activeFilePath, direction };
          return {
            direction,
            activeDocumentId,
            activeRangeLabel,
            requestedRangeLabel: this.renderUnitRange(requestedPage, requestedPage),
            suggestedToolInput,
          };
        }
      }
    }

    const filePaths = Array.isArray(toolInput.file_paths)
      ? toolInput.file_paths.map((item) => String(item).trim()).filter(Boolean)
      : [];
    if (filePaths.length > 0) {
      if (!activeFilePath || filePaths.some((item) => this.pageDirectory(item) !== this.pageDirectory(activeFilePath))) {
        return null;
      }
      const pageNos = filePaths
        .map((item) => this.parsePageNoFromPath(item))
        .filter((item): item is number => item != null)
        .sort((left, right) => left - right);
      if (pageNos.length > 0) {
        const direction = this.classifyRequestedRangeDirection({
          activeStart,
          activeEnd,
          requestedStart: pageNos[0]!,
          requestedEnd: pageNos.at(-1)!,
        });
        if (direction) {
          return {
            direction,
            activeDocumentId,
            activeRangeLabel,
            requestedRangeLabel: this.renderUnitRange(pageNos[0]!, pageNos.at(-1)!),
            suggestedToolInput:
              activeDocumentId && activeStart !== activeEnd
                ? {
                    document_id: activeDocumentId,
                    start_page: activeStart,
                    end_page: activeEnd,
                    direction,
                  }
                : {
                    file_path: activeFilePath,
                    direction,
                  },
          };
        }
      }
    }

    return null;
  }

  private classifyRequestedRangeDirection(input: {
    activeStart: number;
    activeEnd: number;
    requestedStart: number;
    requestedEnd: number;
  }): "previous" | "next" | "both" | null {
    const { activeStart, activeEnd, requestedStart, requestedEnd } = input;
    const beforeDistance = activeStart - requestedEnd;
    const afterDistance = requestedStart - activeEnd;
    const extendsBefore = requestedStart < activeStart && beforeDistance <= 2;
    const extendsAfter = requestedEnd > activeEnd && afterDistance <= 2;
    if (extendsBefore && extendsAfter) {
      return "both";
    }
    if (extendsBefore) {
      return "previous";
    }
    if (extendsAfter) {
      return "next";
    }
    return null;
  }

  private renderUnitRange(start: number, end: number): string {
    return start === end ? `page ${start}` : `pages ${start}-${end}`;
  }

  private toPositiveInt(value: unknown): number | null {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      return null;
    }
    return parsed;
  }

  private parsePageNoFromPath(filePath: string): number | null {
    const match = String(filePath).match(/page-(\d+)\.md$/i);
    return match ? Number.parseInt(match[1] ?? "", 10) : null;
  }

  private pageDirectory(filePath: string): string {
    return String(filePath).replace(/[\\/][^\\/]+$/, "").toLowerCase();
  }
}

export function createAgent(config: AgentConfig, registry: ToolRegistry): FsExplorerAgent {
  return new FsExplorerAgent(config, registry);
}

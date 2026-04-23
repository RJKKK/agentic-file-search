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
  renderActionRepairPrompt,
  renderFinalAnswerSystemPrompt,
  renderFinalAnswerUserPrompt,
  renderRedundantReadPrompt,
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

  private redundantReadGuardHits = 0;

  private readonly readTargetIndex = new Map<string, { documentId: string; unitNo: number }>();

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

    this.history.push({
      role: "assistant",
      content: JSON.stringify(action),
    });
    // Reference: legacy/python/src/fs_explorer/agent.py _maybe_apply_context_plan.
    this.lastContextPlanResult = this.contextState.applyContextPlan(action.context_plan);

    if (finalOnly && toActionType(action) !== "stop") {
      return this.bestEffortStopAction();
    }

    if (toActionType(action) !== "toolcall" || !("tool_name" in action.action)) {
      return action;
    }

    const toolAction = action.action;
    const toolInput = toFnArgs(toolAction);
    const redundantReadPrompt = this.redundantReadPrompt(toolAction.tool_name, toolInput);
    if (redundantReadPrompt) {
      this.redundantReadGuardHits += 1;
      if (this.redundantReadGuardHits >= 2) {
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
        content: redundantReadPrompt,
      });
      return this.takeAction();
    }
    this.redundantReadGuardHits = 0;
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
    this.noteReadTarget(toolName, toolInput);
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

  private toolSignature(toolName: string, toolInput: Record<string, unknown>): string {
    return JSON.stringify({
      tool_name: toolName,
      tool_input: toolInput,
    });
  }

  private redundantReadPrompt(toolName: string, toolInput: Record<string, unknown>): string | null {
    if (toolName !== "read") {
      return null;
    }
    const filePath = typeof toolInput.file_path === "string" ? toolInput.file_path.trim() : "";
    if (!filePath) {
      return null;
    }
    const readTarget = this.readTargetIndex.get(filePath);
    if (!readTarget) {
      return null;
    }

    const snapshot = this.contextState.snapshot();
    const activeDocumentId = snapshot.context_scope.active_document_id;
    const activeRanges = snapshot.context_scope.active_ranges;
    const hasMergedWindow = activeRanges.some((range) => range.end > range.start) || activeRanges.length > 1;
    if (!hasMergedWindow || activeDocumentId !== readTarget.documentId) {
      return null;
    }

    const alreadyActive = activeRanges.some(
      (range) => readTarget.unitNo >= range.start && readTarget.unitNo <= range.end,
    );
    if (!alreadyActive) {
      return null;
    }

    const minUnit = Math.min(...activeRanges.map((range) => range.start));
    const maxUnit = Math.max(...activeRanges.map((range) => range.end));
    const outwardPages = [minUnit > 1 ? minUnit - 1 : null, maxUnit + 1].filter(
      (value): value is number => value != null && value !== readTarget.unitNo,
    );

    return renderRedundantReadPrompt({
      filePath,
      unitNo: readTarget.unitNo,
      activeRanges,
      outwardPages,
    });
  }

  private noteReadTarget(toolName: string, toolInput: Record<string, unknown>): void {
    if (toolName !== "read") {
      return;
    }
    const filePath = typeof toolInput.file_path === "string" ? toolInput.file_path.trim() : "";
    const unitNo = this.extractReadUnitNo(filePath);
    const documentId = this.contextState.snapshot().context_scope.active_document_id;
    if (!filePath || unitNo == null || !documentId) {
      return;
    }
    this.readTargetIndex.set(filePath, {
      documentId,
      unitNo,
    });
  }

  private extractReadUnitNo(filePath: string): number | null {
    const match = /page-(\d+)\.md$/i.exec(filePath.trim());
    if (!match) {
      return null;
    }
    const unitNo = Number.parseInt(match[1] ?? "", 10);
    return Number.isFinite(unitNo) && unitNo > 0 ? unitNo : null;
  }
}

export function createAgent(config: AgentConfig, registry: ToolRegistry): FsExplorerAgent {
  return new FsExplorerAgent(config, registry);
}

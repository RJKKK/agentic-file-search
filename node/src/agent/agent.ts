/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/context_budget.py
Reference: legacy/python/src/fs_explorer/context_state.py
Reference: legacy/python/src/fs_explorer/models.py
*/

import { ContextBudgetManager } from "./context-budget.js";
import { ContextState, type RegisteredDocument } from "./context-state.js";
import {
  fallbackStopAction,
  parseAction,
  toActionType,
  toFnArgs,
  type Action,
} from "../types/actions.js";
import type { AgentMessage } from "../types/messages.js";
import type {
  DocumentCatalog,
  ToolDefinition,
  ToolRegistry,
  ToolServices,
} from "../types/skills.js";
import {
  BEST_EFFORT_FINAL_PROMPT,
  REPEATED_TOOLCALL_PROMPT,
  renderActionRepairPrompt,
  renderSystemPrompt,
} from "./prompts.js";

export interface ActionModelRequest {
  systemPrompt: string;
  messages: AgentMessage[];
  task: string;
  availableTools: ToolDefinition[];
}

export interface ActionModel {
  generateAction(request: ActionModelRequest): Promise<unknown> | unknown;
}

export interface AgentConfig {
  model: ActionModel;
  documentCatalog?: DocumentCatalog;
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

  constructor(
    private readonly config: AgentConfig,
    private readonly registry: ToolRegistry,
  ) {
    this.budgetManager = new ContextBudgetManager(config.contextBudget);
    this.services = {
      documentCatalog: config.documentCatalog,
    };
  }

  configureTask(task: string, documents: RegisteredDocument[] = []): void {
    this.task = task;
    this.contextState.setTask(task);
    this.contextState.registerDocuments(documents);
    this.history.push({
      role: "user",
      content: `User task: ${task}`,
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

  async takeAction(finalOnly = false): Promise<Action> {
    const compacted = this.budgetManager.compactHistory(this.history);
    const contextPack = this.contextState.buildContextPack(4_000);
    const requestMessages = [...compacted.messages];
    if (contextPack.text.trim()) {
      requestMessages.push({
        role: "user",
        content: contextPack.text,
      });
    }

    const rawAction = await this.config.model.generateAction({
      systemPrompt: this.getSystemPrompt(),
      messages: requestMessages,
      task: this.task,
      availableTools: this.registry.order.map((name) => this.registry.tools[name].definition),
    });

    let action: Action;
    try {
      action = parseAction(rawAction);
    } catch (error) {
      const repaired = await this.repairActionResponse(requestMessages, rawAction);
      if (repaired) {
        action = repaired;
      } else {
        action =
          (typeof rawAction === "string" ? fallbackStopAction(rawAction) : null) ?? {
            action: {
              final_result: `Could not parse action response: ${String(error)}`,
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

    if (finalOnly && toActionType(action) !== "stop") {
      return this.bestEffortStopAction();
    }

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

    await this.callTool(toolAction.tool_name, toolInput);
    return action;
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

  private toolSignature(toolName: string, toolInput: Record<string, unknown>): string {
    return JSON.stringify({
      tool_name: toolName,
      tool_input: toolInput,
    });
  }
}

export function createAgent(config: AgentConfig, registry: ToolRegistry): FsExplorerAgent {
  return new FsExplorerAgent(config, registry);
}

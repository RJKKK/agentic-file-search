/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/context_budget.py
*/

export type AgentRole = "system" | "user" | "assistant" | "tool";

export interface AgentMessage {
  role: AgentRole;
  content: string;
}


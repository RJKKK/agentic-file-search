/*
Reference: legacy/python/src/fs_explorer/explore_sessions.py
Reference: legacy/python/src/fs_explorer/workflow.py
Reference: legacy/python/src/fs_explorer/server.py
*/

import type { ActionModel } from "../agent/agent.js";
import type { BlobStore } from "./library.js";
import type { SqliteStorageBackend } from "./storage.js";
import type { ToolRegistry } from "./skills.js";

export type ExploreSessionStatus = "created" | "running" | "awaiting_human" | "completed" | "error" | "closed";

export interface ExploreStreamEventPayload {
  session_id: string;
  type: string;
  sequence: number;
  timestamp: string;
  data: Record<string, unknown>;
}

export interface ExploreSessionSnapshot {
  session_id: string;
  status: ExploreSessionStatus;
  awaiting_human: boolean;
  pending_question: string | null;
  final_result: string | null;
  error: string | null;
  last_focus_anchor: Record<string, unknown> | null;
  context_budget_stats: Record<string, unknown>;
  context_state_snapshot: Record<string, unknown>;
  lazy_indexing_stats: Record<string, unknown>;
  task: string;
  document_ids: string[];
  collection_id: string | null;
  collection_ids: string[];
  db_path: string | null;
  enable_semantic: boolean;
  enable_metadata: boolean;
  created_at: string;
  updated_at: string;
}

export interface CreateExploreSessionInput {
  task: string;
  documentIds?: string[];
  collectionId?: string | null;
  collectionIds?: string[] | null;
  dbPath?: string | null;
  enableSemantic?: boolean;
  enableMetadata?: boolean;
}

export interface HumanReplyInput {
  sessionId: string;
  response: string;
}

export interface ExplorationWorkflowServiceOptions {
  storage: SqliteStorageBackend;
  model: ActionModel;
  blobStore: BlobStore;
  registry?: ToolRegistry;
  skillsRoot?: string;
  rootDirectory?: string;
  retentionMinutes?: number;
}

export interface ExplorationWorkflowStartResult {
  session_id: string;
  status: ExploreSessionStatus;
}

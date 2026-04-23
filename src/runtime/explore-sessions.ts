/*
Reference: legacy/python/src/fs_explorer/explore_sessions.py
Reference: legacy/python/src/fs_explorer/server.py
*/

import { randomUUID } from "node:crypto";

import type {
  BatchMode,
  ExploreSessionSnapshot,
  ExploreSessionStatus,
  ExploreStreamEventPayload,
} from "../types/exploration.js";

export function utcNow(): Date {
  return new Date();
}

export function isoformatUtc(value: Date): string {
  return value.toISOString();
}

export class ExploreStreamEvent {
  constructor(
    readonly sequence: number,
    readonly type: string,
    readonly timestamp: Date,
    readonly data: Record<string, unknown>,
  ) {}

  asPayload(sessionId: string): ExploreStreamEventPayload {
    return {
      session_id: sessionId,
      type: this.type,
      sequence: this.sequence,
      timestamp: isoformatUtc(this.timestamp),
      data: this.data,
    };
  }
}

export class AsyncEventQueue<T> implements AsyncIterable<T> {
  private readonly values: T[] = [];

  private readonly waiters: Array<(value: IteratorResult<T>) => void> = [];

  private closed = false;

  push(value: T): void {
    const waiter = this.waiters.shift();
    if (waiter) {
      waiter({ value, done: false });
      return;
    }
    this.values.push(value);
  }

  close(): void {
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter({ value: undefined as T, done: true });
    }
  }

  next(): Promise<IteratorResult<T>> {
    const value = this.values.shift();
    if (value !== undefined) {
      return Promise.resolve({ value, done: false });
    }
    if (this.closed) {
      return Promise.resolve({ value: undefined as T, done: true });
    }
    return new Promise((resolve) => {
      this.waiters.push(resolve);
    });
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: () => this.next(),
    };
  }
}

export class ExploreSession {
  status: ExploreSessionStatus = "created";

  pendingQuestion: string | null = null;

  finalResult: string | null = null;

  error: string | null = null;

  lastFocusAnchor: Record<string, unknown> | null = null;

  contextBudgetStats: Record<string, unknown> = {};

  contextStateSnapshot: Record<string, unknown> = {};

  lazyIndexingStats: Record<string, unknown> = {
    triggered: false,
    indexed_documents: 0,
    chunks_written: 0,
    embeddings_written: 0,
  };

  candidateDocumentSelection: Record<string, unknown> | null = null;

  documentSummaries: Array<Record<string, unknown>> = [];

  parallelDocumentLimit = 3;

  batchSummaries: Array<Record<string, unknown>> = [];

  cumulativeAnswer: string | null = null;

  readonly createdAt = utcNow();

  updatedAt = this.createdAt;

  readonly history: ExploreStreamEvent[] = [];

  readonly subscribers = new Set<AsyncEventQueue<ExploreStreamEvent>>();

  workflowTask: Promise<void> | null = null;

  runtime: unknown = null;

  private nextSequence = 1;

  constructor(
    readonly sessionId: string,
    readonly task: string,
    readonly documentIds: string[],
    readonly collectionId: string | null,
    readonly collectionIds: string[],
    readonly dbPath: string | null,
    readonly enableSemantic: boolean,
    readonly enableMetadata: boolean,
    readonly batchMode: BatchMode,
    readonly batchSize: number,
    readonly batchThreshold: number,
  ) {}

  snapshot(): ExploreSessionSnapshot {
    return {
      session_id: this.sessionId,
      status: this.status,
      awaiting_human: this.status === "awaiting_human",
      pending_question: this.pendingQuestion,
      final_result: this.finalResult,
      error: this.error,
      last_focus_anchor: this.lastFocusAnchor,
      context_budget_stats: this.contextBudgetStats,
      context_state_snapshot: this.contextStateSnapshot,
      lazy_indexing_stats: this.lazyIndexingStats,
      task: this.task,
      document_ids: [...this.documentIds],
      collection_id: this.collectionId,
      collection_ids: [...this.collectionIds],
      db_path: this.dbPath,
      enable_semantic: this.enableSemantic,
      enable_metadata: this.enableMetadata,
      candidate_document_selection: this.candidateDocumentSelection,
      document_summaries: [...this.documentSummaries],
      parallel_document_limit: this.parallelDocumentLimit,
      batch_summaries: [...this.batchSummaries],
      cumulative_answer: this.cumulativeAnswer,
      batch_mode: this.batchMode,
      batch_size: this.batchSize,
      batch_threshold: this.batchThreshold,
      created_at: isoformatUtc(this.createdAt),
      updated_at: isoformatUtc(this.updatedAt),
    };
  }

  isTerminal(): boolean {
    return ["completed", "error", "closed"].includes(this.status);
  }

  publish(eventType: string, data: Record<string, unknown>): ExploreStreamEvent {
    const event = new ExploreStreamEvent(this.nextSequence, eventType, utcNow(), data);
    this.nextSequence += 1;
    this.updatedAt = event.timestamp;
    this.history.push(event);
    for (const subscriber of this.subscribers) {
      subscriber.push(event);
    }
    return event;
  }
}

export class ExploreSessionManager {
  private readonly sessions = new Map<string, ExploreSession>();

  private readonly retentionMs: number;

  constructor(input: { retentionMinutes?: number } = {}) {
    this.retentionMs = (input.retentionMinutes ?? 15) * 60 * 1000;
  }

  createSession(input: {
    task: string;
    documentIds: string[];
    collectionId?: string | null;
    collectionIds?: string[] | null;
    dbPath?: string | null;
    enableSemantic?: boolean;
    enableMetadata?: boolean;
    batchMode?: BatchMode;
    batchSize?: number | null;
    batchThreshold?: number | null;
  }): ExploreSession {
    this.cleanup();
    const session = new ExploreSession(
      randomUUID().replaceAll("-", ""),
      input.task,
      [...input.documentIds],
      input.collectionId ?? null,
      [
        ...new Set(
          [...(input.collectionIds ?? []), ...(input.collectionId ? [input.collectionId] : [])]
            .map((item) => String(item).trim())
            .filter(Boolean),
        ),
      ],
      input.dbPath ?? null,
      Boolean(input.enableSemantic),
      Boolean(input.enableMetadata),
      input.batchMode ?? "auto",
      Math.max(Number(input.batchSize ?? 5), 1),
      Math.max(Number(input.batchThreshold ?? 10), 1),
    );
    this.sessions.set(session.sessionId, session);
    return session;
  }

  getSession(sessionId: string): ExploreSession | null {
    this.cleanup();
    return this.sessions.get(sessionId) ?? null;
  }

  subscribe(sessionId: string): {
    session: ExploreSession | null;
    queue: AsyncEventQueue<ExploreStreamEvent> | null;
    history: ExploreStreamEvent[] | null;
  } {
    const session = this.getSession(sessionId);
    if (!session) {
      return { session: null, queue: null, history: null };
    }
    const queue = new AsyncEventQueue<ExploreStreamEvent>();
    session.subscribers.add(queue);
    return { session, queue, history: [...session.history] };
  }

  unsubscribe(session: ExploreSession, queue: AsyncEventQueue<ExploreStreamEvent>): void {
    session.subscribers.delete(queue);
    queue.close();
  }

  cleanup(): void {
    const cutoff = Date.now() - this.retentionMs;
    for (const [sessionId, session] of this.sessions) {
      if (session.isTerminal() && session.updatedAt.getTime() < cutoff) {
        this.sessions.delete(sessionId);
      }
    }
  }
}

export function encodeSseEvent(sessionId: string, event: ExploreStreamEvent): string {
  return `event: ${event.type}\ndata: ${JSON.stringify(event.asPayload(sessionId))}\n\n`;
}

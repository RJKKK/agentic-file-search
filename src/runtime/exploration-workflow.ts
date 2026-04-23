/*
Reference: legacy/python/src/fs_explorer/workflow.py
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
Reference: legacy/python/src/fs_explorer/explore_sessions.py
*/

import { resolve } from "node:path";

import { createAgent, type FsExplorerAgent } from "../agent/agent.js";
import { fallbackStopAction, parseAction, toActionType, toFnArgs, type Action } from "../types/actions.js";
import type {
  CreateExploreSessionInput,
  ExplorationWorkflowServiceOptions,
  ExplorationWorkflowStartResult,
  HumanReplyInput,
} from "../types/exploration.js";
import type { DocumentScope } from "../types/library.js";
import type { StoredDocument } from "../types/storage.js";
import type { ToolRegistry } from "../types/skills.js";
import {
  createLibraryDocumentCatalog,
  materializeDocument,
  resolveDocumentScope,
} from "./document-library.js";
import { resolvePagesDirectory } from "./document-pages.js";
import { ExploreSession, ExploreSessionManager } from "./explore-sessions.js";
import { extractCitedSources, ExplorationTrace } from "./exploration-trace.js";
import { IndexSearchService } from "./index-search.js";
import { loadSkills } from "./load-skills.js";
import { buildToolRegistry } from "./registry.js";

interface ExplorationRuntimeState {
  agent: FsExplorerAgent;
  scope: DocumentScope;
  documents: StoredDocument[];
  documentNames: string[];
  collectionName: string | null;
  collectionNames: string[];
  trace: ExplorationTrace;
  stepNumber: number;
  candidatePages: Array<Record<string, unknown>>;
  readPages: Array<Record<string, unknown>>;
  stalePageRanges: Array<Record<string, unknown>>;
  pageQueryHistory: Array<Record<string, unknown>>;
  pageBoundaryLoads: Array<Record<string, unknown>>;
}

interface CandidateDocument {
  document: StoredDocument;
  candidatePages: Array<Record<string, unknown>>;
  score: number;
}

interface DocumentAgentSummary {
  document_id: string;
  document_name: string;
  candidate_pages: Array<Record<string, unknown>>;
  temporary_answer: string;
  status: "answered" | "insufficient_evidence" | "failed";
  cited_sources: string[];
  context_released: Array<Record<string, unknown>>;
  error?: string | null;
}

type ResumeReason = "start" | "tool_result" | "go_deeper" | "human_answer";

const DEFAULT_BATCH_SIZE = 5;
const DEFAULT_BATCH_THRESHOLD = 10;
const PARALLEL_DOCUMENT_LIMIT = 3;

function defaultSkillsRoot(): string {
  return resolve(process.cwd(), "skills");
}

function contextPlanPayload(action: Action): Record<string, unknown> | null {
  return action.context_plan ? (action.context_plan as Record<string, unknown>) : null;
}

function promptForStart(input: {
  task: string;
  documentLabels: string[];
  collectionNames: string[];
}): string {
  const scopeName =
    input.collectionNames.length === 1
      ? `collection '${input.collectionNames[0]}'`
      : input.collectionNames.length > 1
        ? `${input.collectionNames.length} selected collections and selected documents`
        : "selected documents";
  const documentSummary =
    input.documentLabels.length > 0
      ? input.documentLabels.map((label) => `- ${label}`).join("\n")
      : "- (none)";
  const indexHint =
    "Only work inside the selected document set. Start with `glob(directory=\"scope\")` to inspect page files, " +
    "then use `search_candidates` or scope `grep` to find candidate pages, then `read` a few page files. " +
    "If a page looks cut off at a boundary, decide whether you need the previous-page tail or next-page head, then use `page_boundary_context` before reading neighbor pages.";
  return (
    `You can only access the current ${scopeName}. The available documents are:\n\n` +
    `\`\`\`text\n${documentSummary}\n\`\`\`\n\n` +
    `The user task is: '${input.task}'. What action should you take first? ${indexHint}`
  );
}

function promptForBatchStart(input: {
  task: string;
  documentLabels: string[];
  batchIndex: number;
  batchCount: number;
  cumulativeAnswer: string | null;
}): string {
  const documentSummary =
    input.documentLabels.length > 0
      ? input.documentLabels.map((label) => `- ${label}`).join("\n")
      : "- (none)";
  const prior = input.cumulativeAnswer?.trim()
    ? `Previous cumulative answer from earlier batches:\n\n${input.cumulativeAnswer.trim()}\n\n`
    : "No previous cumulative answer exists yet.\n\n";
  return (
    `You are processing batch ${input.batchIndex} of ${input.batchCount}. ` +
    "You can only access the documents in this batch:\n\n" +
    `\`\`\`text\n${documentSummary}\n\`\`\`\n\n` +
    prior +
    `Original user task: '${input.task}'. ` +
    "Use scope glob/search_candidates/grep/read inside this batch, then stop with a batch answer. " +
    "Include citations and explicitly mention unresolved gaps for this batch."
  );
}

function promptForGoDeeper(input: { task: string; documentLabels: string[] }): string {
  const documentSummary =
    input.documentLabels.length > 0
      ? input.documentLabels.map((label) => `- ${label}`).join("\n")
      : "- (none)";
  return (
    `Stay within the selected documents below:\n\n\`\`\`text\n${documentSummary}\n\`\`\`\n\n` +
    `The user task is still: '${input.task}'. Based on what you learned, what action should you take next?`
  );
}

function promptForHumanAnswer(input: { task: string; response: string }): string {
  return (
    `Human response to your question: ${input.response}\n\n` +
    `Based on it, proceed with your exploration based on the original task: ${input.task}`
  );
}

function promptForToolResult(task: string): string {
  return (
    `The user task is still: '${task}'. ` +
      "Given the tool result and structured context you just received, " +
      "choose the next action. Do not assume the previously active pages still contain the answer. " +
      "If the active evidence now answers the task, stop with the final answer immediately. " +
      "Do not flow back to `search_candidates` just to rediscover pages that were already read. " +
      "If the current page appears incomplete, cut off, or part of a continued table/list, " +
      "decide whether the missing context is at the head or tail, then use `page_boundary_context(previous|next)` for that page before expanding to new reads. " +
      "For example, if page 49 looks like a continuation of a table from page 48, anchor on page 48 and check `page_boundary_context(next)` on page 48 before asking to read page 47-48. " +
      "If you just read a consecutive range, only inspect the first-page head and last-page tail, then use `page_boundary_context(document_id=..., start_page=..., end_page=..., direction=...)` for the needed outer boundary. " +
      "If the current range is stale or insufficient, run a fresh search or read genuinely new adjacent/candidate pages. " +
      "If repeated tool calls would be needed, stop and answer from the evidence already collected with any uncertainty noted."
  );
}

function promptForDocumentAgentStart(input: {
  task: string;
  documentLabel: string;
  candidatePages: Array<Record<string, unknown>>;
}): string {
  const candidateSummary = input.candidatePages.length
    ? input.candidatePages
        .map((page) => `- page=${String(page.page_no ?? "?")} score=${String(page.score ?? "?")} path=${String(page.file_path ?? "-")}`)
        .join("\n")
    : "- (no specific candidate pages; inspect the document narrowly and report insufficient evidence if unsupported)";
  return (
    "You are a document-level sub-agent. Only answer from this one document:\n\n" +
    `\`\`\`text\n${input.documentLabel}\n\`\`\`\n\n` +
    `Original user task: '${input.task}'.\n\n` +
    `Candidate pages for this document:\n${candidateSummary}\n\n` +
    "Use scope glob/search_candidates/grep/read inside this document only. " +
    "Prefer the listed candidate pages first. If a page boundary looks incomplete, use page_boundary_context before reading adjacent pages. " +
    "Stop with a temporary answer for this document only. If this document does not support the answer, say so explicitly with any evidence or uncertainty."
  );
}

function promptForCandidateSelection(input: {
  task: string;
  candidates: CandidateDocument[];
}): string {
  const candidateSummary = input.candidates
    .map((candidate) => {
      const label =
        candidate.document.original_filename ||
        candidate.document.relative_path ||
        candidate.document.id;
      const pages = candidate.candidatePages
        .map((page) => `page=${String(page.page_no ?? "?")} score=${String(page.score ?? "?")} snippet=${String(page.snippet ?? "").slice(0, 180)}`)
        .join("\n  ");
      return `- doc_id=${candidate.document.id} name=${label} total_score=${candidate.score}\n  ${pages}`;
    })
    .join("\n");
  return (
    "Select the candidate documents that should receive document-level sub-agents.\n\n" +
    `Original user task: ${input.task}\n\n` +
    `Candidate documents from page retrieval:\n${candidateSummary || "- (none)"}\n\n` +
    "Return exactly JSON with this shape and no markdown:\n" +
    `{"selected_documents":[{"document_id":"...","reason":"..."}]}\n` +
    "Only select document_id values present in the candidate list. Prefer documents whose snippets could answer or materially contradict the task."
  );
}

function promptForFinalSynthesis(input: {
  task: string;
  summaries: DocumentAgentSummary[];
}): string {
  const summaryText = input.summaries.length
    ? input.summaries
        .map(
          (summary, index) =>
            `## Document ${index + 1}: ${summary.document_name} (${summary.document_id})\n` +
            `status: ${summary.status}\n` +
            `cited_sources: ${summary.cited_sources.join(", ") || "-"}\n` +
            `temporary_answer:\n${summary.temporary_answer.trim() || "(empty)"}`,
        )
        .join("\n\n")
    : "No candidate document summaries were produced.";
  return (
    "Synthesize the final answer using only the document-level temporary answers below. Do not call tools.\n\n" +
    `Original user task: ${input.task}\n\n` +
    `${summaryText}\n\n` +
    "Return exactly one JSON Action object with final_result. The final answer should merge consistent findings, preserve citations, mention conflicts, and mention candidate documents with insufficient evidence."
  );
}

export class ExplorationWorkflowService {
  readonly sessions: ExploreSessionManager;

  private cachedRegistry: ToolRegistry | null = null;

  constructor(private readonly options: ExplorationWorkflowServiceOptions) {
    this.sessions = new ExploreSessionManager({
      retentionMinutes: options.retentionMinutes,
    });
    this.cachedRegistry = options.registry ?? null;
  }

  async startSession(input: CreateExploreSessionInput): Promise<ExplorationWorkflowStartResult> {
    const task = input.task.trim();
    if (!task) {
      throw new Error("No task provided");
    }
    const scope = resolveDocumentScope({
      storage: this.options.storage,
      documentIds: input.documentIds ?? [],
      collectionId: input.collectionId ?? null,
      collectionIds: input.collectionIds ?? [],
    });
    if (scope.isEmpty) {
      throw new Error("At least one document or one collection must be selected.");
    }
    const session = this.sessions.createSession({
      task,
      documentIds: scope.documentIds,
      collectionId: scope.collections[0]?.id ?? input.collectionId ?? null,
      collectionIds: scope.collections.map((collection) => collection.id),
        dbPath: input.dbPath ?? null,
        enableSemantic: Boolean(input.enableSemantic),
        enableMetadata: Boolean(input.enableMetadata),
        batchMode: input.batchMode ?? "auto",
        batchSize: input.batchSize ?? DEFAULT_BATCH_SIZE,
        batchThreshold: input.batchThreshold ?? DEFAULT_BATCH_THRESHOLD,
      });
    session.workflowTask = this.runSession(session, "start");
    return {
      session_id: session.sessionId,
      status: session.status,
    };
  }

  getSession(sessionId: string): ExploreSession | null {
    return this.sessions.getSession(sessionId);
  }

  async replyToSession(input: HumanReplyInput): Promise<ExplorationWorkflowStartResult> {
    const session = this.sessions.getSession(input.sessionId);
    if (!session) {
      throw new Error("Session not found");
    }
    if (session.status !== "awaiting_human") {
      throw new Error("Session is not waiting for human input");
    }
    const response = input.response.trim();
    if (!response) {
      throw new Error("Response cannot be empty");
    }
    session.status = "running";
    session.pendingQuestion = null;
    session.updatedAt = new Date();
    session.workflowTask = this.runSession(session, "human_answer", response);
    return {
      session_id: session.sessionId,
      status: session.status,
    };
  }

  async waitForSession(sessionId: string): Promise<void> {
    const session = this.sessions.getSession(sessionId);
    await session?.workflowTask;
  }

  private async registry(): Promise<ToolRegistry> {
    if (this.cachedRegistry) {
      return this.cachedRegistry;
    }
    const skills = await loadSkills(this.options.skillsRoot ?? defaultSkillsRoot());
    this.cachedRegistry = buildToolRegistry(skills);
    return this.cachedRegistry;
  }

  private async createRuntime(
    session: ExploreSession,
    overrideDocumentIds: string[] | null = null,
  ): Promise<ExplorationRuntimeState> {
    const scope = resolveDocumentScope({
      storage: this.options.storage,
      documentIds: overrideDocumentIds ?? session.documentIds,
      collectionId: overrideDocumentIds ? null : session.collectionId,
      collectionIds: overrideDocumentIds ? [] : session.collectionIds,
    });
    if (scope.isEmpty) {
      throw new Error("Question answering requires at least one selected document.");
    }

    const documents: StoredDocument[] = [];
    for (const document of scope.documents) {
      documents.push(
        await materializeDocument({
          storage: this.options.storage,
          blobStore: this.options.blobStore,
          document,
        }),
      );
    }

    const documentNames = documents.map((document) => {
      const pagesDir = document.pages_prefix
        ? resolvePagesDirectory({
            blobStore: this.options.blobStore,
            pagesPrefix: document.pages_prefix,
          })
        : "";
      return (
        `${document.original_filename || document.relative_path || document.id} | ` +
        `source=${document.absolute_path} | ` +
        `pages_dir=${pagesDir} | ` +
        `page_count=${document.page_count || 0}`
      );
    });

    const candidatePages: Array<Record<string, unknown>> = [];
    const readPages: Array<Record<string, unknown>> = [];
    const stalePageRanges: Array<Record<string, unknown>> = [];
    const pageQueryHistory: Array<Record<string, unknown>> = [];
    const pageBoundaryLoads: Array<Record<string, unknown>> = [];
    const emitRuntimeEvent = (eventType: string, data: Record<string, unknown>): void => {
      if (eventType === "candidate_pages_found") {
        candidatePages.push({ ...data });
        pageQueryHistory.push({
          document_id: data.document_id,
          candidate_pages: Array.isArray(data.candidate_pages) ? data.candidate_pages : [],
        });
      } else if (eventType === "pages_read") {
        readPages.push({ ...data });
      } else if (eventType === "stale_page_range_detected") {
        stalePageRanges.push({ ...data });
      } else if (eventType === "page_boundary_loaded") {
        pageBoundaryLoads.push({ ...data });
      }
      session.publish(eventType, data);
    };
    const indexSearch = new IndexSearchService({
      storage: this.options.storage,
      blobStore: this.options.blobStore,
      documentIds: scope.documentIds,
      collectionId: scope.collection?.id ?? null,
      collectionIds: scope.collections.map((collection) => collection.id),
      scopeLabel:
        scope.collections.length > 0
          ? scope.collections.map((collection) => collection.name).join(", ")
          : `${scope.documentIds.length} selected documents`,
      emitRuntimeEvent,
    });

    const agent = createAgent(
      {
        model: this.options.model,
        documentCatalog: createLibraryDocumentCatalog({
          storage: this.options.storage,
          blobStore: this.options.blobStore,
        }),
        indexSearch,
        emitRuntimeEvent,
      },
      await this.registry(),
    );

    return {
      agent,
      scope,
      documents,
      documentNames,
      collectionName: scope.collection?.name ?? null,
      collectionNames: scope.collections.map((collection) => collection.name),
      trace: new ExplorationTrace(this.options.rootDirectory ?? process.cwd()),
      stepNumber: 0,
      candidatePages,
      readPages,
      stalePageRanges,
      pageQueryHistory,
      pageBoundaryLoads,
    };
  }

  private async runtimeFor(session: ExploreSession): Promise<ExplorationRuntimeState> {
    if (session.runtime) {
      return session.runtime as ExplorationRuntimeState;
    }
    const runtime = await this.createRuntime(session);
    session.runtime = runtime;
    return runtime;
  }

  private async runSession(
    session: ExploreSession,
    resumeReason: ResumeReason,
    humanResponse = "",
  ): Promise<void> {
    try {
      const runtime = await this.runtimeFor(session);
      if (resumeReason === "start") {
        session.status = "running";
        session.updatedAt = new Date();
        session.publish("start", {
          task: session.task,
          document_ids: [...runtime.scope.documentIds],
          collection_id: runtime.scope.collection?.id ?? null,
          collection_ids: runtime.scope.collections.map((collection) => collection.id),
          document_names: [...runtime.documentNames],
          batch_mode: session.batchMode,
          batch_size: session.batchSize,
          batch_threshold: session.batchThreshold,
        });
        if (this.shouldRunBatch(session, runtime)) {
          await this.runParallelDocumentSession(session, runtime);
          return;
        }
        runtime.agent.configureTask(
          promptForStart({
            task: session.task,
            documentLabels: runtime.documentNames,
            collectionNames: runtime.collectionNames,
          }),
          runtime.documents.map((document) => ({
            documentId: document.id,
            label: document.original_filename || document.relative_path || document.id,
            filePath: document.absolute_path,
          })),
          { rawHistory: true },
        );
        session.publish("context_scope_updated", {
          context_scope: runtime.agent.getContextState().snapshot().context_scope,
        });
      } else if (resumeReason === "human_answer") {
        runtime.agent.configureTask(
          promptForHumanAnswer({
            task: session.task,
            response: humanResponse,
          }),
          [],
          { rawHistory: true },
        );
      }

      await this.processActions(session, runtime);
    } catch (error) {
      this.failSession(session, error);
    }
  }

  private shouldRunBatch(session: ExploreSession, runtime: ExplorationRuntimeState): boolean {
    if (session.batchMode === "off") {
      return false;
    }
    if (session.batchMode === "force") {
      return true;
    }
    return runtime.scope.documentIds.length > session.batchThreshold;
  }

  private chunkDocuments(documents: StoredDocument[], batchSize: number): StoredDocument[][] {
    const ordered = [...documents].sort(
      (left, right) =>
        (left.original_filename || left.relative_path || left.id)
          .toLowerCase()
          .localeCompare((right.original_filename || right.relative_path || right.id).toLowerCase()) ||
        left.id.localeCompare(right.id),
    );
    const chunks: StoredDocument[][] = [];
    for (let index = 0; index < ordered.length; index += batchSize) {
      chunks.push(ordered.slice(index, index + batchSize));
    }
    return chunks;
  }

  private async runParallelDocumentSession(
    session: ExploreSession,
    runtime: ExplorationRuntimeState,
  ): Promise<void> {
    session.parallelDocumentLimit = PARALLEL_DOCUMENT_LIMIT;
    const candidates = await this.discoverCandidateDocuments(session, runtime);
    const selected = await this.selectCandidateDocuments(session, candidates);
    session.candidateDocumentSelection = {
      strategy: "retrieval_plus_llm",
      query: session.task,
      candidate_count: candidates.length,
      selected_document_ids: selected.map((candidate) => candidate.document.id),
      selected_count: selected.length,
    };
    session.publish("candidate_documents_found", {
      query: session.task,
      strategy: "retrieval_plus_llm",
      candidate_documents: candidates.map((candidate) => this.candidateDocumentPayload(candidate)),
      selected_documents: selected.map((candidate) => this.candidateDocumentPayload(candidate)),
      parallel_document_limit: PARALLEL_DOCUMENT_LIMIT,
    });

    if (selected.length === 0) {
      const finalDraft = await this.synthesizeFinalAnswer(session, runtime, []);
      await this.completeSession(session, runtime, finalDraft);
      return;
    }

    const summaries = await this.runDocumentAgentsInParallel(session, selected);
    session.documentSummaries = summaries.map((summary) => ({ ...summary }));
    session.cumulativeAnswer = summaries
      .map(
        (summary, index) =>
          `## Document ${index + 1}: ${summary.document_name}\n\n${summary.temporary_answer.trim()}`,
      )
      .join("\n\n");
    session.publish("cumulative_answer_updated", {
      cumulative_answer: session.cumulativeAnswer,
      cited_sources: extractCitedSources(session.cumulativeAnswer),
      document_count: summaries.length,
    });

    const finalDraft = await this.synthesizeFinalAnswer(session, runtime, summaries);
    await this.completeSession(session, runtime, finalDraft);
  }

  private async discoverCandidateDocuments(
    session: ExploreSession,
    runtime: ExplorationRuntimeState,
  ): Promise<CandidateDocument[]> {
    const search = new IndexSearchService({
      storage: this.options.storage,
      blobStore: this.options.blobStore,
      documentIds: runtime.scope.documentIds,
      collectionId: runtime.scope.collection?.id ?? null,
      collectionIds: runtime.scope.collections.map((collection) => collection.id),
      scopeLabel: runtime.scope.collections.length
        ? runtime.scope.collections.map((collection) => collection.name).join(", ")
        : `${runtime.scope.documentIds.length} selected documents`,
    });
    const result = await search.searchPagesAcrossScope(session.task, {
      maxHitsPerDocument: 3,
      maxTotalHits: Math.max(24, runtime.scope.documentIds.length * 3),
      regex: false,
    });
    const byDocument = new Map<string, CandidateDocument>();
    for (const hit of result.hits) {
      const document = runtime.documents.find((item) => item.id === hit.doc_id);
      if (!document) {
        continue;
      }
      const candidate =
        byDocument.get(document.id) ??
        {
          document,
          candidatePages: [],
          score: 0,
        };
      candidate.score += Number(hit.score ?? 0);
      candidate.candidatePages.push({
        document_id: hit.doc_id,
        page_no: hit.source_unit_no,
        file_path: hit.absolute_path,
        score: hit.score,
        snippet: hit.text,
      });
      byDocument.set(document.id, candidate);
      runtime.candidatePages.push({
        document_id: hit.doc_id,
        page_no: hit.source_unit_no,
        file_path: hit.absolute_path,
        score: hit.score,
        query: session.task,
      });
    }
    return [...byDocument.values()].sort(
      (left, right) =>
        right.score - left.score ||
        (left.document.original_filename || left.document.relative_path || left.document.id).localeCompare(
          right.document.original_filename || right.document.relative_path || right.document.id,
        ),
    );
  }

  private async selectCandidateDocuments(
    session: ExploreSession,
    candidates: CandidateDocument[],
  ): Promise<CandidateDocument[]> {
    if (candidates.length === 0) {
      return [];
    }
    let selectedIds: string[] = [];
    try {
      const rawSelection = await this.options.model.generateAction({
        systemPrompt:
          "You are the coordinator agent selecting which candidate documents deserve document-level sub-agents.",
        messages: [
          {
            role: "user",
            content: promptForCandidateSelection({ task: session.task, candidates }),
          },
        ],
        task: session.task,
        availableTools: [],
      });
      selectedIds = this.parseSelectedDocumentIds(rawSelection);
    } catch {
      selectedIds = [];
    }
    const allowed = new Set(candidates.map((candidate) => candidate.document.id));
    const filteredIds = [...new Set(selectedIds.filter((id) => allowed.has(id)))];
    if (filteredIds.length === 0) {
      return [];
    }
    return filteredIds
      .map((id) => candidates.find((candidate) => candidate.document.id === id))
      .filter((candidate): candidate is CandidateDocument => Boolean(candidate));
  }

  private parseSelectedDocumentIds(input: unknown): string[] {
    const tryCandidate = (candidate: unknown): string[] => {
      if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
        return [];
      }
      const record = candidate as Record<string, unknown>;
      const selected = Array.isArray(record.selected_documents)
        ? record.selected_documents
        : Array.isArray(record.selectedDocuments)
          ? record.selectedDocuments
          : [];
      return selected
        .map((item) => {
          if (typeof item === "string") {
            return item;
          }
          if (item && typeof item === "object" && !Array.isArray(item)) {
            return String((item as Record<string, unknown>).document_id ?? (item as Record<string, unknown>).documentId ?? "");
          }
          return "";
        })
        .map((item) => item.trim())
        .filter(Boolean);
    };

    const direct = tryCandidate(input);
    if (direct.length > 0) {
      return direct;
    }
    if (typeof input !== "string") {
      return [];
    }
    try {
      return tryCandidate(JSON.parse(input));
    } catch {
      const match = input.match(/\{[\s\S]*\}/);
      if (!match) {
        return [];
      }
      try {
        return tryCandidate(JSON.parse(match[0]));
      } catch {
        return [];
      }
    }
  }

  private async runDocumentAgentsInParallel(
    session: ExploreSession,
    candidates: CandidateDocument[],
  ): Promise<DocumentAgentSummary[]> {
    const summaries: DocumentAgentSummary[] = new Array(candidates.length);
    let nextIndex = 0;
    const worker = async (): Promise<void> => {
      while (nextIndex < candidates.length) {
        const index = nextIndex;
        nextIndex += 1;
        summaries[index] = await this.runSingleDocumentAgent(session, candidates[index]!, index + 1, candidates.length);
      }
    };
    const workers = Array.from(
      { length: Math.min(PARALLEL_DOCUMENT_LIMIT, candidates.length) },
      () => worker(),
    );
    await Promise.all(workers);
    return summaries.filter(Boolean);
  }

  private async runSingleDocumentAgent(
    session: ExploreSession,
    candidate: CandidateDocument,
    documentIndex: number,
    documentCount: number,
  ): Promise<DocumentAgentSummary> {
    const documentName =
      candidate.document.original_filename ||
      candidate.document.relative_path ||
      candidate.document.id;
    let runtime: ExplorationRuntimeState | null = null;
    try {
      runtime = await this.createRuntime(session, [candidate.document.id]);
      session.publish("batch_started", {
        batch_index: documentIndex,
        batch_count: documentCount,
        document_ids: [candidate.document.id],
        document_names: [documentName],
        previous_cumulative_answer: session.cumulativeAnswer || null,
        document_id: candidate.document.id,
      });
      session.publish("document_agent_started", {
        document_index: documentIndex,
        document_count: documentCount,
        document_id: candidate.document.id,
        document_name: documentName,
        candidate_pages: candidate.candidatePages,
      });
      runtime.agent.configureTask(
        promptForDocumentAgentStart({
          task: session.task,
          documentLabel: runtime.documentNames[0] ?? documentName,
          candidatePages: candidate.candidatePages,
        }),
        runtime.documents.map((document) => ({
          documentId: document.id,
          label: document.original_filename || document.relative_path || document.id,
          filePath: document.absolute_path,
        })),
        { rawHistory: true },
      );
      session.publish("context_scope_updated", {
        document_id: candidate.document.id,
        context_scope: runtime.agent.getContextState().snapshot().context_scope,
      });
      const draft = await this.processDocumentActions(session, runtime, candidate.document.id, documentIndex);
      let temporaryAnswer = "";
      try {
        for await (const chunk of runtime.agent.streamFinalAnswer(draft)) {
          const deltaText = String(chunk ?? "");
          if (!deltaText) {
            continue;
          }
          temporaryAnswer += deltaText;
          session.publish("document_answer_delta", {
            document_index: documentIndex,
            document_id: candidate.document.id,
            delta_text: deltaText,
            accumulated_text: temporaryAnswer,
          });
        }
      } catch (error) {
        session.publish("answer_stream_failed", {
          document_index: documentIndex,
          document_id: candidate.document.id,
          message: error instanceof Error ? error.message : String(error),
        });
      }
      if (!temporaryAnswer.trim()) {
        temporaryAnswer = draft;
      }
      const citedSources = extractCitedSources(temporaryAnswer);
      const contextRelease = runtime.agent.getContextState().releaseDocumentEvidence({
        documentId: candidate.document.id,
        reason: `document agent ${documentIndex} completed`,
      });
      const status = this.classifyTemporaryAnswer(temporaryAnswer);
      const summary: DocumentAgentSummary = {
        document_id: candidate.document.id,
        document_name: documentName,
        candidate_pages: candidate.candidatePages,
        temporary_answer: temporaryAnswer,
        status,
        cited_sources: citedSources,
        context_released: [contextRelease],
        error: null,
      };
      session.publish("document_answer_done", {
        document_index: documentIndex,
        document_id: candidate.document.id,
        temporary_answer: temporaryAnswer,
        cited_sources: citedSources,
        status,
      });
      session.publish("batch_answer_done", {
        batch_index: documentIndex,
        batch_answer: temporaryAnswer,
        cited_sources: citedSources,
        document_id: candidate.document.id,
      });
      session.publish("batch_context_released", {
        batch_index: documentIndex,
        releases: [contextRelease],
        document_id: candidate.document.id,
      });
      session.publish("document_agent_completed", summary as unknown as Record<string, unknown>);
      session.documentSummaries.push(summary as unknown as Record<string, unknown>);
      const batchSummary = {
        batch_index: documentIndex,
        batch_count: documentCount,
        document_ids: [candidate.document.id],
        document_names: [documentName],
        batch_answer: temporaryAnswer,
        cited_sources: citedSources,
        context_released: [contextRelease],
        document_summary: summary,
      };
      session.batchSummaries.push(batchSummary);
      session.publish("batch_completed", batchSummary);
      return summary;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const summary: DocumentAgentSummary = {
        document_id: candidate.document.id,
        document_name: documentName,
        candidate_pages: candidate.candidatePages,
        temporary_answer: `This document could not be processed: ${message}`,
        status: "failed",
        cited_sources: [],
        context_released: [],
        error: message,
      };
      session.documentSummaries.push(summary as unknown as Record<string, unknown>);
      session.batchSummaries.push({
        batch_index: documentIndex,
        batch_count: documentCount,
        document_ids: [candidate.document.id],
        document_names: [documentName],
        batch_answer: summary.temporary_answer,
        cited_sources: [],
        context_released: [],
        document_summary: summary,
      });
      session.publish("document_agent_completed", summary as unknown as Record<string, unknown>);
      return summary;
    }
  }

  private classifyTemporaryAnswer(answer: string): "answered" | "insufficient_evidence" | "failed" {
    if (/insufficient|not enough|no evidence|does not support|未找到|证据不足|不支持|无法确认/i.test(answer)) {
      return "insufficient_evidence";
    }
    return "answered";
  }

  private async processDocumentActions(
    session: ExploreSession,
    runtime: ExplorationRuntimeState,
    documentId: string,
    documentIndex: number,
  ): Promise<string> {
    for (let actionCount = 0; actionCount < 30; actionCount += 1) {
      const action = await runtime.agent.takeAction();
      const planResult = runtime.agent.consumeLastContextPlanResult();
      if (planResult) {
        session.publish(planResult.applied ? "context_scope_updated" : "context_compacted", {
          document_index: documentIndex,
          document_id: documentId,
          operation: planResult.operation,
          applied: planResult.applied,
          ...planResult.payload,
        });
      }

      const actionType = toActionType(action);
      if (actionType === "stop" && "final_result" in action.action) {
        return action.action.final_result;
      }

      runtime.stepNumber += 1;
      if (actionType === "toolcall" && "tool_name" in action.action) {
        const toolInput = toFnArgs(action.action);
        runtime.trace.recordToolCall({
          stepNumber: runtime.stepNumber,
          toolName: action.action.tool_name,
          toolInput,
          resolvedDocumentPath: null,
        });
        session.publish("document_agent_tool_call", {
          document_index: documentIndex,
          document_id: documentId,
          step: runtime.stepNumber,
          tool_name: action.action.tool_name,
          tool_input: toolInput,
          reason: action.reason,
          context_plan: contextPlanPayload(action),
        });
        await runtime.agent.callTool(action.action.tool_name, toolInput);
        runtime.agent.configureTask(promptForToolResult(session.task), [], { rawHistory: true });
        continue;
      }

      if (actionType === "askhuman" && "question" in action.action) {
        return `Insufficient evidence in this document without human clarification: ${action.action.question}`;
      }

      if (actionType === "godeeper" && "directory" in action.action) {
        runtime.trace.recordGoDeeper({
          stepNumber: runtime.stepNumber,
          directory: action.action.directory,
        });
        runtime.agent.configureTask(
          promptForGoDeeper({
            task: session.task,
            documentLabels: runtime.documentNames,
          }),
          [],
          { rawHistory: true },
        );
        continue;
      }
    }
    return runtime.agent.getContextState().bestEffortAnswer();
  }

  private async synthesizeFinalAnswer(
    session: ExploreSession,
    runtime: ExplorationRuntimeState,
    summaries: DocumentAgentSummary[],
  ): Promise<string> {
    session.publish("final_synthesis_started", {
      document_count: summaries.length,
      document_ids: summaries.map((summary) => summary.document_id),
    });
    let finalDraft = "";
    try {
      const raw = await this.options.model.generateAction({
        systemPrompt: "You are the coordinator agent synthesizing document-level temporary answers into the final answer.",
        messages: [
          {
            role: "user",
            content: promptForFinalSynthesis({ task: session.task, summaries }),
          },
        ],
        task: session.task,
        availableTools: [],
      });
      finalDraft = this.parseFinalResult(raw);
    } catch (error) {
      finalDraft = `Could not synthesize final answer: ${error instanceof Error ? error.message : String(error)}`;
    }
    if (!finalDraft.trim()) {
      finalDraft =
        summaries.length > 0
          ? summaries.map((summary) => summary.temporary_answer).join("\n\n")
          : "No candidate documents or supporting evidence were found for the requested task.";
    }
    session.publish("final_synthesis_delta", {
      delta_text: finalDraft,
      accumulated_text: finalDraft,
    });
    session.publish("final_synthesis_done", {
      final_result: finalDraft,
      cited_sources: extractCitedSources(finalDraft),
    });
    return finalDraft;
  }

  private parseFinalResult(raw: unknown): string {
    try {
      const action = parseAction(raw);
      if ("final_result" in action.action) {
        return action.action.final_result;
      }
    } catch {
      // Fall through to plain text/object fallbacks.
    }
    if (typeof raw === "string") {
      const fallback = fallbackStopAction(raw);
      return fallback && "final_result" in fallback.action ? fallback.action.final_result : raw;
    }
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
      const record = raw as Record<string, unknown>;
      if (typeof record.final_result === "string") {
        return record.final_result;
      }
      if (typeof record.finalResult === "string") {
        return record.finalResult;
      }
    }
    return "";
  }

  private candidateDocumentPayload(candidate: CandidateDocument): Record<string, unknown> {
    return {
      document_id: candidate.document.id,
      document_name:
        candidate.document.original_filename ||
        candidate.document.relative_path ||
        candidate.document.id,
      score: candidate.score,
      candidate_pages: candidate.candidatePages,
    };
  }

  private async runBatchSession(
    session: ExploreSession,
    runtime: ExplorationRuntimeState,
  ): Promise<void> {
    const batches = this.chunkDocuments(runtime.documents, Math.max(session.batchSize, 1));
    let cumulativeAnswer = session.cumulativeAnswer ?? "";
    for (const [batchIndexZero, documents] of batches.entries()) {
      const batchIndex = batchIndexZero + 1;
      const batchRuntime = await this.createRuntime(session, documents.map((document) => document.id));
      session.publish("batch_started", {
        batch_index: batchIndex,
        batch_count: batches.length,
        document_ids: documents.map((document) => document.id),
        document_names: [...batchRuntime.documentNames],
        previous_cumulative_answer: cumulativeAnswer || null,
      });
      batchRuntime.agent.configureTask(
        promptForBatchStart({
          task: session.task,
          documentLabels: batchRuntime.documentNames,
          batchIndex,
          batchCount: batches.length,
          cumulativeAnswer: cumulativeAnswer || null,
        }),
        batchRuntime.documents.map((document) => ({
          documentId: document.id,
          label: document.original_filename || document.relative_path || document.id,
          filePath: document.absolute_path,
        })),
        { rawHistory: true },
      );
      session.publish("context_scope_updated", {
        batch_index: batchIndex,
        context_scope: batchRuntime.agent.getContextState().snapshot().context_scope,
      });

      const batchDraft = await this.processBatchActions(session, batchRuntime, batchIndex);
      let batchAnswer = "";
      try {
        for await (const chunk of batchRuntime.agent.streamFinalAnswer(batchDraft)) {
          const deltaText = String(chunk ?? "");
          if (!deltaText) {
            continue;
          }
          batchAnswer += deltaText;
          session.publish("batch_answer_delta", {
            batch_index: batchIndex,
            delta_text: deltaText,
            accumulated_text: batchAnswer,
          });
        }
      } catch (error) {
        session.publish("answer_stream_failed", {
          batch_index: batchIndex,
          message: error instanceof Error ? error.message : String(error),
        });
      }
      if (!batchAnswer.trim()) {
        batchAnswer = batchDraft;
      }
      const citedSources = extractCitedSources(batchAnswer);
      session.publish("batch_answer_done", {
        batch_index: batchIndex,
        batch_answer: batchAnswer,
        cited_sources: citedSources,
      });

      const releases = documents.map((document) =>
        batchRuntime.agent.getContextState().releaseDocumentEvidence({
          documentId: document.id,
          reason: `batch ${batchIndex} completed`,
        }),
      );
      session.publish("batch_context_released", {
        batch_index: batchIndex,
        releases,
      });

      cumulativeAnswer = [
        cumulativeAnswer.trim(),
        `## Batch ${batchIndex}/${batches.length}`,
        batchAnswer.trim(),
      ]
        .filter(Boolean)
        .join("\n\n");
      session.cumulativeAnswer = cumulativeAnswer;
      session.publish("cumulative_answer_updated", {
        batch_index: batchIndex,
        cumulative_answer: cumulativeAnswer,
        cited_sources: extractCitedSources(cumulativeAnswer),
      });

      const summary = {
        batch_index: batchIndex,
        batch_count: batches.length,
        document_ids: documents.map((document) => document.id),
        document_names: [...batchRuntime.documentNames],
        batch_answer: batchAnswer,
        cited_sources: citedSources,
        context_released: releases,
      };
      session.batchSummaries.push(summary);
      session.publish("batch_completed", summary);
    }

    await this.completeSession(session, runtime, session.cumulativeAnswer || "No batch answer was produced.");
  }

  private async processBatchActions(
    session: ExploreSession,
    runtime: ExplorationRuntimeState,
    batchIndex: number,
  ): Promise<string> {
    for (let actionCount = 0; actionCount < 30; actionCount += 1) {
      const action = await runtime.agent.takeAction();
      const planResult = runtime.agent.consumeLastContextPlanResult();
      if (planResult) {
        session.publish(planResult.applied ? "context_scope_updated" : "context_compacted", {
          batch_index: batchIndex,
          operation: planResult.operation,
          applied: planResult.applied,
          ...planResult.payload,
        });
      }

      const actionType = toActionType(action);
      if (actionType === "stop" && "final_result" in action.action) {
        return action.action.final_result;
      }

      runtime.stepNumber += 1;
      if (actionType === "toolcall" && "tool_name" in action.action) {
        const toolInput = toFnArgs(action.action);
        runtime.trace.recordToolCall({
          stepNumber: runtime.stepNumber,
          toolName: action.action.tool_name,
          toolInput,
          resolvedDocumentPath: null,
        });
        session.publish("tool_call", {
          batch_index: batchIndex,
          step: runtime.stepNumber,
          tool_name: action.action.tool_name,
          tool_input: toolInput,
          reason: action.reason,
          context_plan: contextPlanPayload(action),
        });
        await runtime.agent.callTool(action.action.tool_name, toolInput);
        runtime.agent.configureTask(promptForToolResult(session.task), [], { rawHistory: true });
        continue;
      }

      if (actionType === "askhuman" && "question" in action.action) {
        session.status = "awaiting_human";
        session.pendingQuestion = action.action.question;
        session.updatedAt = new Date();
        session.publish("ask_human", {
          batch_index: batchIndex,
          step: runtime.stepNumber,
          question: action.action.question,
          reason: action.reason,
          context_plan: contextPlanPayload(action),
        });
        throw new Error("Batch workflow cannot resume ask_human inside a batch yet.");
      }

      if (actionType === "godeeper" && "directory" in action.action) {
        session.publish("go_deeper", {
          batch_index: batchIndex,
          step: runtime.stepNumber,
          directory: action.action.directory,
          reason: action.reason,
          context_plan: contextPlanPayload(action),
        });
        runtime.agent.configureTask(
          promptForGoDeeper({
            task: session.task,
            documentLabels: runtime.documentNames,
          }),
          [],
          { rawHistory: true },
        );
        continue;
      }
    }
    return runtime.agent.getContextState().bestEffortAnswer();
  }

  private async processActions(
    session: ExploreSession,
    runtime: ExplorationRuntimeState,
  ): Promise<void> {
    while (true) {
      const action = await runtime.agent.takeAction();
      const planResult = runtime.agent.consumeLastContextPlanResult();
      if (planResult) {
        session.publish(planResult.applied ? "context_scope_updated" : "context_compacted", {
          operation: planResult.operation,
          applied: planResult.applied,
          ...planResult.payload,
        });
      }

      const actionType = toActionType(action);
      if (actionType === "stop" && "final_result" in action.action) {
        await this.completeSession(session, runtime, action.action.final_result);
        return;
      }

      runtime.stepNumber += 1;
      if (actionType === "toolcall" && "tool_name" in action.action) {
        const toolInput = toFnArgs(action.action);
        let resolvedDocumentPath: string | null = null;
        if (action.action.tool_name === "get_document") {
          const docId = typeof toolInput.doc_id === "string" ? toolInput.doc_id : "";
          const document = docId ? this.options.storage.getDocument(docId) : null;
          if (document && !document.is_deleted) {
            resolvedDocumentPath = document.absolute_path;
          }
        }
        runtime.trace.recordToolCall({
          stepNumber: runtime.stepNumber,
          toolName: action.action.tool_name,
          toolInput,
          resolvedDocumentPath,
        });
        session.publish("tool_call", {
          step: runtime.stepNumber,
          tool_name: action.action.tool_name,
          tool_input: toolInput,
          reason: action.reason,
          context_plan: contextPlanPayload(action),
        });
        await runtime.agent.callTool(action.action.tool_name, toolInput);
        runtime.agent.configureTask(promptForToolResult(session.task), [], { rawHistory: true });
        continue;
      }

      if (actionType === "godeeper" && "directory" in action.action) {
        runtime.trace.recordGoDeeper({
          stepNumber: runtime.stepNumber,
          directory: action.action.directory,
        });
        session.publish("go_deeper", {
          step: runtime.stepNumber,
          directory: action.action.directory,
          reason: action.reason,
          context_plan: contextPlanPayload(action),
        });
        runtime.agent.configureTask(
          promptForGoDeeper({
            task: session.task,
            documentLabels: runtime.documentNames,
          }),
          [],
          { rawHistory: true },
        );
        continue;
      }

      if (actionType === "askhuman" && "question" in action.action) {
        session.status = "awaiting_human";
        session.pendingQuestion = action.action.question;
        session.updatedAt = new Date();
        session.publish("ask_human", {
          step: runtime.stepNumber,
          question: action.action.question,
          reason: action.reason,
          context_plan: contextPlanPayload(action),
        });
        return;
      }

      this.failSession(session, new Error("Could not produce action to take"));
      return;
    }
  }

  private async completeSession(
    session: ExploreSession,
    runtime: ExplorationRuntimeState,
    finalResult: string,
  ): Promise<void> {
    let resolvedFinalResult = finalResult;
    session.publish("answer_started", {
      draft_final_result: finalResult,
    });
    let streamedFinalResult = "";
    try {
      for await (const chunk of runtime.agent.streamFinalAnswer(finalResult)) {
        const deltaText = String(chunk ?? "");
        if (!deltaText) {
          continue;
        }
        streamedFinalResult += deltaText;
        session.publish("answer_delta", {
          delta_text: deltaText,
          accumulated_text: streamedFinalResult,
        });
      }
    } catch (error) {
      session.publish("answer_stream_failed", {
        message: error instanceof Error ? error.message : String(error),
      });
    }
    if (streamedFinalResult.trim()) {
      resolvedFinalResult = streamedFinalResult;
    }
    session.publish("answer_done", {
      final_result: resolvedFinalResult,
    });

    const citedSources = extractCitedSources(resolvedFinalResult);
    runtime.agent.getContextState().markCitedSources(citedSources);
    const contextStateSnapshot = runtime.agent.getContextState().snapshot();
    const contextBudgetStats = runtime.agent.getContextBudgetStats();
    const lastFocusAnchor = contextStateSnapshot.context_scope as Record<string, unknown>;
    session.status = "completed";
    session.pendingQuestion = null;
    session.finalResult = resolvedFinalResult;
    session.error = null;
    session.lastFocusAnchor = lastFocusAnchor;
    session.contextBudgetStats = contextBudgetStats;
    session.contextStateSnapshot = contextStateSnapshot as unknown as Record<string, unknown>;
    session.updatedAt = new Date();
    session.publish("complete", {
      final_result: resolvedFinalResult,
      error: null,
      stats: {
        steps: runtime.stepNumber,
        api_calls: 0,
        documents_scanned: runtime.scope.documentIds.length,
        documents_parsed: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        tool_result_chars: 0,
        estimated_cost: 0,
        last_focus_anchor: lastFocusAnchor,
        context_budget: contextBudgetStats,
        context_scope: contextStateSnapshot.context_scope,
        lazy_indexing: session.lazyIndexingStats,
        candidate_pages: runtime.candidatePages,
        read_pages: runtime.readPages,
        batch_summaries: [...session.batchSummaries],
        cumulative_answer: session.cumulativeAnswer,
        candidate_document_selection: session.candidateDocumentSelection,
        document_summaries: [...session.documentSummaries],
        parallel_document_limit: session.parallelDocumentLimit,
        page_boundary_loads: runtime.pageBoundaryLoads,
      },
      trace: {
        step_path: runtime.trace.stepPath,
        referenced_documents: runtime.trace.sortedDocuments(),
        cited_sources: citedSources,
        context_scope: contextStateSnapshot.context_scope,
        coverage_by_document: contextStateSnapshot.coverage_by_document,
        compaction_actions: contextStateSnapshot.compaction_actions,
        active_ranges: contextStateSnapshot.context_scope.active_ranges,
        candidate_pages: runtime.candidatePages,
        read_pages: runtime.readPages,
        active_page_ranges: contextStateSnapshot.context_scope.active_ranges,
        stale_page_ranges: runtime.stalePageRanges,
        page_query_history: runtime.pageQueryHistory,
        page_boundary_loads: runtime.pageBoundaryLoads,
        promoted_evidence_units: contextStateSnapshot.promoted_evidence_units,
        batch_summaries: [...session.batchSummaries],
        cumulative_answer: session.cumulativeAnswer,
        candidate_document_selection: session.candidateDocumentSelection,
        document_summaries: [...session.documentSummaries],
        parallel_document_limit: session.parallelDocumentLimit,
      },
    });
  }

  private failSession(session: ExploreSession, error: unknown): void {
    const message = error instanceof Error ? error.message : String(error);
    const runtime = session.runtime as ExplorationRuntimeState | null;
    session.status = "error";
    session.pendingQuestion = null;
    session.error = message;
    if (runtime) {
      session.contextStateSnapshot = runtime.agent.getContextState().snapshot() as unknown as Record<
        string,
        unknown
      >;
      session.contextBudgetStats = runtime.agent.getContextBudgetStats();
    }
    session.updatedAt = new Date();
    session.publish("error", { message });
  }
}

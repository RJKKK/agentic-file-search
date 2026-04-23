<script setup>
import { ChatDotRound, Collection, Document, Refresh } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import MarkdownIt from "markdown-it";
import { computed, inject, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import {
  buildDbParams,
  documentStatusType,
  formatValue,
  requestJson,
  shortText,
  withQuery,
} from "../api.js";

const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true });
const appState = inject("appState");
const activeTraceNames = ref([]);
const activeSessionId = ref("");
let eventSource = null;

const askOptions = reactive({
  documents: [],
  collections: [],
  documentQuery: "",
  loadingDocuments: false,
  loadingCollections: false,
});

const askForm = reactive({
  task: "",
  documentIds: [],
  collectionIds: [],
  enableSemantic: false,
  enableMetadata: false,
});

const sessionState = reactive({
  status: "idle",
  events: [],
  finalAnswer: "",
  displayedAnswer: "",
  error: "",
  trace: null,
  stats: null,
  candidateDocumentSelection: null,
  documentSummaries: [],
  parallelDocumentLimit: 3,
  batchSummaries: [],
  cumulativeAnswer: "",
  activeBatch: null,
  activeBatchAnswer: "",
  finalSynthesisDraft: "",
  finalSynthesisActive: false,
  showHumanModal: false,
  humanQuestion: "",
  humanResponse: "",
  replying: false,
});

const isAsking = computed(() =>
  ["creating", "running", "indexing", "awaiting_human", "answering"].includes(sessionState.status),
);
const selectedDocuments = computed(() => {
  const selected = new Set(askForm.documentIds);
  return askOptions.documents.filter((item) => selected.has(item.id));
});
const selectedCollections = computed(() => {
  const selected = new Set(askForm.collectionIds);
  return askOptions.collections.filter((item) => selected.has(item.id));
});
const renderedAnswer = computed(() => markdown.render(sessionState.displayedAnswer || ""));
const coordinatorVisible = computed(
  () =>
    sessionState.candidateDocumentSelection ||
    sessionState.documentSummaries.length ||
    sessionState.activeBatch ||
    sessionState.batchSummaries.length ||
    sessionState.cumulativeAnswer,
);
const traceStatusType = computed(() => {
  if (sessionState.status === "completed") return "success";
  if (sessionState.status === "error") return "danger";
  if (isAsking.value) return "warning";
  return "info";
});

onMounted(async () => {
  await Promise.all([refreshAskDocuments(), refreshCollections()]);
});

watch(
  () => appState.refreshTick,
  () => Promise.all([refreshAskDocuments(), refreshCollections()]),
);

onBeforeUnmount(() => {
  closeEventStream();
});

async function refreshAskDocuments(query = askOptions.documentQuery) {
  askOptions.loadingDocuments = true;
  askOptions.documentQuery = query || "";
  try {
    const params = buildDbParams(appState.dbPath, { page: 1, page_size: 100 });
    if (askOptions.documentQuery.trim()) params.set("q", askOptions.documentQuery.trim());
    const payload = await requestJson(`/api/documents?${params.toString()}`);
    askOptions.documents = payload.items || [];
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    askOptions.loadingDocuments = false;
  }
}

async function refreshCollections() {
  askOptions.loadingCollections = true;
  try {
    const payload = await requestJson("/api/collections");
    askOptions.collections = payload.items || [];
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    askOptions.loadingCollections = false;
  }
}

function closeEventStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

async function startAskSession() {
  if (!askForm.documentIds.length && !askForm.collectionIds.length) {
    ElMessage.warning("请先选择至少一个文档或 Collection");
    return;
  }
  if (!askForm.task.trim()) {
    ElMessage.warning("请输入问题");
    return;
  }

  closeEventStream();
  sessionState.status = "creating";
  sessionState.events = [];
  sessionState.finalAnswer = "";
  sessionState.displayedAnswer = "";
  sessionState.error = "";
  sessionState.trace = null;
  sessionState.stats = null;
  sessionState.candidateDocumentSelection = null;
  sessionState.documentSummaries = [];
  sessionState.parallelDocumentLimit = 3;
  sessionState.batchSummaries = [];
  sessionState.cumulativeAnswer = "";
  sessionState.activeBatch = null;
  sessionState.activeBatchAnswer = "";
  sessionState.finalSynthesisDraft = "";
  sessionState.finalSynthesisActive = false;
  activeTraceNames.value = [];

  try {
    const payload = await requestJson("/api/explore/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: askForm.task.trim(),
        document_ids: askForm.documentIds,
        collection_ids: askForm.collectionIds,
        collection_id: askForm.collectionIds[0] || null,
        db_path: appState.dbPath.trim() || null,
        enable_semantic: askForm.enableSemantic,
        enable_metadata: askForm.enableMetadata,
        batch_mode: "auto",
        batch_size: 5,
        batch_threshold: 10,
      }),
    });
    activeSessionId.value = payload.session_id;
    sessionState.status = "running";
    openEventStream(payload.session_id);
  } catch (error) {
    sessionState.status = "error";
    sessionState.error = error.message;
    ElMessage.error(error.message);
  }
}

function openEventStream(sessionId) {
  eventSource = new EventSource(`/api/explore/sessions/${encodeURIComponent(sessionId)}/events`);
  const eventTypes = [
    "start",
    "tool_call",
    "go_deeper",
    "ask_human",
    "context_scope_updated",
    "evidence_added",
    "context_compacted",
    "coverage_gap_detected",
    "page_scope_resolved",
    "candidate_pages_found",
    "candidate_documents_found",
    "pages_read",
    "page_boundary_loaded",
    "page_context_compacted",
    "stale_page_range_detected",
    "cache_hit",
    "image_enhance_started",
    "image_enhance_done",
    "lazy_indexing_started",
    "lazy_indexing_done",
    "answer_started",
    "answer_delta",
    "answer_done",
    "batch_started",
    "batch_answer_delta",
    "batch_answer_done",
    "cumulative_answer_updated",
    "batch_completed",
    "batch_context_released",
    "document_agent_started",
    "document_agent_tool_call",
    "document_answer_delta",
    "document_answer_done",
    "document_agent_completed",
    "final_synthesis_started",
    "final_synthesis_delta",
    "final_synthesis_done",
    "complete",
    "error",
  ];

  for (const type of eventTypes) {
    eventSource.addEventListener(type, (event) => {
      const payload = JSON.parse(event.data);
      sessionState.events = [...sessionState.events, payload];
      nextTick(scrollTraceToBottom);

      if (type === "ask_human") {
        sessionState.status = "awaiting_human";
        sessionState.showHumanModal = true;
        sessionState.humanQuestion = payload.data.question || "";
      } else if (type === "answer_started") {
        sessionState.status = "answering";
        sessionState.finalAnswer = "";
        sessionState.displayedAnswer = "";
        sessionState.showHumanModal = false;
      } else if (type === "answer_delta") {
        const deltaText = payload.data.delta_text || "";
        sessionState.status = "answering";
        sessionState.finalAnswer += deltaText;
        sessionState.displayedAnswer = sessionState.finalAnswer;
        nextTick(scrollAnswerToBottom);
      } else if (type === "answer_done") {
        sessionState.finalAnswer = payload.data.final_result || sessionState.finalAnswer;
        sessionState.displayedAnswer = sessionState.finalAnswer;
        nextTick(scrollAnswerToBottom);
      } else if (type === "batch_started") {
        sessionState.status = "running";
        sessionState.activeBatch = payload.data;
        sessionState.activeBatchAnswer = "";
      } else if (type === "candidate_documents_found") {
        sessionState.status = "running";
        sessionState.candidateDocumentSelection = payload.data;
        sessionState.parallelDocumentLimit = payload.data.parallel_document_limit || sessionState.parallelDocumentLimit;
      } else if (type === "document_agent_started") {
        sessionState.status = "running";
        sessionState.activeBatch = {
          batch_index: payload.data.document_index,
          batch_count: payload.data.document_count,
          document_ids: [payload.data.document_id],
          document_names: [payload.data.document_name],
        };
        sessionState.activeBatchAnswer = "";
        upsertDocumentSummary({
          document_id: payload.data.document_id,
          document_name: payload.data.document_name,
          candidate_pages: payload.data.candidate_pages || [],
          temporary_answer: "",
          status: "running",
          cited_sources: [],
        });
      } else if (type === "document_answer_delta") {
        sessionState.status = "running";
        sessionState.activeBatchAnswer = payload.data.accumulated_text || "";
        upsertDocumentSummary({
          document_id: payload.data.document_id,
          temporary_answer: payload.data.accumulated_text || "",
          status: "running",
        });
      } else if (type === "document_answer_done") {
        sessionState.activeBatchAnswer = payload.data.temporary_answer || sessionState.activeBatchAnswer;
        upsertDocumentSummary({
          document_id: payload.data.document_id,
          temporary_answer: payload.data.temporary_answer || "",
          status: payload.data.status || "answered",
          cited_sources: payload.data.cited_sources || [],
        });
      } else if (type === "document_agent_completed") {
        upsertDocumentSummary(payload.data);
      } else if (type === "batch_answer_delta") {
        sessionState.status = "running";
        sessionState.activeBatchAnswer = payload.data.accumulated_text || "";
      } else if (type === "batch_answer_done") {
        sessionState.activeBatchAnswer = payload.data.batch_answer || sessionState.activeBatchAnswer;
      } else if (type === "cumulative_answer_updated") {
        sessionState.cumulativeAnswer = payload.data.cumulative_answer || "";
      } else if (type === "final_synthesis_started") {
        sessionState.status = "answering";
        sessionState.finalSynthesisActive = true;
        sessionState.finalSynthesisDraft = "";
        sessionState.finalAnswer = "";
        sessionState.displayedAnswer = "";
      } else if (type === "final_synthesis_delta") {
        sessionState.status = "answering";
        sessionState.finalSynthesisDraft = payload.data.accumulated_text || payload.data.delta_text || "";
        sessionState.finalAnswer = sessionState.finalSynthesisDraft;
        sessionState.displayedAnswer = sessionState.finalAnswer;
        nextTick(scrollAnswerToBottom);
      } else if (type === "final_synthesis_done") {
        sessionState.finalSynthesisActive = false;
        sessionState.finalAnswer = payload.data.final_result || sessionState.finalAnswer;
        sessionState.displayedAnswer = sessionState.finalAnswer;
        nextTick(scrollAnswerToBottom);
      } else if (type === "batch_completed") {
        const batchIndex = payload.data.batch_index;
        sessionState.batchSummaries = [
          ...sessionState.batchSummaries.filter((item) => item.batch_index !== batchIndex),
          payload.data,
        ].sort((left, right) => (left.batch_index || 0) - (right.batch_index || 0));
      } else if (type === "complete") {
        sessionState.status = "completed";
        sessionState.finalAnswer = payload.data.final_result || sessionState.finalAnswer;
        sessionState.displayedAnswer = sessionState.finalAnswer;
        sessionState.trace = payload.data.trace || null;
        sessionState.stats = payload.data.stats || null;
        sessionState.batchSummaries =
          payload.data.stats?.batch_summaries || payload.data.trace?.batch_summaries || sessionState.batchSummaries;
        sessionState.documentSummaries =
          payload.data.stats?.document_summaries || payload.data.trace?.document_summaries || sessionState.documentSummaries;
        sessionState.candidateDocumentSelection =
          payload.data.stats?.candidate_document_selection ||
          payload.data.trace?.candidate_document_selection ||
          sessionState.candidateDocumentSelection;
        sessionState.parallelDocumentLimit =
          payload.data.stats?.parallel_document_limit ||
          payload.data.trace?.parallel_document_limit ||
          sessionState.parallelDocumentLimit;
        sessionState.cumulativeAnswer =
          payload.data.stats?.cumulative_answer || payload.data.trace?.cumulative_answer || sessionState.cumulativeAnswer;
        sessionState.finalSynthesisActive = false;
        sessionState.showHumanModal = false;
        closeEventStream();
        nextTick(scrollAnswerToBottom);
      } else if (type === "error") {
        sessionState.status = "error";
        sessionState.error = payload.data.message || "Session failed.";
        closeEventStream();
        ElMessage.error(sessionState.error);
      } else {
        sessionState.status = type.includes("lazy") ? "indexing" : "running";
      }
    });
  }

  eventSource.onerror = () => {
    void hydrateSessionSnapshot(sessionId);
    if (!["completed", "error"].includes(sessionState.status)) {
      sessionState.error = sessionState.error || "事件流连接已断开";
      ElMessage.warning(sessionState.error);
    }
  };

  window.setTimeout(() => {
    void hydrateSessionSnapshot(sessionId);
  }, 300);
}

async function hydrateSessionSnapshot(sessionId) {
  if (!sessionId || activeSessionId.value !== sessionId) return;
  try {
    const snapshot = await requestJson(`/api/explore/sessions/${encodeURIComponent(sessionId)}`);
    if (activeSessionId.value !== sessionId) return;
    sessionState.batchSummaries = snapshot.batch_summaries || sessionState.batchSummaries;
    sessionState.documentSummaries = snapshot.document_summaries || sessionState.documentSummaries;
    sessionState.candidateDocumentSelection =
      snapshot.candidate_document_selection || sessionState.candidateDocumentSelection;
    sessionState.parallelDocumentLimit = snapshot.parallel_document_limit || sessionState.parallelDocumentLimit;
    sessionState.cumulativeAnswer = snapshot.cumulative_answer || sessionState.cumulativeAnswer;
    if (snapshot.status === "completed") {
      sessionState.status = "completed";
      sessionState.finalAnswer = snapshot.final_result || "";
      sessionState.displayedAnswer = sessionState.finalAnswer;
      sessionState.showHumanModal = false;
      closeEventStream();
      nextTick(scrollAnswerToBottom);
    } else if (snapshot.status === "error") {
      sessionState.status = "error";
      sessionState.error = snapshot.error || "Session failed.";
      sessionState.showHumanModal = false;
      closeEventStream();
    } else if (snapshot.status === "awaiting_human") {
      sessionState.status = "awaiting_human";
      sessionState.showHumanModal = true;
      sessionState.humanQuestion = snapshot.pending_question || sessionState.humanQuestion;
    }
  } catch {
    // Best-effort sync for sessions that finish before SSE renders.
  }
}

function upsertDocumentSummary(nextSummary) {
  const documentId = nextSummary?.document_id;
  if (!documentId) return;
  const previous = sessionState.documentSummaries.find((item) => item.document_id === documentId) || {};
  const merged = { ...previous, ...nextSummary };
  sessionState.documentSummaries = [
    ...sessionState.documentSummaries.filter((item) => item.document_id !== documentId),
    merged,
  ].sort((left, right) => String(left.document_name || left.document_id).localeCompare(String(right.document_name || right.document_id)));
}

function statusTagType(status) {
  if (status === "answered") return "success";
  if (status === "failed") return "danger";
  if (status === "insufficient_evidence") return "warning";
  if (status === "running") return "primary";
  return "info";
}

function selectedCoordinatorDocs(selection) {
  if (!selection) return [];
  if (Array.isArray(selection.selected_documents)) return selection.selected_documents;
  if (Array.isArray(selection.selected_document_ids)) {
    return selection.selected_document_ids.map((documentId) => ({
      document_id: documentId,
      document_name: documentId,
    }));
  }
  return [];
}

async function submitHumanReply() {
  if (!activeSessionId.value || !sessionState.humanResponse.trim()) return;
  sessionState.replying = true;
  try {
    await requestJson(`/api/explore/sessions/${encodeURIComponent(activeSessionId.value)}/reply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ response: sessionState.humanResponse.trim() }),
    });
    sessionState.showHumanModal = false;
    sessionState.humanResponse = "";
    sessionState.status = "running";
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    sessionState.replying = false;
  }
}

function traceSummary(event) {
  const data = event.data || {};
  const pieces = [];
  if (data.tool_name) pieces.push(data.tool_name);
  if (data.batch_index) pieces.push(`Batch ${data.batch_index}${data.batch_count ? `/${data.batch_count}` : ""}`);
  if (data.document_index) pieces.push(`Doc Agent ${data.document_index}${data.document_count ? `/${data.document_count}` : ""}`);
  if (data.document_id) pieces.push(`文档 ${shortText(data.document_id, 16)}`);
  if (data.step) pieces.push(`步骤 ${data.step}`);
  if (data.reason) pieces.push(shortText(data.reason, 42));
  if (data.question) pieces.push(shortText(data.question, 42));
  if (data.message) pieces.push(shortText(data.message, 42));
  if (data.task) pieces.push(shortText(data.task, 42));
  if (data.context_scope?.active_document_id) {
    pieces.push(`当前 ${shortText(data.context_scope.active_document_id, 16)}`);
  }
  if (Array.isArray(data.candidate_pages)) pieces.push(`候选页 ${data.candidate_pages.length}`);
  if (Array.isArray(data.candidate_documents)) pieces.push(`候选文档 ${data.candidate_documents.length}`);
  if (Array.isArray(data.selected_documents)) pieces.push(`已选文档 ${data.selected_documents.length}`);
  if (Array.isArray(data.pages)) pieces.push(`读取页 ${data.pages.length}`);
  return pieces.length ? pieces.join(" / ") : shortText(readableFallback(data), 68);
}

function traceDetails(event) {
  const data = event.data || {};
  const details = [];
  const add = (label, value) => {
    if (value === undefined || value === null || value === "") return;
    if (Array.isArray(value) && !value.length) return;
    details.push({ label, value: formatValue(value) });
  };

  switch (event.type) {
    case "tool_call":
      add("工具", data.tool_name);
      add("输入", data.tool_input);
      add("原因", data.reason);
      add("上下文计划", data.context_plan);
      break;
    case "pages_read":
      add("文档", data.document_id || data.active_document_id || data.file_path);
      add("页码", data.pages || data.page_numbers || data.active_ranges);
      add("摘要", data.summary || data.reason || data.content_preview);
      break;
    case "candidate_pages_found":
      add("文档", data.document_id);
      add("候选页", data.candidate_pages);
      add("查询", data.query || data.search_terms);
      break;
    case "candidate_documents_found":
      add("策略", data.strategy);
      add("候选文档", data.candidate_documents);
      add("已选文档", data.selected_documents);
      add("并发上限", data.parallel_document_limit);
      break;
    case "document_agent_started":
      add("Document Agent", `${data.document_index}/${data.document_count}`);
      add("文档", data.document_name || data.document_id);
      add("候选页", data.candidate_pages);
      break;
    case "document_agent_tool_call":
      add("Document Agent", data.document_index);
      add("文档", data.document_id);
      add("工具", data.tool_name);
      add("输入", data.tool_input);
      add("原因", data.reason);
      break;
    case "document_answer_done":
      add("文档", data.document_id);
      add("状态", data.status);
      add("临时答案", data.temporary_answer);
      add("引用", data.cited_sources);
      break;
    case "document_agent_completed":
      add("文档", data.document_name || data.document_id);
      add("状态", data.status);
      add("临时答案", data.temporary_answer);
      add("引用", data.cited_sources);
      break;
    case "final_synthesis_started":
      add("文档数", data.document_count);
      add("文档", data.document_ids);
      break;
    case "final_synthesis_done":
      add("最终答案", shortText(data.final_result, 600));
      add("引用", data.cited_sources);
      break;
    case "context_scope_updated":
      add("当前文档", data.context_scope?.active_document_id || data.context_scope?.active_file_path);
      add("活动范围", data.context_scope?.active_ranges);
      add("范围", data.context_scope);
      break;
    case "complete":
      add("最终答案", shortText(data.final_result, 600));
      add("统计", data.stats);
      break;
    case "batch_started":
      add("Batch", `${data.batch_index}/${data.batch_count}`);
      add("Documents", data.document_names || data.document_ids);
      add("Previous cumulative answer", data.previous_cumulative_answer);
      break;
    case "batch_answer_done":
      add("Batch", data.batch_index);
      add("Batch answer", data.batch_answer);
      add("Citations", data.cited_sources);
      break;
    case "cumulative_answer_updated":
      add("Batch", data.batch_index);
      add("Cumulative answer", data.cumulative_answer);
      add("Citations", data.cited_sources);
      break;
    case "batch_completed":
      add("Batch", `${data.batch_index}/${data.batch_count}`);
      add("Documents", data.document_names || data.document_ids);
      add("Batch answer", data.batch_answer);
      break;
    case "batch_context_released":
      add("Batch", data.batch_index);
      add("Releases", data.releases);
      break;
    case "error":
      add("错误", data.message);
      break;
    default:
      add("任务", data.task);
      add("消息", data.message);
      add("原因", data.reason);
      add("问题", data.question);
      add("内容", readableFallback(data));
      break;
  }
  return details;
}

function readableFallback(data) {
  if (!data || typeof data !== "object") return String(data || "");
  const keys = ["message", "reason", "question", "task", "tool_name", "document_id", "summary", "final_result"];
  return keys.filter((key) => data[key]).map((key) => `${key}: ${formatValue(data[key])}`).join("\n");
}

function scrollTraceToBottom() {
  const el = document.querySelector(".trace-list");
  if (el) el.scrollTop = el.scrollHeight;
}

function scrollAnswerToBottom() {
  const el = document.querySelector(".answer-scroll");
  if (el) el.scrollTop = el.scrollHeight;
}
</script>

<template>
  <section class="page qa-page">
    <div class="qa-picker">
      <el-card shadow="never" class="toolbar-card">
        <el-form label-position="top" class="scope-form">
          <el-form-item label="选择 Collection">
            <el-select
              v-model="askForm.collectionIds"
              multiple
              filterable
              clearable
              :loading="askOptions.loadingCollections"
              placeholder="选择一个或多个 Collection"
              class="document-select"
            >
              <el-option
                v-for="item in askOptions.collections"
                :key="item.id"
                :label="item.name"
                :value="item.id"
              >
                <div class="select-option">
                  <span>{{ item.name }}</span>
                  <el-tag size="small" type="info">{{ item.document_count || 0 }} 文档</el-tag>
                </div>
              </el-option>
            </el-select>
          </el-form-item>

          <el-form-item label="选择文档">
            <el-select
              v-model="askForm.documentIds"
              multiple
              filterable
              remote
              reserve-keyword
              clearable
              :remote-method="refreshAskDocuments"
              :loading="askOptions.loadingDocuments"
              placeholder="搜索并选择额外文档"
              class="document-select"
            >
              <el-option
                v-for="item in askOptions.documents"
                :key="item.id"
                :label="item.original_filename"
                :value="item.id"
              >
                <div class="select-option">
                  <span>{{ item.original_filename }}</span>
                  <el-tag size="small" :type="documentStatusType(item.status)">{{ item.status }}</el-tag>
                  <span class="option-meta">{{ item.page_count }} 页</span>
                </div>
              </el-option>
            </el-select>
          </el-form-item>
        </el-form>

        <div class="selected-docs">
          <el-tag
            v-for="collection in selectedCollections"
            :key="collection.id"
            :icon="Collection"
            closable
            @close="askForm.collectionIds = askForm.collectionIds.filter((id) => id !== collection.id)"
          >
            {{ collection.name }}
          </el-tag>
          <el-tag
            v-for="doc in selectedDocuments"
            :key="doc.id"
            :icon="Document"
            closable
            @close="askForm.documentIds = askForm.documentIds.filter((id) => id !== doc.id)"
          >
            {{ doc.original_filename }}
          </el-tag>
          <el-text v-if="!selectedCollections.length && !selectedDocuments.length" type="info">
            请选择文档或 Collection 后开始问答
          </el-text>
        </div>
      </el-card>
    </div>

    <div class="qa-workspace">
      <div class="trace-answer-panel">
        <section class="trace-section">
          <div class="section-title">
            <div>
              <h2>Trace</h2>
              <span>运行过程实时追加，展开可查看完整信息</span>
            </div>
            <el-tag :type="traceStatusType">{{ sessionState.status }}</el-tag>
          </div>
          <div class="trace-list">
            <el-empty v-if="!sessionState.events.length" description="暂无 trace" :image-size="72" />
            <el-collapse v-else v-model="activeTraceNames">
              <el-collapse-item
                v-for="event in sessionState.events"
                :key="`${event.sequence}-${event.type}`"
                :name="`${event.sequence}-${event.type}`"
              >
                <template #title>
                  <div class="trace-title">
                    <el-tag size="small" effect="plain">{{ event.type }}</el-tag>
                    <span>#{{ event.sequence }}</span>
                    <span class="trace-summary">{{ traceSummary(event) }}</span>
                  </div>
                </template>
                <div class="trace-detail">
                  <div v-for="item in traceDetails(event)" :key="item.label" class="trace-row">
                    <strong>{{ item.label }}</strong>
                    <pre>{{ item.value }}</pre>
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>
        </section>

        <section v-if="coordinatorVisible" class="batch-section agent-section">
          <div class="section-title">
            <div>
              <h2>文档子 Agent</h2>
              <span>主 Agent 先筛候选文档，各文档并行生成临时答案，最后统一综合</span>
            </div>
            <div class="section-tags">
              <el-tag v-if="sessionState.candidateDocumentSelection" type="info">
                候选 {{ sessionState.candidateDocumentSelection.candidate_documents?.length || sessionState.candidateDocumentSelection.candidate_count || 0 }}
              </el-tag>
              <el-tag v-if="sessionState.candidateDocumentSelection" type="success">
                已选 {{ sessionState.candidateDocumentSelection.selected_documents?.length || sessionState.candidateDocumentSelection.selected_count || 0 }}
              </el-tag>
              <el-tag v-if="sessionState.parallelDocumentLimit" type="warning">
                并发 {{ sessionState.parallelDocumentLimit }}
              </el-tag>
            </div>
          </div>
          <div class="batch-cache">
            <div v-if="sessionState.candidateDocumentSelection" class="batch-card coordinator-card">
              <strong>主 Agent 候选选择</strong>
              <p>
                {{ sessionState.candidateDocumentSelection.strategy || "retrieval_plus_llm" }}
                <span v-if="sessionState.candidateDocumentSelection.query">
                  · {{ shortText(sessionState.candidateDocumentSelection.query, 80) }}
                </span>
              </p>
              <div class="candidate-doc-list">
                <el-tag
                  v-for="doc in selectedCoordinatorDocs(sessionState.candidateDocumentSelection)"
                  :key="doc.document_id"
                  size="small"
                  type="success"
                  effect="plain"
                >
                  {{ doc.document_name || doc.document_id }}
                  <span v-if="doc.candidate_pages"> · {{ doc.candidate_pages.length }} 页</span>
                </el-tag>
                <el-text
                  v-if="!selectedCoordinatorDocs(sessionState.candidateDocumentSelection).length"
                  type="warning"
                  size="small"
                >
                  主 Agent 未选择可处理文档
                </el-text>
              </div>
            </div>
            <div v-if="sessionState.activeBatch" class="batch-card active-batch">
              <strong>运行中的文档 Agent</strong>
              <p>{{ (sessionState.activeBatch.document_names || []).join("；") }}</p>
              <pre v-if="sessionState.activeBatchAnswer">{{ sessionState.activeBatchAnswer }}</pre>
            </div>
            <el-collapse v-if="sessionState.documentSummaries.length">
              <el-collapse-item
                v-for="summary in sessionState.documentSummaries"
                :key="summary.document_id"
                :title="summary.document_name || summary.document_id"
              >
                <div class="document-summary-meta">
                  <el-tag size="small" :type="statusTagType(summary.status)">
                    {{ summary.status || "pending" }}
                  </el-tag>
                  <el-tag
                    v-for="page in summary.candidate_pages || []"
                    :key="`${summary.document_id}-${page.page_no || page.file_path}`"
                    size="small"
                    type="info"
                    effect="plain"
                  >
                    p{{ page.page_no || "?" }}
                  </el-tag>
                </div>
                <pre>{{ summary.temporary_answer || "等待该文档临时答案..." }}</pre>
                <el-tag
                  v-for="source in summary.cited_sources || []"
                  :key="source"
                  size="small"
                  type="info"
                >
                  {{ source }}
                </el-tag>
              </el-collapse-item>
            </el-collapse>
            <el-collapse v-if="sessionState.batchSummaries.length && !sessionState.documentSummaries.length">
              <el-collapse-item
                v-for="batch in sessionState.batchSummaries"
                :key="batch.batch_index"
                :title="`Batch ${batch.batch_index}/${batch.batch_count}`"
              >
                <p class="batch-docs">{{ (batch.document_names || batch.document_ids || []).join("；") }}</p>
                <pre>{{ batch.batch_answer }}</pre>
                <el-tag
                  v-for="source in batch.cited_sources || []"
                  :key="source"
                  size="small"
                  type="info"
                >
                  {{ source }}
                </el-tag>
              </el-collapse-item>
            </el-collapse>
            <div v-if="sessionState.cumulativeAnswer" class="batch-card">
              <strong>临时答案汇总</strong>
              <pre>{{ sessionState.cumulativeAnswer }}</pre>
            </div>
            <div v-if="sessionState.finalSynthesisActive || sessionState.finalSynthesisDraft" class="batch-card synthesis-card">
              <strong>最终综合</strong>
              <p>{{ sessionState.finalSynthesisActive ? "主 Agent 正在综合所有文档临时答案" : "综合完成" }}</p>
              <pre v-if="sessionState.finalSynthesisDraft">{{ sessionState.finalSynthesisDraft }}</pre>
            </div>
          </div>
        </section>

        <section class="answer-section">
          <div class="section-title">
            <div>
              <h2>答案</h2>
              <span>{{ sessionState.error || "完成后以 Markdown 渲染" }}</span>
            </div>
          </div>
          <div class="answer-scroll">
            <el-empty v-if="!sessionState.displayedAnswer" description="答案会显示在这里" :image-size="82" />
            <article v-else class="markdown-body answer-markdown" v-html="renderedAnswer" />
          </div>
        </section>
      </div>
    </div>

    <div class="ask-bar">
      <el-input
        v-model="askForm.task"
        type="textarea"
        :autosize="{ minRows: 2, maxRows: 5 }"
        resize="none"
        placeholder="请输入问题，例如：请总结这些文档里的核心条款"
        @keydown.ctrl.enter.prevent="startAskSession"
      />
      <div class="ask-actions">
        <el-checkbox v-model="askForm.enableSemantic">语义检索</el-checkbox>
        <el-checkbox v-model="askForm.enableMetadata">Metadata 过滤</el-checkbox>
        <el-button :icon="Refresh" @click="Promise.all([refreshAskDocuments(), refreshCollections()])">
          刷新
        </el-button>
        <el-button type="primary" :loading="isAsking" :icon="ChatDotRound" @click="startAskSession">
          提问
        </el-button>
      </div>
    </div>
  </section>

  <el-dialog v-model="sessionState.showHumanModal" title="需要人工回复" width="560px">
    <p class="human-question">{{ sessionState.humanQuestion }}</p>
    <el-input v-model="sessionState.humanResponse" type="textarea" :rows="5" placeholder="请输入回复" />
    <template #footer>
      <el-button @click="sessionState.showHumanModal = false">稍后</el-button>
      <el-button type="primary" :loading="sessionState.replying" @click="submitHumanReply">发送回复</el-button>
    </template>
  </el-dialog>
</template>

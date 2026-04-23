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

function unwrapFinalAnswer(value) {
  const text = String(value || "");
  const trimmed = text.trim();
  if (!trimmed.startsWith("{")) return text;
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object") {
      if (typeof parsed.final_result === "string" && parsed.final_result.trim()) {
        return parsed.final_result;
      }
      if (
        parsed.action &&
        typeof parsed.action === "object" &&
        typeof parsed.action.final_result === "string" &&
        parsed.action.final_result.trim()
      ) {
        return parsed.action.final_result;
      }
    }
  } catch {
    // Keep the original text when it is not valid JSON.
  }
  return text;
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
    "pages_read",
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
        sessionState.finalAnswer = unwrapFinalAnswer(payload.data.final_result || sessionState.finalAnswer);
        sessionState.displayedAnswer = sessionState.finalAnswer;
        nextTick(scrollAnswerToBottom);
      } else if (type === "complete") {
        sessionState.status = "completed";
        sessionState.finalAnswer = unwrapFinalAnswer(payload.data.final_result || sessionState.finalAnswer);
        sessionState.displayedAnswer = sessionState.finalAnswer;
        sessionState.trace = payload.data.trace || null;
        sessionState.stats = payload.data.stats || null;
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
    if (snapshot.status === "completed") {
      sessionState.status = "completed";
      sessionState.finalAnswer = unwrapFinalAnswer(snapshot.final_result || "");
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
    case "context_scope_updated":
      add("当前文档", data.context_scope?.active_document_id || data.context_scope?.active_file_path);
      add("活动范围", data.context_scope?.active_ranges);
      add("范围", data.context_scope);
      break;
    case "complete":
      add("最终答案", shortText(data.final_result, 600));
      add("统计", data.stats);
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

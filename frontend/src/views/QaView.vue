<script setup>
import { ChatDotRound, Collection, Document, Refresh } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import MarkdownIt from "markdown-it";
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";

import {
  documentStatusType,
  formatValue,
  requestJson,
  shortText,
  streamJsonEvents,
} from "../api.js";
import { useRetrievalScope } from "../composables/useRetrievalScope.js";

const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true });
const {
  appState,
  askOptions,
  askForm,
  selectedDocuments,
  selectedCollections,
  refreshAskDocuments,
  refreshCollections,
} = useRetrievalScope();

const queryMode = ref("agent");
const activeTraceNames = ref([]);
const activeSessionId = ref("");
let eventSource = null;
let traditionalAbortController = null;

const sessionState = reactive({
  status: "idle",
  events: [],
  finalAnswer: "",
  displayedAnswer: "",
  error: "",
  showHumanModal: false,
  humanQuestion: "",
  humanResponse: "",
  replying: false,
});

const ragState = reactive({
  loading: false,
  answer: "",
  usedChunks: [],
  detailChunks: [],
  warnings: [],
  error: "",
  detailContentByChunkId: {},
  detailLoadingByChunkId: {},
  detailOpenByChunkId: {},
  detailErrorByChunkId: {},
});

const isAsking = computed(() =>
  ["creating", "running", "indexing", "awaiting_human", "answering"].includes(sessionState.status),
);

const expandableDetailChunks = computed(() =>
  [...(ragState.detailChunks || []), ...(ragState.usedChunks || [])]
    .filter((chunk, index, list) =>
      chunk.show_full_chunk_detail &&
      list.findIndex((item) => item.reference_id === chunk.reference_id && item.reference_kind === chunk.reference_kind) === index,
    ),
);

const renderedAnswer = computed(() => {
  const source = queryMode.value === "agent" ? sessionState.displayedAnswer : ragState.answer;
  const html = markdown.render(source || "");
  return decorateTraditionalAnswerHtml(html);
});

onMounted(async () => {
  await Promise.all([refreshAskDocuments(), refreshCollections()]);
});

watch(
  () => appState.refreshTick,
  () => Promise.all([refreshAskDocuments(), refreshCollections()]),
);

watch(queryMode, (mode) => {
  if (mode === "agent") {
    closeTraditionalStream();
    return;
  }
  closeEventStream();
});

onBeforeUnmount(() => {
  closeEventStream();
  closeTraditionalStream();
});

function closeEventStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function closeTraditionalStream() {
  if (traditionalAbortController) {
    traditionalAbortController.abort();
    traditionalAbortController = null;
  }
}

function unwrapFinalAnswer(value) {
  const text = String(value || "");
  const trimmed = text.trim();
  if (!trimmed.startsWith("{")) {
    return text;
  }
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed?.final_result) {
      return parsed.final_result;
    }
    if (parsed?.action?.final_result) {
      return parsed.action.final_result;
    }
  } catch {
    // Keep raw text.
  }
  return text;
}

async function submitQuestion() {
  if (!askForm.documentIds.length && !askForm.collectionIds.length) {
    ElMessage.warning("Please select at least one document or collection.");
    return;
  }
  if (!askForm.task.trim()) {
    ElMessage.warning("Please enter a question.");
    return;
  }
  if (queryMode.value === "agent") {
    await startAgentSession();
    return;
  }
  await startTraditionalQuery();
}

async function startTraditionalQuery() {
  closeTraditionalStream();
  ragState.loading = true;
  ragState.answer = "";
  ragState.usedChunks = [];
  ragState.detailChunks = [];
  ragState.warnings = [];
  ragState.error = "";
  ragState.detailContentByChunkId = {};
  ragState.detailLoadingByChunkId = {};
  ragState.detailOpenByChunkId = {};
  ragState.detailErrorByChunkId = {};
  traditionalAbortController = new AbortController();
  try {
    await streamJsonEvents(
      "/api/rag/query/stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: traditionalAbortController.signal,
        body: JSON.stringify({
          question: askForm.task.trim(),
          mode: askForm.retrievalMode,
          document_ids: askForm.documentIds,
          collection_ids: askForm.collectionIds,
          keyword_weight: Number(askForm.keywordWeight || 0.5),
          semantic_weight: Number(askForm.semanticWeight || 0.5),
        }),
      },
      {
        start(payload) {
          ragState.warnings = payload.warnings || [];
        },
        answer_delta(payload) {
          ragState.answer += payload.delta_text || "";
          nextTick(scrollAnswerToBottom);
        },
        complete(payload) {
          ragState.answer = payload.answer || ragState.answer;
          ragState.usedChunks = payload.used_chunks || [];
          ragState.detailChunks = payload.detail_chunks || [];
          ragState.warnings = payload.warnings || [];
        },
        error(payload) {
          throw new Error(payload.message || "Traditional RAG stream failed.");
        },
      },
    );
  } catch (error) {
    if (error?.name === "AbortError") {
      return;
    }
    ragState.error = error.message;
    ElMessage.error(error.message);
  } finally {
    traditionalAbortController = null;
    ragState.loading = false;
    nextTick(scrollAnswerToBottom);
  }
}

async function startAgentSession() {
  closeEventStream();
  closeTraditionalStream();
  sessionState.status = "creating";
  sessionState.events = [];
  sessionState.finalAnswer = "";
  sessionState.displayedAnswer = "";
  sessionState.error = "";
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
        sessionState.status = "answering";
        sessionState.finalAnswer += payload.data.delta_text || "";
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
    if (!["completed", "error"].includes(sessionState.status)) {
      sessionState.error = sessionState.error || "Event stream disconnected.";
      ElMessage.warning(sessionState.error);
    }
  };
}

async function submitHumanReply() {
  if (!activeSessionId.value || !sessionState.humanResponse.trim()) {
    return;
  }
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
  if (data.document_id) pieces.push(`doc ${shortText(data.document_id, 16)}`);
  if (data.step) pieces.push(`step ${data.step}`);
  if (data.reason) pieces.push(shortText(data.reason, 42));
  if (data.question) pieces.push(shortText(data.question, 42));
  if (data.message) pieces.push(shortText(data.message, 42));
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
  add("type", event.type);
  add("tool", data.tool_name);
  add("reason", data.reason);
  add("message", data.message);
  add("question", data.question);
  add("payload", readableFallback(data));
  return details;
}

function readableFallback(data) {
  if (!data || typeof data !== "object") return String(data || "");
  return Object.entries(data)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `${key}: ${formatValue(value)}`)
    .join("\n");
}

function renderMarkdown(text) {
  return markdown.render(String(text || ""));
}

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildInlineDetailMarkup(chunk) {
  const chunkId = chunk.reference_id;
  const open = Boolean(ragState.detailOpenByChunkId[chunkId]);
  const loading = Boolean(ragState.detailLoadingByChunkId[chunkId]);
  const error = String(ragState.detailErrorByChunkId[chunkId] || "");
  const content = String(ragState.detailContentByChunkId[chunkId] || "");
  let detailHtml = "";
  if (open) {
    if (loading) {
      detailHtml = '<div class="inline-answer-detail-body"><div class="inline-answer-detail-state">加载中...</div></div>';
    } else if (error) {
      detailHtml = `<div class="inline-answer-detail-body"><div class="inline-answer-detail-state inline-answer-detail-error">${error}</div></div>`;
    } else if (content) {
      detailHtml =
        `<div class="inline-answer-detail-body markdown-body answer-markdown detail-markdown">` +
        `${renderMarkdown(content)}` +
        "</div>";
    }
  }
  return (
    `<span class="inline-answer-detail">` +
    `<button type="button" class="inline-answer-detail-toggle" data-answer-detail-button="true" data-chunk-id="${chunkId}">` +
    `${open ? "收起" : "展开"}引用 ${chunk.citation_no} 对应完整块` +
    `</button>` +
    detailHtml +
    `</span>`
  );
}

function decorateTraditionalAnswerHtml(html) {
  if (queryMode.value !== "traditional" || ragState.loading || !expandableDetailChunks.value.length) {
    return html;
  }
  let decorated = html;
  for (const chunk of expandableDetailChunks.value) {
    const anchorPattern = new RegExp(
      `<a\\s+href="${escapeRegExp(chunk.source_link)}"[^>]*>.*?<\\/a>`,
      "g",
    );
    decorated = decorated.replace(anchorPattern, buildInlineDetailMarkup(chunk));
  }
  return decorated;
}

async function toggleFullChunkDetail(chunk) {
  const chunkId = chunk.reference_id;
  const isOpen = Boolean(ragState.detailOpenByChunkId[chunkId]);
  if (isOpen) {
    ragState.detailOpenByChunkId = {
      ...ragState.detailOpenByChunkId,
      [chunkId]: false,
    };
    return;
  }

  ragState.detailOpenByChunkId = {
    ...ragState.detailOpenByChunkId,
    [chunkId]: true,
  };
  if (ragState.detailContentByChunkId[chunkId] || ragState.detailLoadingByChunkId[chunkId]) {
    return;
  }

  ragState.detailLoadingByChunkId = {
    ...ragState.detailLoadingByChunkId,
    [chunkId]: true,
  };
  ragState.detailErrorByChunkId = {
    ...ragState.detailErrorByChunkId,
    [chunkId]: "",
  };
  try {
    const payload = await requestJson(chunk.source_link);
    ragState.detailContentByChunkId = {
      ...ragState.detailContentByChunkId,
      [chunkId]: payload?.chunk?.content_md || "",
    };
  } catch (error) {
    ragState.detailErrorByChunkId = {
      ...ragState.detailErrorByChunkId,
      [chunkId]: error.message || "Failed to load chunk content.",
    };
  } finally {
    ragState.detailLoadingByChunkId = {
      ...ragState.detailLoadingByChunkId,
      [chunkId]: false,
    };
  }
}

function handleAnswerInteraction(event) {
  const button = event.target instanceof Element ? event.target.closest("[data-answer-detail-button]") : null;
  if (!button) {
    return;
  }
  event.preventDefault();
  const chunkId = button.getAttribute("data-chunk-id") || "";
  const chunk = [...(ragState.detailChunks || []), ...(ragState.usedChunks || [])]
    .find((item) => item.reference_id === chunkId);
  if (!chunk) {
    return;
  }
  void toggleFullChunkDetail(chunk);
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
          <el-form-item label="Collections">
            <el-select
              v-model="askForm.collectionIds"
              multiple
              filterable
              clearable
              :loading="askOptions.loadingCollections"
              placeholder="Select one or more collections"
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
                  <el-tag size="small" type="info">{{ item.document_count || 0 }} docs</el-tag>
                </div>
              </el-option>
            </el-select>
          </el-form-item>

          <el-form-item label="Documents">
            <el-select
              v-model="askForm.documentIds"
              multiple
              filterable
              remote
              reserve-keyword
              clearable
              :remote-method="refreshAskDocuments"
              :loading="askOptions.loadingDocuments"
              placeholder="Search and add documents"
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
                  <span class="option-meta">{{ item.page_count }} pages</span>
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
        </div>

        <div class="qa-mode-bar">
          <el-radio-group v-model="queryMode">
            <el-radio-button label="agent">Agent 检索</el-radio-button>
            <el-radio-button label="traditional">传统检索</el-radio-button>
          </el-radio-group>

          <div v-if="queryMode === 'traditional'" class="rag-controls">
            <el-select v-model="askForm.retrievalMode" class="rag-mode-select">
              <el-option label="Hybrid" value="hybrid" />
              <el-option label="Keyword" value="keyword" />
              <el-option label="Semantic" value="semantic" />
            </el-select>
            <el-input-number
              v-if="askForm.retrievalMode === 'hybrid'"
              v-model="askForm.keywordWeight"
              :min="0"
              :max="1"
              :step="0.1"
              controls-position="right"
            />
            <el-input-number
              v-if="askForm.retrievalMode === 'hybrid'"
              v-model="askForm.semanticWeight"
              :min="0"
              :max="1"
              :step="0.1"
              controls-position="right"
            />
          </div>

          <div v-else class="rag-controls">
            <el-checkbox v-model="askForm.enableSemantic">Semantic</el-checkbox>
            <el-checkbox v-model="askForm.enableMetadata">Metadata</el-checkbox>
          </div>
        </div>
      </el-card>
    </div>

    <div class="qa-workspace">
      <div class="trace-answer-panel">
        <section v-if="queryMode === 'agent'" class="trace-section">
          <div class="section-title">
            <div>
              <h2>Trace</h2>
              <span>Live agent execution events</span>
            </div>
            <el-tag>{{ sessionState.status }}</el-tag>
          </div>
          <div class="trace-list">
            <el-empty v-if="!sessionState.events.length" description="No trace yet" :image-size="72" />
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
              <h2>Answer</h2>
              <span>
                {{
                  queryMode === "agent"
                    ? sessionState.error || "Markdown answer stream"
                    : ragState.error || "Traditional RAG answer stream"
                }}
              </span>
            </div>
          </div>
          <div class="answer-scroll">
            <el-alert
              v-for="warning in ragState.warnings"
              v-if="queryMode === 'traditional'"
              :key="warning"
              type="warning"
              :closable="false"
              show-icon
              class="rag-warning"
              :title="warning"
            />
            <el-empty
              v-if="!(queryMode === 'agent' ? sessionState.displayedAnswer : ragState.answer)"
              description="Answer will appear here"
              :image-size="82"
            />
            <article
              v-else
              class="markdown-body answer-markdown"
              v-html="renderedAnswer"
              @click="handleAnswerInteraction"
            />

            <div v-if="queryMode === 'traditional' && ragState.usedChunks.length" class="rag-citations">
              <h3>详情内容如下</h3>
              <div v-for="chunk in ragState.usedChunks" :key="chunk.reference_id" class="citation-card">
                <div class="citation-meta">
                  <span>[{{ chunk.citation_no }}]</span>
                  <span>{{ chunk.document_name }}</span>
                  <span>{{ chunk.reference_kind }}</span>
                  <span>score {{ Number(chunk.score || 0).toFixed(4) }}</span>
                  <span>pages {{ (chunk.page_nos || []).join(", ") }}</span>
                  <span>locator {{ chunk.source_locator || "-" }}</span>
                  <span>compression {{ chunk.compression_applied ? "yes" : "no" }}</span>
                </div>
                <div class="citation-links">
                  <span>retrieval ids: {{ (chunk.retrieval_unit_ids || []).join(", ") || "-" }}</span>
                  <a :href="chunk.source_link" target="_blank" rel="noreferrer">Open Chunk</a>
                </div>
              </div>
            </div>
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
        placeholder="Ask a question about the selected documents"
        @keydown.ctrl.enter.prevent="submitQuestion"
      />
      <div class="ask-actions">
        <el-button :icon="Refresh" @click="Promise.all([refreshAskDocuments(), refreshCollections()])">
          Refresh
        </el-button>
        <el-button
          type="primary"
          :loading="queryMode === 'agent' ? isAsking : ragState.loading"
          :icon="ChatDotRound"
          @click="submitQuestion"
        >
          Ask
        </el-button>
      </div>
    </div>
  </section>

  <el-dialog v-model="sessionState.showHumanModal" title="Human Input Needed" width="560px">
    <p class="human-question">{{ sessionState.humanQuestion }}</p>
    <el-input v-model="sessionState.humanResponse" type="textarea" :rows="5" placeholder="Type your reply" />
    <template #footer>
      <el-button @click="sessionState.showHumanModal = false">Later</el-button>
      <el-button type="primary" :loading="sessionState.replying" @click="submitHumanReply">Send</el-button>
    </template>
  </el-dialog>
</template>

<script setup>
import { ChatDotRound, Delete, Document, Refresh, Search, Setting, Upload, View } from "@element-plus/icons-vue";
import { ElMessage, ElMessageBox } from "element-plus";
import MarkdownIt from "markdown-it";
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from "vue";

const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true });

const activeView = ref("qa");
const dbPath = ref("");
const activeSessionId = ref("");
const activeTraceNames = ref([]);
let eventSource = null;
let typewriterTimer = null;

const documentList = reactive({
  items: [],
  query: "",
  page: 1,
  pageSize: 20,
  total: 0,
  loading: false,
});

const askOptions = reactive({ items: [], query: "", loading: false });
const uploadState = reactive({ uploading: false, progress: 0 });

const detailState = reactive({
  visible: false,
  loading: false,
  parseLoading: false,
  document: null,
  pageSummary: null,
  metadataDraft: "{}",
  pages: [],
  pagesPage: 1,
  pagesPageSize: 8,
  pagesTotal: 0,
});

const askForm = reactive({
  task: "",
  documentIds: [],
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
  ["creating", "running", "indexing", "awaiting_human"].includes(sessionState.status),
);
const selectedDocuments = computed(() => {
  const selected = new Set(askForm.documentIds);
  return askOptions.items.filter((item) => selected.has(item.id));
});
const renderedAnswer = computed(() => markdown.render(sessionState.displayedAnswer || ""));
const traceStatusType = computed(() => {
  if (sessionState.status === "completed") return "success";
  if (sessionState.status === "error") return "danger";
  if (isAsking.value) return "warning";
  return "info";
});

const documentStatusType = (status) => {
  if (status === "deleted") return "danger";
  if (["pages_ready", "indexed", "completed"].includes(status)) return "success";
  if (status === "uploaded") return "warning";
  return "info";
};

onMounted(async () => {
  await Promise.all([refreshDocuments(), refreshAskDocuments()]);
});

onBeforeUnmount(() => {
  closeEventStream();
  stopTypewriter();
});

function buildDbParams() {
  const params = new URLSearchParams();
  if (dbPath.value.trim()) params.set("db_path", dbPath.value.trim());
  return params;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.message || payload.error || payload.detail || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload;
}

async function refreshDocuments(page = documentList.page) {
  documentList.loading = true;
  documentList.page = page;
  try {
    const params = buildDbParams();
    params.set("page", String(page));
    params.set("page_size", String(documentList.pageSize));
    if (documentList.query.trim()) params.set("q", documentList.query.trim());
    const payload = await requestJson(`/api/documents?${params.toString()}`);
    documentList.items = payload.items || [];
    documentList.total = payload.total || 0;
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    documentList.loading = false;
  }
}

async function refreshAskDocuments(query = askOptions.query) {
  askOptions.loading = true;
  askOptions.query = query || "";
  try {
    const params = buildDbParams();
    params.set("page", "1");
    params.set("page_size", "100");
    if (askOptions.query.trim()) params.set("q", askOptions.query.trim());
    const payload = await requestJson(`/api/documents?${params.toString()}`);
    askOptions.items = payload.items || [];
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    askOptions.loading = false;
  }
}

function handlePageSizeChange(size) {
  documentList.pageSize = size;
  refreshDocuments(1);
}

async function uploadDocument(uploadRequest) {
  uploadState.uploading = true;
  uploadState.progress = 0;
  try {
    const formData = new FormData();
    formData.append("file", uploadRequest.file);
    if (dbPath.value.trim()) formData.append("db_path", dbPath.value.trim());
    const payload = await uploadWithProgress("/api/documents", formData, (percent) => {
      uploadState.progress = percent;
      uploadRequest.onProgress({ percent });
    });
    uploadState.progress = 100;
    ElMessage.success(`${payload.document.original_filename} 上传完成`);
    await Promise.all([refreshDocuments(1), refreshAskDocuments()]);
    uploadRequest.onSuccess(payload);
  } catch (error) {
    ElMessage.error(error.message);
    uploadRequest.onError(error);
  } finally {
    uploadState.uploading = false;
  }
}

function handleUploadProgress(event) {
  uploadState.progress = Math.round(event.percent || 0);
}

function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      onProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
    };
    xhr.onload = () => {
      const payload = JSON.parse(xhr.responseText || "{}");
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload);
        return;
      }
      reject(new Error(payload.message || payload.error || payload.detail || `Request failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("上传失败，请检查网络或服务状态"));
    xhr.send(formData);
  });
}

async function openDocumentDetail(row) {
  detailState.visible = true;
  detailState.document = row;
  detailState.pagesPage = 1;
  await loadDocumentDetail(row.id);
  await loadDocumentPages(row.id, 1);
}

async function loadDocumentDetail(docId = detailState.document?.id) {
  if (!docId) return;
  detailState.loading = true;
  try {
    const params = buildDbParams();
    const suffix = params.toString();
    const payload = await requestJson(
      `/api/documents/${encodeURIComponent(docId)}${suffix ? `?${suffix}` : ""}`,
    );
    detailState.document = payload.document;
    detailState.pageSummary = payload.page_summary || payload.parse_summary || null;
    detailState.metadataDraft = JSON.stringify(payload.document.metadata || {}, null, 2);
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    detailState.loading = false;
  }
}

async function loadDocumentPages(docId = detailState.document?.id, page = detailState.pagesPage) {
  if (!docId) return;
  detailState.loading = true;
  detailState.pagesPage = page;
  try {
    const params = buildDbParams();
    params.set("page", String(page));
    params.set("page_size", String(detailState.pagesPageSize));
    const payload = await requestJson(
      `/api/documents/${encodeURIComponent(docId)}/pages?${params.toString()}`,
    );
    detailState.pages = payload.items || [];
    detailState.pagesTotal = payload.total || 0;
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    detailState.loading = false;
  }
}

async function saveMetadata() {
  if (!detailState.document?.id) return;
  try {
    const metadata = JSON.parse(detailState.metadataDraft || "{}");
    const params = buildDbParams();
    const suffix = params.toString();
    await requestJson(
      `/api/documents/${encodeURIComponent(detailState.document.id)}${suffix ? `?${suffix}` : ""}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ metadata }),
      },
    );
    ElMessage.success("Metadata 已保存");
    await Promise.all([loadDocumentDetail(), refreshDocuments(documentList.page), refreshAskDocuments()]);
  } catch (error) {
    ElMessage.error(error.message);
  }
}

async function parseDocument(row = detailState.document) {
  if (!row?.id) return;
  detailState.parseLoading = true;
  try {
    const params = buildDbParams();
    const suffix = params.toString();
    await requestJson(
      `/api/documents/${encodeURIComponent(row.id)}/parse${suffix ? `?${suffix}` : ""}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "full", force: true }),
      },
    );
    ElMessage.success("文档解析已完成");
    await Promise.all([
      refreshDocuments(documentList.page),
      refreshAskDocuments(),
      detailState.visible ? loadDocumentDetail(row.id) : Promise.resolve(),
      detailState.visible ? loadDocumentPages(row.id, 1) : Promise.resolve(),
    ]);
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    detailState.parseLoading = false;
  }
}

async function deleteDocument(row) {
  try {
    await ElMessageBox.confirm(`确定删除「${row.original_filename}」吗？`, "删除文档", {
      confirmButtonText: "删除",
      cancelButtonText: "取消",
      type: "warning",
    });
  } catch {
    return;
  }
  try {
    const params = buildDbParams();
    const suffix = params.toString();
    await requestJson(
      `/api/documents/${encodeURIComponent(row.id)}${suffix ? `?${suffix}` : ""}`,
      { method: "DELETE" },
    );
    askForm.documentIds = askForm.documentIds.filter((id) => id !== row.id);
    if (detailState.document?.id === row.id) detailState.visible = false;
    ElMessage.success("文档已删除");
    await Promise.all([refreshDocuments(documentList.page), refreshAskDocuments()]);
  } catch (error) {
    ElMessage.error(error.message);
  }
}

function closeEventStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

async function startAskSession() {
  if (!askForm.documentIds.length) {
    ElMessage.warning("请先选择至少一个文档");
    return;
  }
  if (!askForm.task.trim()) {
    ElMessage.warning("请输入问题");
    return;
  }

  closeEventStream();
  stopTypewriter();
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
        collection_id: null,
        db_path: dbPath.value.trim() || null,
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
      } else if (type === "complete") {
        sessionState.status = "completed";
        sessionState.finalAnswer = payload.data.final_result || "";
        sessionState.trace = payload.data.trace || null;
        sessionState.stats = payload.data.stats || null;
        sessionState.showHumanModal = false;
        closeEventStream();
        startTypewriter(sessionState.finalAnswer);
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
      sessionState.error = sessionState.error || "事件流连接已断开";
      ElMessage.warning(sessionState.error);
    }
  };
}

function startTypewriter(text) {
  stopTypewriter();
  if (!text) return;
  let index = 0;
  sessionState.displayedAnswer = "";
  typewriterTimer = window.setInterval(() => {
    const chunkSize = text.length > 1200 ? 8 : 3;
    sessionState.displayedAnswer = text.slice(0, index + chunkSize);
    index += chunkSize;
    if (index >= text.length) {
      sessionState.displayedAnswer = text;
      stopTypewriter();
    }
    nextTick(scrollAnswerToBottom);
  }, 18);
}

function stopTypewriter() {
  if (typewriterTimer) {
    window.clearInterval(typewriterTimer);
    typewriterTimer = null;
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
  return pieces.length ? pieces.join(" · ") : shortText(readableFallback(data), 68);
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
    case "ask_human":
      add("问题", data.question);
      add("原因", data.reason);
      add("上下文计划", data.context_plan);
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
      add("文档", data.document_id || data.active_document_id);
      add("页码", data.pages || data.candidate_pages || data.active_ranges);
      add("内容", readableFallback(data));
      break;
  }
  return details;
}

function readableFallback(data) {
  if (!data || typeof data !== "object") return String(data || "");
  const keys = ["message", "reason", "question", "task", "tool_name", "document_id", "summary", "content_preview", "final_result"];
  const parts = [];
  for (const key of keys) {
    if (data[key]) parts.push(`${labelize(key)}：${formatValue(data[key])}`);
  }
  return parts.join("\n");
}

function formatValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    return value.map((item) => (typeof item === "object" ? compactObject(item) : String(item))).join("\n");
  }
  if (typeof value === "object") return compactObject(value);
  return String(value);
}

function compactObject(value) {
  if (!value || typeof value !== "object") return String(value || "");
  return Object.entries(value)
    .map(([key, item]) => `${labelize(key)}：${typeof item === "object" ? JSON.stringify(item, null, 2) : item}`)
    .join("\n");
}

function labelize(value) {
  return String(value).replaceAll("_", " ");
}

function shortText(value, maxLength = 48) {
  const text = formatValue(value).replace(/\s+/g, " ").trim();
  if (!text) return "";
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function formatFileSize(size) {
  const bytes = Number(size || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(value) {
  const timestamp = Number(value || 0);
  if (!timestamp) return "-";
  return new Date(timestamp * 1000).toLocaleString();
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
  <el-container class="app-shell">
    <el-header class="app-header">
      <div class="brand">
        <el-icon><Document /></el-icon>
        <span>FsExplorer</span>
      </div>
      <el-menu :default-active="activeView" mode="horizontal" class="top-menu" @select="activeView = $event">
        <el-menu-item index="qa">
          <el-icon><ChatDotRound /></el-icon>
          <span>问答页</span>
        </el-menu-item>
        <el-menu-item index="documents">
          <el-icon><Document /></el-icon>
          <span>文档管理</span>
        </el-menu-item>
      </el-menu>
      <el-popover placement="bottom-end" width="420" trigger="click">
        <template #reference>
          <el-button :icon="Setting">设置</el-button>
        </template>
        <el-form label-position="top">
          <el-form-item label="可选 DB Path">
            <el-input v-model="dbPath" clearable placeholder="留空使用默认数据库" />
          </el-form-item>
          <el-button type="primary" :icon="Refresh" @click="Promise.all([refreshDocuments(1), refreshAskDocuments()])">
            刷新数据
          </el-button>
        </el-form>
      </el-popover>
    </el-header>

    <el-main class="app-main">
      <section v-show="activeView === 'qa'" class="page qa-page">
        <div class="qa-picker">
          <el-card shadow="never" class="toolbar-card">
            <el-form label-position="top">
              <el-form-item label="选择文档">
                <el-select
                  v-model="askForm.documentIds"
                  multiple
                  filterable
                  remote
                  reserve-keyword
                  clearable
                  :remote-method="refreshAskDocuments"
                  :loading="askOptions.loading"
                  placeholder="搜索并选择文档"
                  class="document-select"
                >
                  <el-option
                    v-for="item in askOptions.items"
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
                v-for="doc in selectedDocuments"
                :key="doc.id"
                closable
                @close="askForm.documentIds = askForm.documentIds.filter((id) => id !== doc.id)"
              >
                {{ doc.original_filename }}
              </el-tag>
              <el-text v-if="!selectedDocuments.length" type="info">请选择文档后开始问答</el-text>
            </div>
          </el-card>
        </div>

        <div class="qa-workspace">
          <div class="trace-answer-panel">
            <section class="trace-section">
              <div class="section-title">
                <div>
                  <h2>Trace</h2>
                  <span>运行过程实时追加，展开可查看完整文字信息</span>
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
                  <span>{{ sessionState.error || "完成后以打字效果输出，并按 Markdown 渲染" }}</span>
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
            <el-button type="primary" :loading="isAsking" :icon="ChatDotRound" @click="startAskSession">
              提问
            </el-button>
          </div>
        </div>
      </section>

      <section v-show="activeView === 'documents'" class="page documents-page">
        <el-card shadow="never" class="toolbar-card document-toolbar">
          <div class="toolbar-left">
            <el-input
              v-model="documentList.query"
              clearable
              placeholder="按文件名或 metadata 搜索"
              :prefix-icon="Search"
              @keyup.enter="refreshDocuments(1)"
              @clear="refreshDocuments(1)"
            />
            <el-button type="primary" :icon="Search" @click="refreshDocuments(1)">搜索</el-button>
            <el-button :icon="Refresh" @click="refreshDocuments(documentList.page)">刷新</el-button>
          </div>
          <div class="toolbar-right">
            <el-upload
              :http-request="uploadDocument"
              :on-progress="handleUploadProgress"
              :show-file-list="false"
              :disabled="uploadState.uploading"
            >
              <el-button type="primary" :loading="uploadState.uploading" :icon="Upload">上传并解析</el-button>
            </el-upload>
            <el-progress
              v-if="uploadState.uploading || uploadState.progress === 100"
              :percentage="uploadState.progress"
              :stroke-width="8"
              class="upload-progress"
            />
          </div>
        </el-card>

        <el-card shadow="never" class="table-card">
          <el-table
            v-loading="documentList.loading"
            :data="documentList.items"
            height="100%"
            border
            stripe
            row-key="id"
            empty-text="暂无文档"
          >
            <el-table-column prop="original_filename" label="文件名" min-width="260" show-overflow-tooltip />
            <el-table-column prop="content_type" label="类型" min-width="150" show-overflow-tooltip>
              <template #default="{ row }">{{ row.content_type || "-" }}</template>
            </el-table-column>
            <el-table-column prop="status" label="状态" width="130">
              <template #default="{ row }">
                <el-tag :type="documentStatusType(row.status)">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="page_count" label="页数" width="90" align="right" />
            <el-table-column prop="file_size" label="大小" width="120" align="right">
              <template #default="{ row }">{{ formatFileSize(row.file_size) }}</template>
            </el-table-column>
            <el-table-column prop="file_mtime" label="更新时间" width="190">
              <template #default="{ row }">{{ formatTime(row.file_mtime) }}</template>
            </el-table-column>
            <el-table-column label="操作" width="260" fixed="right">
              <template #default="{ row }">
                <el-button size="small" :icon="View" @click="openDocumentDetail(row)">查看</el-button>
                <el-button size="small" :loading="detailState.parseLoading && detailState.document?.id === row.id" @click="parseDocument(row)">
                  解析
                </el-button>
                <el-button size="small" type="danger" :icon="Delete" @click="deleteDocument(row)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <div class="pagination-bar">
          <el-pagination
            v-model:current-page="documentList.page"
            v-model:page-size="documentList.pageSize"
            :page-sizes="[10, 20, 50, 100]"
            :total="documentList.total"
            layout="total, sizes, prev, pager, next, jumper"
            @size-change="handlePageSizeChange"
            @current-change="refreshDocuments"
          />
        </div>
      </section>
    </el-main>
  </el-container>

  <el-drawer v-model="detailState.visible" size="62%" title="文档详情" class="document-drawer">
    <div v-loading="detailState.loading" class="drawer-content">
      <template v-if="detailState.document">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="文件名">{{ detailState.document.original_filename }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="documentStatusType(detailState.document.status)">{{ detailState.document.status }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="页数">{{ detailState.pageSummary?.page_count || detailState.document.page_count || 0 }}</el-descriptions-item>
          <el-descriptions-item label="大小">{{ formatFileSize(detailState.document.file_size) }}</el-descriptions-item>
          <el-descriptions-item label="Storage URI" :span="2">{{ detailState.document.storage_uri }}</el-descriptions-item>
        </el-descriptions>

        <div class="drawer-actions">
          <el-button type="primary" :loading="detailState.parseLoading" @click="parseDocument(detailState.document)">重新解析</el-button>
          <el-button @click="loadDocumentPages(detailState.document.id, detailState.pagesPage)">刷新页面预览</el-button>
          <el-button type="success" @click="saveMetadata">保存 Metadata</el-button>
        </div>

        <el-tabs>
          <el-tab-pane label="Metadata">
            <el-input v-model="detailState.metadataDraft" type="textarea" :rows="18" resize="vertical" spellcheck="false" />
          </el-tab-pane>
          <el-tab-pane label="页面预览">
            <div class="page-preview-list">
              <el-card v-for="page in detailState.pages" :key="`${page.page_no}-${page.content_hash}`" shadow="never">
                <template #header>
                  <div class="page-preview-header">
                    <strong>Page {{ page.page_no }}</strong>
                    <span>{{ page.heading || page.page_label || "Untitled" }}</span>
                  </div>
                </template>
                <pre class="page-markdown">{{ page.markdown }}</pre>
              </el-card>
              <el-empty v-if="!detailState.pages.length" description="暂无页面内容" />
            </div>
            <div class="drawer-pagination">
              <el-pagination
                v-model:current-page="detailState.pagesPage"
                :page-size="detailState.pagesPageSize"
                :total="detailState.pagesTotal"
                layout="total, prev, pager, next"
                @current-change="(page) => loadDocumentPages(detailState.document.id, page)"
              />
            </div>
          </el-tab-pane>
        </el-tabs>
      </template>
    </div>
  </el-drawer>

  <el-dialog v-model="sessionState.showHumanModal" title="需要人工回复" width="560px">
    <p class="human-question">{{ sessionState.humanQuestion }}</p>
    <el-input v-model="sessionState.humanResponse" type="textarea" :rows="5" placeholder="请输入回复" />
    <template #footer>
      <el-button @click="sessionState.showHumanModal = false">稍后</el-button>
      <el-button type="primary" :loading="sessionState.replying" @click="submitHumanReply">发送回复</el-button>
    </template>
  </el-dialog>
</template>

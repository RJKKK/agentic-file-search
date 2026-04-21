<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

const dbPath = ref("");
const activeDocumentId = ref("");
const activeCollectionId = ref("");
const askCollectionId = ref("");
const askDocumentIds = ref([]);
const activeSessionId = ref("");
let eventSource = null;

const toast = reactive({ kind: "", message: "" });

const uploadForm = reactive({
  file: null,
  loading: false,
  error: "",
});

const documentList = reactive({
  items: [],
  query: "",
  page: 1,
  pageSize: 20,
  total: 0,
  loading: false,
});

const collectionState = reactive({
  items: [],
  loading: false,
  createName: "",
});

const detailState = reactive({
  loading: false,
  document: null,
  parseSummary: null,
  pages: [],
  pagesPage: 1,
  pagesPageSize: 6,
  pagesTotal: 0,
  metadataDraft: "{}",
});

const parseForm = reactive({
  mode: "incremental",
  force: false,
  focusHint: "",
  anchor: "",
  window: 1,
  maxUnits: "",
});

const askForm = reactive({
  task: "",
  enableSemantic: false,
  enableMetadata: false,
});

const sessionState = reactive({
  status: "idle",
  steps: [],
  result: "",
  error: "",
  trace: null,
  stats: null,
  showHumanModal: false,
  humanQuestion: "",
  humanResponse: "",
  replying: false,
});

const activeDocument = computed(() =>
  documentList.items.find((item) => item.id === activeDocumentId.value) ||
  detailState.document,
);

const activeCollection = computed(() =>
  collectionState.items.find((item) => item.id === activeCollectionId.value) || null,
);

const selectedScopeCount = computed(() => {
  const docCount = askDocumentIds.value.length;
  return docCount + (askCollectionId.value ? 1 : 0);
});

const pageRangeLabel = computed(() => {
  if (!detailState.pagesTotal) return "No pages generated yet";
  const start = (detailState.pagesPage - 1) * detailState.pagesPageSize + 1;
  const end = Math.min(
    detailState.pagesTotal,
    detailState.pagesPage * detailState.pagesPageSize,
  );
  return `${start}-${end} / ${detailState.pagesTotal}`;
});

const citedSources = computed(() => {
  const raw = sessionState.trace?.cited_sources;
  return Array.isArray(raw) ? raw : [];
});

const referencedDocuments = computed(() => {
  const raw = sessionState.trace?.referenced_documents;
  return Array.isArray(raw) ? raw : [];
});

const contextScope = computed(() => {
  const raw = sessionState.trace?.context_scope || sessionState.stats?.context_scope;
  return raw && typeof raw === "object" ? raw : null;
});

const coverageEntries = computed(() => {
  const raw = sessionState.trace?.coverage_by_document;
  if (!raw || typeof raw !== "object") return [];
  return Object.entries(raw);
});

const compactionActions = computed(() => {
  const raw = sessionState.trace?.compaction_actions;
  return Array.isArray(raw) ? raw : [];
});

const promotedEvidenceUnits = computed(() => {
  const raw = sessionState.trace?.promoted_evidence_units;
  return Array.isArray(raw) ? raw : [];
});

const formattedResult = computed(() => {
  const escaped = escapeHtml(sessionState.result || "No answer yet.");
  return escaped.replace(
    /\[Source:[^\]]+\]/g,
    (match) => `<mark class="result-citation">${match}</mark>`,
  );
});

onMounted(async () => {
  await Promise.all([refreshDocuments(), refreshCollections()]);
});

onBeforeUnmount(() => {
  closeEventStream();
});

function notify(kind, message) {
  toast.kind = kind;
  toast.message = message;
  window.clearTimeout(notify._timer);
  notify._timer = window.setTimeout(() => {
    toast.kind = "";
    toast.message = "";
  }, 3200);
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

function buildDbParams() {
  const params = new URLSearchParams();
  if (dbPath.value.trim()) params.set("db_path", dbPath.value.trim());
  return params;
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
    if (activeDocumentId.value && !documentList.items.some((item) => item.id === activeDocumentId.value)) {
      activeDocumentId.value = "";
      detailState.document = null;
      detailState.pages = [];
    }
    if (!activeDocumentId.value && documentList.items.length) {
      await selectDocument(documentList.items[0].id);
    }
  } catch (error) {
    notify("error", error.message);
  } finally {
    documentList.loading = false;
  }
}

async function refreshCollections() {
  collectionState.loading = true;
  try {
    const params = buildDbParams();
    const suffix = params.toString();
    const payload = await requestJson(`/api/collections${suffix ? `?${suffix}` : ""}`);
    collectionState.items = payload.items || [];
    if (
      activeCollectionId.value &&
      !collectionState.items.some((item) => item.id === activeCollectionId.value)
    ) {
      activeCollectionId.value = "";
    }
    if (
      askCollectionId.value &&
      !collectionState.items.some((item) => item.id === askCollectionId.value)
    ) {
      askCollectionId.value = "";
    }
  } catch (error) {
    notify("error", error.message);
  } finally {
    collectionState.loading = false;
  }
}

async function selectDocument(docId) {
  activeDocumentId.value = docId;
  await Promise.all([loadDocumentDetail(docId), loadDocumentPages(docId, 1)]);
}

async function loadDocumentDetail(docId = activeDocumentId.value) {
  if (!docId) return;
  detailState.loading = true;
  try {
    const params = buildDbParams();
    const suffix = params.toString();
    const payload = await requestJson(
      `/api/documents/${encodeURIComponent(docId)}${suffix ? `?${suffix}` : ""}`,
    );
    detailState.document = payload.document;
    detailState.parseSummary = payload.page_summary;
    detailState.metadataDraft = JSON.stringify(payload.document.metadata || {}, null, 2);
  } catch (error) {
    notify("error", error.message);
  } finally {
    detailState.loading = false;
  }
}

async function loadDocumentPages(docId = activeDocumentId.value, page = 1) {
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
    notify("error", error.message);
  } finally {
    detailState.loading = false;
  }
}

async function uploadDocument() {
  if (!uploadForm.file) return;
  uploadForm.loading = true;
  uploadForm.error = "";
  try {
    const formData = new FormData();
    formData.append("file", uploadForm.file);
    if (dbPath.value.trim()) formData.append("db_path", dbPath.value.trim());
    const payload = await requestJson("/api/documents", { method: "POST", body: formData });
    uploadForm.file = null;
    notify("success", `${payload.document.original_filename} uploaded.`);
    await Promise.all([refreshDocuments(1), refreshCollections()]);
  } catch (error) {
    uploadForm.error = error.message;
    notify("error", error.message);
  } finally {
    uploadForm.loading = false;
  }
}

async function saveMetadata() {
  if (!activeDocumentId.value) return;
  try {
    const metadata = JSON.parse(detailState.metadataDraft || "{}");
    const params = buildDbParams();
    await requestJson(
      `/api/documents/${encodeURIComponent(activeDocumentId.value)}${params.toString() ? `?${params.toString()}` : ""}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ metadata }),
      },
    );
    notify("success", "Metadata updated.");
    await Promise.all([refreshDocuments(documentList.page), loadDocumentDetail()]);
  } catch (error) {
    notify("error", error.message);
  }
}

async function parseDocumentUnits() {
  if (!activeDocumentId.value) return;
  try {
    const params = buildDbParams();
    await requestJson(
      `/api/documents/${encodeURIComponent(activeDocumentId.value)}/parse${params.toString() ? `?${params.toString()}` : ""}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: parseForm.mode,
          force: parseForm.force,
          focus_hint: parseForm.focusHint || null,
          anchor: parseForm.anchor ? Number(parseForm.anchor) : null,
          window: Number(parseForm.window || 1),
          max_units: parseForm.maxUnits ? Number(parseForm.maxUnits) : null,
        }),
      },
    );
    notify("success", "Page files rebuilt.");
    await Promise.all([loadDocumentDetail(), loadDocumentPages(activeDocumentId.value, 1), refreshDocuments(documentList.page)]);
  } catch (error) {
    notify("error", error.message);
  }
}

async function deleteDocument() {
  if (!activeDocumentId.value) return;
  if (!window.confirm("Remove this document from the library?")) return;
  try {
    const params = buildDbParams();
    await requestJson(
      `/api/documents/${encodeURIComponent(activeDocumentId.value)}${params.toString() ? `?${params.toString()}` : ""}`,
      { method: "DELETE" },
    );
    askDocumentIds.value = askDocumentIds.value.filter((id) => id !== activeDocumentId.value);
    activeDocumentId.value = "";
    notify("success", "Document removed from active library views.");
    await Promise.all([refreshDocuments(documentList.page), refreshCollections()]);
  } catch (error) {
    notify("error", error.message);
  }
}

function toggleAskDocument(docId) {
  if (askDocumentIds.value.includes(docId)) {
    askDocumentIds.value = askDocumentIds.value.filter((id) => id !== docId);
  } else {
    askDocumentIds.value = [...askDocumentIds.value, docId];
  }
}

async function createCollection() {
  const name = collectionState.createName.trim();
  if (!name) return;
  try {
    const params = buildDbParams();
    const payload = await requestJson(
      `/api/collections${params.toString() ? `?${params.toString()}` : ""}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      },
    );
    collectionState.createName = "";
    activeCollectionId.value = payload.collection.id;
    notify("success", "Collection created.");
    await refreshCollections();
  } catch (error) {
    notify("error", error.message);
  }
}

async function renameCollection(collection) {
  const name = window.prompt("Rename collection", collection.name);
  if (!name || !name.trim()) return;
  try {
    const params = buildDbParams();
    await requestJson(
      `/api/collections/${encodeURIComponent(collection.id)}${params.toString() ? `?${params.toString()}` : ""}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      },
    );
    notify("success", "Collection renamed.");
    await refreshCollections();
  } catch (error) {
    notify("error", error.message);
  }
}

async function deleteCollection(collection) {
  if (!window.confirm(`Delete collection "${collection.name}"?`)) return;
  try {
    const params = buildDbParams();
    await requestJson(
      `/api/collections/${encodeURIComponent(collection.id)}${params.toString() ? `?${params.toString()}` : ""}`,
      { method: "DELETE" },
    );
    if (activeCollectionId.value === collection.id) activeCollectionId.value = "";
    if (askCollectionId.value === collection.id) askCollectionId.value = "";
    notify("success", "Collection deleted.");
    await refreshCollections();
  } catch (error) {
    notify("error", error.message);
  }
}

async function addCheckedDocsToCollection(collectionId = activeCollectionId.value) {
  if (!collectionId || !askDocumentIds.value.length) return;
  try {
    const params = buildDbParams();
    await requestJson(
      `/api/collections/${encodeURIComponent(collectionId)}/documents${params.toString() ? `?${params.toString()}` : ""}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_ids: askDocumentIds.value }),
      },
    );
    notify("success", "Documents added to collection.");
  } catch (error) {
    notify("error", error.message);
  }
}

async function removeDocumentFromCollection(collectionId, docId) {
  try {
    const params = buildDbParams();
    await requestJson(
      `/api/collections/${encodeURIComponent(collectionId)}/documents/${encodeURIComponent(docId)}${params.toString() ? `?${params.toString()}` : ""}`,
      { method: "DELETE" },
    );
    notify("success", "Document removed from collection.");
  } catch (error) {
    notify("error", error.message);
  }
}

function closeEventStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

async function startAskSession() {
  if (!askForm.task.trim()) {
    notify("error", "Please enter a question.");
    return;
  }
  if (!askDocumentIds.value.length && !askCollectionId.value) {
    notify("error", "Select at least one temporary document or one collection.");
    return;
  }
  closeEventStream();
  sessionState.status = "creating";
  sessionState.steps = [];
  sessionState.result = "";
  sessionState.error = "";
  sessionState.trace = null;
  sessionState.stats = null;

  try {
    const payload = await requestJson("/api/explore/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: askForm.task.trim(),
        document_ids: askDocumentIds.value,
        collection_id: askCollectionId.value || null,
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
    notify("error", error.message);
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
      sessionState.steps = [...sessionState.steps, payload];
      if (type === "ask_human") {
        sessionState.status = "awaiting_human";
        sessionState.showHumanModal = true;
        sessionState.humanQuestion = payload.data.question;
      } else if (type === "complete") {
        sessionState.status = "completed";
        sessionState.result = payload.data.final_result || "";
        sessionState.trace = payload.data.trace || null;
        sessionState.stats = payload.data.stats || null;
        sessionState.showHumanModal = false;
        closeEventStream();
      } else if (type === "error") {
        sessionState.status = "error";
        sessionState.error = payload.data.message || "Session failed.";
        closeEventStream();
      } else {
        sessionState.status = type.includes("lazy") ? "indexing" : "running";
      }
    });
  }

  eventSource.onerror = () => {
    if (sessionState.status !== "completed") {
      sessionState.error = sessionState.error || "Event stream disconnected.";
    }
  };
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
    notify("error", error.message);
  } finally {
    sessionState.replying = false;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
</script>

<template>
  <div class="app-shell library-shell">
    <div class="ambient ambient--left" />
    <div class="ambient ambient--right" />

    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">FsExplorer Document Library</div>
        <h1>Upload once. Ask against the exact documents you choose.</h1>
        <p class="lede">
          The product flow now centers on a shared document library, one optional collection,
          and temporary multi-select documents for each question.
        </p>
      </div>
      <div class="hero-card">
        <div class="hero-card__label">Current Scope</div>
        <div class="metric-grid metric-grid--tight">
          <div class="metric-card">
            <span class="metric-label">Documents</span>
            <strong>{{ documentList.total }}</strong>
          </div>
          <div class="metric-card">
            <span class="metric-label">Collections</span>
            <strong>{{ collectionState.items.length }}</strong>
          </div>
          <div class="metric-card">
            <span class="metric-label">Ask Scope</span>
            <strong>{{ selectedScopeCount }}</strong>
          </div>
          <div class="metric-card">
            <span class="metric-label">Session</span>
            <strong>{{ sessionState.status }}</strong>
          </div>
        </div>
        <label class="field">
          <span class="field-label">Optional DB Path</span>
          <input v-model="dbPath" class="input" placeholder="Leave blank for default database" />
        </label>
      </div>
    </section>

    <section class="library-grid">
      <aside class="panel library-column">
        <div class="panel-head">
          <div>
            <div class="panel-kicker">Library</div>
            <h2>Documents</h2>
          </div>
          <span class="status-pill" :data-state="documentList.loading ? 'running' : 'ready'">
            {{ documentList.loading ? "Loading" : `${documentList.total} files` }}
          </span>
        </div>

        <div class="stack">
          <div class="field">
            <span class="field-label">Search Library</span>
            <input
              v-model="documentList.query"
              class="input"
              placeholder="Filter by filename or metadata"
              @keyup.enter="refreshDocuments(1)"
            />
          </div>
          <div class="upload-row">
            <input type="file" @change="uploadForm.file = $event.target.files?.[0] ?? null" />
            <button class="button button--primary" :disabled="uploadForm.loading || !uploadForm.file" @click="uploadDocument">
              {{ uploadForm.loading ? "Uploading..." : "Upload" }}
            </button>
          </div>
          <p v-if="uploadForm.error" class="support-copy support-copy--error">{{ uploadForm.error }}</p>
        </div>

        <div class="document-stack">
          <button
            v-for="document in documentList.items"
            :key="document.id"
            class="document-card"
            :data-active="document.id === activeDocumentId"
            @click="selectDocument(document.id)"
          >
            <div class="document-card__top">
              <label class="checkbox-line" @click.stop>
                <input
                  type="checkbox"
                  :checked="askDocumentIds.includes(document.id)"
                  @change="toggleAskDocument(document.id)"
                />
                <span>Ask with this doc</span>
              </label>
              <span class="status-pill" :data-state="document.status">{{ document.status }}</span>
            </div>
            <strong>{{ document.original_filename }}</strong>
            <div class="support-copy">{{ document.content_type || "unknown type" }}</div>
          </button>
        </div>
      </aside>

      <main class="panel detail-column">
        <div class="panel-head">
          <div>
            <div class="panel-kicker">Document Detail</div>
            <h2>{{ activeDocument?.original_filename || "Select a document" }}</h2>
          </div>
          <span class="status-pill" :data-state="activeDocument?.status || 'idle'">
            {{ activeDocument?.status || "idle" }}
          </span>
        </div>

        <div v-if="activeDocument" class="stack">
          <div class="detail-grid detail-grid--hero">
            <div class="detail-stat">
              <span class="metric-label">Storage URI</span>
              <strong>{{ activeDocument.storage_uri }}</strong>
            </div>
            <div class="detail-stat">
              <span class="metric-label">Pages</span>
              <strong>{{ detailState.parseSummary?.page_count || 0 }}</strong>
            </div>
            <div class="detail-stat">
              <span class="metric-label">File Size</span>
              <strong>{{ activeDocument.file_size }}</strong>
            </div>
          </div>

          <div class="field-grid">
            <label class="field">
              <span class="field-label">Focus Hint</span>
              <input v-model="parseForm.focusHint" class="input" placeholder="Optional note for rebuild history" />
            </label>
            <label class="field">
              <span class="field-label">Anchor Page</span>
              <input v-model="parseForm.anchor" class="input" placeholder="1" />
            </label>
            <label class="field">
              <span class="field-label">Window</span>
              <input v-model="parseForm.window" class="input" type="number" min="0" />
            </label>
            <label class="field">
              <span class="field-label">Max Pages</span>
              <input v-model="parseForm.maxUnits" class="input" placeholder="Optional" />
            </label>
          </div>

          <div class="upload-row">
            <button class="button button--primary" @click="parseDocumentUnits">Rebuild Pages</button>
            <button class="button button--ghost" @click="saveMetadata">Save Metadata</button>
            <button class="button button--danger" @click="deleteDocument">Delete Document</button>
          </div>

          <label class="field">
            <span class="field-label">Metadata JSON</span>
            <textarea v-model="detailState.metadataDraft" class="textarea textarea--tall" />
          </label>

          <div class="panel-head panel-head--compact">
            <h3>Stored Pages</h3>
            <span class="support-copy">{{ pageRangeLabel }}</span>
          </div>
          <div class="page-grid">
            <article v-for="page in detailState.pages" :key="`${page.page_no}-${page.content_hash}`" class="page-card">
              <div class="page-card__meta">Page {{ page.page_no }} · {{ page.heading || "Untitled" }}</div>
              <pre class="page-card__body">{{ page.markdown }}</pre>
            </article>
          </div>
        </div>

        <div v-else class="empty-card">
          <strong>Select one uploaded document.</strong>
          <p>We’ll show stored page files, page summary, and editable metadata here.</p>
        </div>
      </main>

      <aside class="panel ask-column">
        <div class="panel-head">
          <div>
            <div class="panel-kicker">Question Scope</div>
            <h2>Collections + Temporary Docs</h2>
          </div>
          <span class="status-pill" :data-state="sessionState.status">{{ sessionState.status }}</span>
        </div>

        <div class="stack">
          <label class="field">
            <span class="field-label">Create Collection</span>
            <div class="upload-row">
              <input v-model="collectionState.createName" class="input" placeholder="Board pack" />
              <button class="button button--primary" @click="createCollection">Create</button>
            </div>
          </label>

          <div class="collection-stack">
            <article
              v-for="collection in collectionState.items"
              :key="collection.id"
              class="collection-card"
              :data-active="collection.id === activeCollectionId"
            >
              <button class="collection-card__header" @click="activeCollectionId = collection.id">
                <strong>{{ collection.name }}</strong>
                <span class="support-copy">Saved scope</span>
              </button>
              <div class="checkbox-line">
                <input
                  :id="`ask-collection-${collection.id}`"
                  :checked="askCollectionId === collection.id"
                  type="radio"
                  name="askCollection"
                  @change="askCollectionId = collection.id"
                />
                <label :for="`ask-collection-${collection.id}`">Use in this question</label>
              </div>
              <div class="upload-row">
                <button class="button button--ghost" @click="renameCollection(collection)">Rename</button>
                <button class="button button--ghost" @click="addCheckedDocsToCollection(collection.id)">Add checked docs</button>
                <button class="button button--danger" @click="deleteCollection(collection)">Delete</button>
              </div>
            </article>
          </div>

          <div v-if="activeCollection" class="collection-members">
            <div class="panel-head panel-head--compact">
              <h3>{{ activeCollection.name }}</h3>
              <span class="support-copy">Manage attached docs</span>
            </div>
            <CollectionDocuments
              :collection-id="activeCollection.id"
              :db-path="dbPath"
              @remove-document="removeDocumentFromCollection"
            />
          </div>

          <label class="field">
            <span class="field-label">Question</span>
            <textarea
              v-model="askForm.task"
              class="textarea"
              placeholder="例如：请输出所有董事会成员"
            />
          </label>

          <div class="checkbox-wrap">
            <label class="checkbox-line">
              <input v-model="askForm.enableSemantic" type="checkbox" />
              <span>Enable legacy semantic retrieval</span>
            </label>
            <label class="checkbox-line">
              <input v-model="askForm.enableMetadata" type="checkbox" />
              <span>Enable metadata filters</span>
            </label>
          </div>

          <button class="button button--primary button--wide" @click="startAskSession">
            Ask Selected Scope
          </button>

          <div class="result-card">
            <div class="panel-head panel-head--compact">
              <h3>Answer</h3>
              <span class="support-copy">{{ sessionState.error || "SSE timeline stays live here." }}</span>
            </div>
            <div class="result-body" v-html="formattedResult" />
          </div>

          <div v-if="citedSources.length" class="reference-grid">
            <div class="info-card" v-for="source in citedSources" :key="source">
              <span class="metric-label">Citation</span>
              <strong>{{ source }}</strong>
            </div>
          </div>

          <div v-if="referencedDocuments.length" class="reference-grid">
            <div class="info-card" v-for="doc in referencedDocuments" :key="doc">
              <span class="metric-label">Referenced</span>
              <strong>{{ doc }}</strong>
            </div>
          </div>

          <div v-if="contextScope" class="reference-grid">
            <div class="info-card">
              <span class="metric-label">Active document</span>
              <strong>{{ contextScope.active_document_id || contextScope.active_file_path || "None" }}</strong>
            </div>
            <div class="info-card">
              <span class="metric-label">Active ranges</span>
              <strong>
                {{
                  Array.isArray(contextScope.active_ranges) && contextScope.active_ranges.length
                    ? contextScope.active_ranges.map((item) =>
                        item.start === item.end ? `${item.start}` : `${item.start}-${item.end}`,
                      ).join(", ")
                    : "None"
                }}
              </strong>
            </div>
          </div>

          <div v-if="coverageEntries.length" class="timeline-list">
            <article v-for="[docId, coverage] in coverageEntries" :key="docId" class="timeline-item">
              <div class="timeline-item__meta">coverage · {{ coverage.label || docId }}</div>
              <pre class="timeline-item__body">{{ JSON.stringify(coverage, null, 2) }}</pre>
            </article>
          </div>

          <div v-if="compactionActions.length" class="reference-grid">
            <div class="info-card" v-for="(action, index) in compactionActions" :key="`${action.action || 'action'}-${index}`">
              <span class="metric-label">Compaction</span>
              <strong>{{ action.action || "context change" }}</strong>
              <span class="support-copy">{{ action.reason || "" }}</span>
            </div>
          </div>

          <div v-if="promotedEvidenceUnits.length" class="reference-grid">
            <div class="info-card" v-for="item in promotedEvidenceUnits" :key="item">
              <span class="metric-label">Promoted evidence</span>
              <strong>{{ item }}</strong>
            </div>
          </div>

          <div class="timeline-list">
            <article v-for="step in sessionState.steps" :key="`${step.sequence}-${step.type}`" class="timeline-item">
              <div class="timeline-item__meta">{{ step.type }} · #{{ step.sequence }}</div>
              <pre class="timeline-item__body">{{ JSON.stringify(step.data, null, 2) }}</pre>
            </article>
          </div>
        </div>
      </aside>
    </section>

    <div v-if="sessionState.showHumanModal" class="modal-shell">
      <div class="modal-card">
        <div class="panel-kicker">Human Reply</div>
        <h3>{{ sessionState.humanQuestion }}</h3>
        <textarea v-model="sessionState.humanResponse" class="textarea" placeholder="Type the answer for the agent" />
        <div class="upload-row">
          <button class="button button--primary" :disabled="sessionState.replying" @click="submitHumanReply">
            {{ sessionState.replying ? "Sending..." : "Send Reply" }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="toast.message" class="toast" :data-kind="toast.kind">
      {{ toast.message }}
    </div>
  </div>
</template>

<script>
export default {
  components: {
    CollectionDocuments: {
      props: {
        collectionId: { type: String, required: true },
        dbPath: { type: Object, required: true },
      },
      emits: ["remove-document"],
      data() {
        return { items: [], loading: false };
      },
      watch: {
        collectionId: {
          immediate: true,
          handler() {
            this.load();
          },
        },
        dbPath: {
          deep: false,
          handler() {
            this.load();
          },
        },
      },
      methods: {
        async load() {
          if (!this.collectionId) return;
          this.loading = true;
          const params = new URLSearchParams();
          if (this.dbPath?.trim?.()) params.set("db_path", this.dbPath.trim());
          const response = await fetch(`/api/collections/${encodeURIComponent(this.collectionId)}/documents${params.toString() ? `?${params.toString()}` : ""}`);
          const payload = await response.json().catch(() => ({}));
          this.items = payload.items || [];
          this.loading = false;
        },
      },
      template: `
        <div class="member-stack">
          <div v-if="!items.length" class="empty-card compact-empty">
            <strong>No documents attached yet.</strong>
          </div>
          <div v-for="item in items" :key="item.id" class="member-card">
            <div>
              <strong>{{ item.original_filename }}</strong>
              <div class="support-copy">{{ item.status }}</div>
            </div>
            <button class="button button--ghost" @click="$emit('remove-document', collectionId, item.id)">Remove</button>
          </div>
        </div>
      `,
    },
  },
};
</script>

<script setup>
import { Collection, Document, Refresh, Search } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import MarkdownIt from "markdown-it";
import { computed, onMounted, reactive, ref, watch } from "vue";

import { documentStatusType, requestJson } from "../api.js";
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

const retrievalState = reactive({
  loading: false,
  results: [],
  warnings: [],
  error: "",
});
const promptState = reactive({
  preview: null,
  error: "",
});
const activeTab = ref("results");
const promptViewMode = ref("structured");

const resultCountText = computed(() => `${retrievalState.results.length} results`);
const promptStatusText = computed(() => {
  if (promptState.error) {
    return promptState.error;
  }
  if (!promptState.preview) {
    return "Prompt preview will appear here.";
  }
  return `${promptState.preview.messages?.length || 0} message(s) ready`;
});

function renderDebugMarkdown(value) {
  return markdown.render(String(value || ""));
}

function buildRetrievalPayload() {
  return {
    question: askForm.task.trim(),
    mode: askForm.retrievalMode,
    document_ids: askForm.documentIds,
    collection_ids: askForm.collectionIds,
    keyword_weight: Number(askForm.keywordWeight || 0.5),
    semantic_weight: Number(askForm.semanticWeight || 0.5),
  };
}

onMounted(async () => {
  await Promise.all([refreshAskDocuments(), refreshCollections()]);
});

watch(
  () => appState.refreshTick,
  () => Promise.all([refreshAskDocuments(), refreshCollections()]),
);

async function runRetrieval() {
  if (!askForm.documentIds.length && !askForm.collectionIds.length) {
    ElMessage.warning("Please select at least one document or collection.");
    return;
  }
  if (!askForm.task.trim()) {
    ElMessage.warning("Please enter a question.");
    return;
  }
  retrievalState.loading = true;
  retrievalState.results = [];
  retrievalState.warnings = [];
  retrievalState.error = "";
  promptState.preview = null;
  promptState.error = "";
  activeTab.value = "results";
  promptViewMode.value = "structured";
  const payload = buildRetrievalPayload();
  try {
    const [retrievalResult, promptResult] = await Promise.allSettled([
      requestJson("/api/rag/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      requestJson("/api/rag/prompt-preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ]);

    if (retrievalResult.status === "fulfilled") {
      retrievalState.results = retrievalResult.value.retrieved_chunks || [];
      retrievalState.warnings = retrievalResult.value.warnings || [];
    } else {
      retrievalState.error = retrievalResult.reason?.message || "Failed to load retrieval results.";
      ElMessage.error(retrievalState.error);
    }

    if (promptResult.status === "fulfilled") {
      promptState.preview = promptResult.value;
    } else {
      promptState.error = promptResult.reason?.message || "Failed to load prompt preview.";
    }
  } finally {
    retrievalState.loading = false;
  }
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
          <div class="rag-controls">
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
          <el-button :icon="Refresh" @click="Promise.all([refreshAskDocuments(), refreshCollections()])">
            Refresh
          </el-button>
        </div>
      </el-card>
    </div>

    <div class="qa-workspace">
      <div class="trace-answer-panel retrieval-panel">
        <section class="answer-section">
          <div class="section-title">
            <div>
              <h2>召回调试</h2>
              <span>{{ activeTab === "results" ? retrievalState.error || resultCountText : promptStatusText }}</span>
            </div>
            <el-button type="primary" :icon="Search" :loading="retrievalState.loading" @click="runRetrieval">
              开始召回
            </el-button>
          </div>
          <div class="answer-scroll">
            <el-tabs v-model="activeTab" class="retrieval-tabs">
              <el-tab-pane label="Retrieval Results" name="results">
                <el-alert
                  v-for="warning in retrievalState.warnings"
                  :key="warning"
                  type="warning"
                  :closable="false"
                  show-icon
                  class="rag-warning"
                  :title="warning"
                />
                <el-empty
                  v-if="!retrievalState.results.length"
                  description="召回结果会显示在这里"
                  :image-size="82"
                />
                <div v-else class="rag-citations">
                  <div v-for="chunk in retrievalState.results" :key="chunk.reference_id" class="citation-card">
                    <div class="citation-meta">
                      <span>[{{ chunk.citation_no }}]</span>
                      <span>{{ chunk.document_name }}</span>
                      <span>{{ chunk.reference_kind }}</span>
                      <span>score {{ Number(chunk.score || 0).toFixed(4) }}</span>
                      <span>pages {{ (chunk.page_nos || []).join(", ") }}</span>
                      <span>locator {{ chunk.source_locator || "-" }}</span>
                    </div>
                    <div class="citation-links">
                      <span>retrieval ids: {{ (chunk.retrieval_unit_ids || []).join(", ") || "-" }}</span>
                      <a :href="chunk.source_link" target="_blank" rel="noreferrer">Open Chunk</a>
                    </div>
                    <div v-if="chunk.debug_cer_context_refs?.length" class="citation-links">
                      <span>
                        CER contexts:
                        {{
                          chunk.debug_cer_context_refs
                            .map((item) => `${item.reference_kind}:${item.reference_id}@${item.source_locator || "-"}`)
                            .join(" | ")
                        }}
                      </span>
                    </div>
                    <div
                      v-if="chunk.debug_enriched_content"
                      class="markdown-body answer-markdown detail-markdown retrieval-debug-markdown"
                      v-html="renderDebugMarkdown(chunk.debug_enriched_content)"
                    />
                  </div>
                </div>
              </el-tab-pane>

              <el-tab-pane label="Prompt Preview" name="prompt">
                <el-alert
                  v-if="promptState.error"
                  type="error"
                  :closable="false"
                  show-icon
                  class="rag-warning"
                  :title="promptState.error"
                />
                <el-empty
                  v-else-if="!promptState.preview"
                  description="完整送模提示词会显示在这里"
                  :image-size="82"
                />
                <div v-else class="prompt-preview">
                  <div class="citation-meta">
                    <span>variant {{ promptState.preview.prompt_variant }}</span>
                    <span>model {{ promptState.preview.model || "-" }}</span>
                    <span>temperature {{ promptState.preview.temperature }}</span>
                    <span>messages {{ promptState.preview.messages?.length || 0 }}</span>
                  </div>
                  <div class="prompt-toolbar">
                    <el-radio-group v-model="promptViewMode" size="small">
                      <el-radio-button label="structured">分段阅读</el-radio-button>
                      <el-radio-button label="raw">原始 JSON</el-radio-button>
                    </el-radio-group>
                  </div>

                  <template v-if="promptViewMode === 'structured'">
                    <div class="prompt-section">
                      <h3>System Prompt</h3>
                      <pre class="prompt-preview-pre">{{ promptState.preview.system_prompt }}</pre>
                    </div>

                    <div class="prompt-section">
                      <h3>User Prompt</h3>
                      <pre class="prompt-preview-pre">{{ promptState.preview.user_prompt }}</pre>
                    </div>

                    <div class="prompt-section">
                      <h3>Request JSON</h3>
                      <pre class="prompt-preview-pre">{{ promptState.preview.request_body_json }}</pre>
                    </div>
                  </template>

                  <div v-else class="prompt-section">
                    <h3>Raw Request JSON</h3>
                    <pre class="prompt-preview-pre">{{ promptState.preview.request_body_json }}</pre>
                  </div>
                </div>
              </el-tab-pane>
            </el-tabs>
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
        @keydown.ctrl.enter.prevent="runRetrieval"
      />
      <div class="ask-actions">
        <el-button
          type="primary"
          :loading="retrievalState.loading"
          :icon="Search"
          @click="runRetrieval"
        >
          召回测试
        </el-button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.prompt-preview {
  display: grid;
  gap: 16px;
}

.prompt-section h3 {
  margin: 0 0 8px;
  font-size: 14px;
}

.prompt-toolbar {
  display: flex;
  justify-content: flex-start;
}

.prompt-preview-pre {
  margin: 0;
  padding: 12px 14px;
  border-radius: 12px;
  background: #0f172a;
  color: #e2e8f0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.55;
  overflow-x: auto;
}
</style>

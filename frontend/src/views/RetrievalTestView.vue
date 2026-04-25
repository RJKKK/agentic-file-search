<script setup>
import { Collection, Document, Refresh, Search } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import { computed, onMounted, reactive, watch } from "vue";

import { documentStatusType, requestJson } from "../api.js";
import { useRetrievalScope } from "../composables/useRetrievalScope.js";

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

const resultCountText = computed(() => `${retrievalState.results.length} results`);

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
  try {
    const payload = await requestJson("/api/rag/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: askForm.task.trim(),
        mode: askForm.retrievalMode,
        document_ids: askForm.documentIds,
        collection_ids: askForm.collectionIds,
        keyword_weight: Number(askForm.keywordWeight || 0.5),
        semantic_weight: Number(askForm.semanticWeight || 0.5),
      }),
    });
    retrievalState.results = payload.retrieved_chunks || [];
    retrievalState.warnings = payload.warnings || [];
  } catch (error) {
    retrievalState.error = error.message;
    ElMessage.error(error.message);
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
              <h2>召回结果</h2>
              <span>{{ retrievalState.error || resultCountText }}</span>
            </div>
            <el-button type="primary" :icon="Search" :loading="retrievalState.loading" @click="runRetrieval">
              开始召回
            </el-button>
          </div>
          <div class="answer-scroll">
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
              <div v-for="chunk in retrievalState.results" :key="chunk.document_chunk_id" class="citation-card">
                <div class="citation-meta">
                  <span>[{{ chunk.citation_no }}]</span>
                  <span>{{ chunk.document_name }}</span>
                  <span>score {{ Number(chunk.score || 0).toFixed(4) }}</span>
                  <span>pages {{ (chunk.page_nos || []).join(", ") }}</span>
                  <span>locator {{ chunk.source_locator || "-" }}</span>
                </div>
                <div class="citation-links">
                  <span>retrieval ids: {{ (chunk.retrieval_chunk_ids || []).join(", ") || "-" }}</span>
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

import { computed, inject, reactive } from "vue";
import { ElMessage } from "element-plus";

import { buildDbParams, requestJson } from "../api.js";

export function useRetrievalScope() {
  const appState = inject("appState");

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
    retrievalMode: "hybrid",
    keywordWeight: 0.5,
    semanticWeight: 0.5,
  });

  const selectedDocuments = computed(() => {
    const selected = new Set(askForm.documentIds);
    return askOptions.documents.filter((item) => selected.has(item.id));
  });

  const selectedCollections = computed(() => {
    const selected = new Set(askForm.collectionIds);
    return askOptions.collections.filter((item) => selected.has(item.id));
  });

  async function refreshAskDocuments(query = askOptions.documentQuery) {
    askOptions.loadingDocuments = true;
    askOptions.documentQuery = query || "";
    try {
      const params = buildDbParams(appState.dbPath, { page: 1, page_size: 100 });
      if (askOptions.documentQuery.trim()) {
        params.set("q", askOptions.documentQuery.trim());
      }
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

  return {
    appState,
    askOptions,
    askForm,
    selectedDocuments,
    selectedCollections,
    refreshAskDocuments,
    refreshCollections,
  };
}

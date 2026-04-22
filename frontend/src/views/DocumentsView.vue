<script setup>
import { Delete, Edit, Refresh, Search, Upload, View } from "@element-plus/icons-vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { inject, onMounted, reactive, watch } from "vue";
import {
  buildDbParams,
  documentStatusType,
  formatFileSize,
  formatTime,
  requestJson,
  uploadWithProgress,
  withQuery,
} from "../api.js";

const appState = inject("appState");

const documentList = reactive({
  items: [],
  query: "",
  page: 1,
  pageSize: 20,
  total: 0,
  loading: false,
});

const uploadState = reactive({ uploading: false, progress: 0 });

const detailState = reactive({
  visible: false,
  loading: false,
  rebuildLoading: false,
  document: null,
  pageSummary: null,
  metadataDraft: "{}",
  pages: [],
  pagesPage: 1,
  pagesPageSize: 8,
  pagesTotal: 0,
});

const assignState = reactive({
  visible: false,
  loading: false,
  saving: false,
  document: null,
  collections: [],
  selectedIds: [],
});

onMounted(async () => {
  await refreshDocuments();
});

watch(
  () => appState.refreshTick,
  () => refreshDocuments(1),
);

async function refreshDocuments(page = documentList.page) {
  documentList.loading = true;
  documentList.page = page;
  try {
    const params = buildDbParams(appState.dbPath, {
      page,
      page_size: documentList.pageSize,
    });
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
    if (appState.dbPath.trim()) formData.append("db_path", appState.dbPath.trim());
    const payload = await uploadWithProgress("/api/documents", formData, (percent) => {
      uploadState.progress = percent;
      uploadRequest.onProgress({ percent });
    });
    uploadState.progress = 100;
    ElMessage.success(`${payload.document.original_filename} 上传完成`);
    await refreshDocuments(1);
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
    const params = buildDbParams(appState.dbPath);
    const payload = await requestJson(withQuery(`/api/documents/${encodeURIComponent(docId)}`, params));
    detailState.document = payload.document;
    detailState.pageSummary = payload.page_summary || null;
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
    const params = buildDbParams(appState.dbPath, {
      page,
      page_size: detailState.pagesPageSize,
    });
    const payload = await requestJson(`/api/documents/${encodeURIComponent(docId)}/pages?${params.toString()}`);
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
    const params = buildDbParams(appState.dbPath);
    await requestJson(withQuery(`/api/documents/${encodeURIComponent(detailState.document.id)}`, params), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metadata }),
    });
    ElMessage.success("Metadata 已保存");
    await Promise.all([loadDocumentDetail(), refreshDocuments(documentList.page)]);
  } catch (error) {
    ElMessage.error(error.message);
  }
}

async function rebuildDocumentPages(row = detailState.document) {
  if (!row?.id) return;
  detailState.rebuildLoading = true;
  try {
    const params = buildDbParams(appState.dbPath);
    await requestJson(withQuery(`/api/documents/${encodeURIComponent(row.id)}/parse`, params), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "full", force: true }),
    });
    ElMessage.success("文档分页已重建");
    await Promise.all([
      refreshDocuments(documentList.page),
      detailState.visible ? loadDocumentDetail(row.id) : Promise.resolve(),
      detailState.visible ? loadDocumentPages(row.id, 1) : Promise.resolve(),
    ]);
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    detailState.rebuildLoading = false;
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
    const params = buildDbParams(appState.dbPath);
    await requestJson(withQuery(`/api/documents/${encodeURIComponent(row.id)}`, params), { method: "DELETE" });
    if (detailState.document?.id === row.id) detailState.visible = false;
    ElMessage.success("文档已删除");
    await refreshDocuments(documentList.page);
  } catch (error) {
    ElMessage.error(error.message);
  }
}

async function openAssignDialog(row) {
  assignState.visible = true;
  assignState.loading = true;
  assignState.document = row;
  assignState.selectedIds = [];
  try {
    const [collections, documentCollections] = await Promise.all([
      requestJson("/api/collections"),
      requestJson(`/api/documents/${encodeURIComponent(row.id)}/collections`),
    ]);
    assignState.collections = collections.items || [];
    assignState.selectedIds = (documentCollections.items || []).map((item) => item.id);
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    assignState.loading = false;
  }
}

async function saveDocumentCollections() {
  if (!assignState.document?.id) return;
  assignState.saving = true;
  try {
    await requestJson(`/api/documents/${encodeURIComponent(assignState.document.id)}/collections`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collection_ids: assignState.selectedIds }),
    });
    ElMessage.success("Collection 分配已保存");
    assignState.visible = false;
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    assignState.saving = false;
  }
}
</script>

<template>
  <section class="page documents-page">
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
          <el-button type="primary" :loading="uploadState.uploading" :icon="Upload">上传并生成分页</el-button>
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
        <el-table-column label="操作" width="330" fixed="right">
          <template #default="{ row }">
            <el-button size="small" :icon="View" @click="openDocumentDetail(row)">查看</el-button>
            <el-button size="small" :icon="Edit" @click="openAssignDialog(row)">分配</el-button>
            <el-button
              size="small"
              :loading="detailState.rebuildLoading && detailState.document?.id === row.id"
              @click="rebuildDocumentPages(row)"
            >
              重建
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

  <el-drawer v-model="detailState.visible" size="62%" title="文档详情" class="document-drawer">
    <div v-loading="detailState.loading" class="drawer-content">
      <template v-if="detailState.document">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="文件名">{{ detailState.document.original_filename }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="documentStatusType(detailState.document.status)">{{ detailState.document.status }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="页数">
            {{ detailState.pageSummary?.page_count || detailState.document.page_count || 0 }}
          </el-descriptions-item>
          <el-descriptions-item label="大小">{{ formatFileSize(detailState.document.file_size) }}</el-descriptions-item>
          <el-descriptions-item label="Storage URI" :span="2">{{ detailState.document.storage_uri }}</el-descriptions-item>
        </el-descriptions>

        <div class="drawer-actions">
          <el-button type="primary" :loading="detailState.rebuildLoading" @click="rebuildDocumentPages(detailState.document)">
            重建分页
          </el-button>
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

  <el-dialog v-model="assignState.visible" title="分配 Collection" width="620px">
    <div v-loading="assignState.loading">
      <p class="dialog-subtitle">{{ assignState.document?.original_filename }}</p>
      <el-select v-model="assignState.selectedIds" multiple filterable clearable class="wide-control" placeholder="选择 Collection">
        <el-option v-for="item in assignState.collections" :key="item.id" :label="item.name" :value="item.id" />
      </el-select>
    </div>
    <template #footer>
      <el-button @click="assignState.visible = false">取消</el-button>
      <el-button type="primary" :loading="assignState.saving" @click="saveDocumentCollections">保存</el-button>
    </template>
  </el-dialog>
</template>

<script setup>
import { Delete, Edit, FolderOpened, Plus, Refresh, Search } from "@element-plus/icons-vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { inject, onMounted, reactive, watch } from "vue";
import { buildDbParams, documentStatusType, formatFileSize, requestJson } from "../api.js";

const appState = inject("appState");

const collectionState = reactive({
  items: [],
  loading: false,
  creating: false,
  newName: "",
});

const editState = reactive({
  visible: false,
  saving: false,
  collection: null,
  name: "",
});

const memberState = reactive({
  visible: false,
  loading: false,
  saving: false,
  collection: null,
  documents: [],
  selectedIds: [],
  documentQuery: "",
});

onMounted(async () => {
  await refreshCollections();
});

watch(
  () => appState.refreshTick,
  () => refreshCollections(),
);

async function refreshCollections() {
  collectionState.loading = true;
  try {
    const payload = await requestJson("/api/collections");
    collectionState.items = payload.items || [];
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    collectionState.loading = false;
  }
}

async function createCollection() {
  const name = collectionState.newName.trim();
  if (!name) {
    ElMessage.warning("请输入 Collection 名称");
    return;
  }
  collectionState.creating = true;
  try {
    await requestJson("/api/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    collectionState.newName = "";
    ElMessage.success("Collection 已创建");
    await refreshCollections();
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    collectionState.creating = false;
  }
}

function openEditDialog(row) {
  editState.collection = row;
  editState.name = row.name;
  editState.visible = true;
}

async function saveCollectionName() {
  if (!editState.collection?.id) return;
  const name = editState.name.trim();
  if (!name) {
    ElMessage.warning("请输入 Collection 名称");
    return;
  }
  editState.saving = true;
  try {
    await requestJson(`/api/collections/${encodeURIComponent(editState.collection.id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    ElMessage.success("Collection 已重命名");
    editState.visible = false;
    await refreshCollections();
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    editState.saving = false;
  }
}

async function deleteCollection(row) {
  try {
    await ElMessageBox.confirm(`确定删除 Collection「${row.name}」吗？文档不会被删除。`, "删除 Collection", {
      confirmButtonText: "删除",
      cancelButtonText: "取消",
      type: "warning",
    });
  } catch {
    return;
  }
  try {
    await requestJson(`/api/collections/${encodeURIComponent(row.id)}`, { method: "DELETE" });
    ElMessage.success("Collection 已删除");
    await refreshCollections();
  } catch (error) {
    ElMessage.error(error.message);
  }
}

async function openMembers(row) {
  memberState.visible = true;
  memberState.loading = true;
  memberState.collection = row;
  memberState.selectedIds = [];
  try {
    const [documents, members] = await Promise.all([
      loadDocumentsForTransfer(),
      requestJson(`/api/collections/${encodeURIComponent(row.id)}/documents`),
    ]);
    memberState.documents = documents;
    memberState.selectedIds = (members.items || []).map((item) => item.id);
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    memberState.loading = false;
  }
}

async function loadDocumentsForTransfer() {
  const params = buildDbParams(appState.dbPath, { page: 1, page_size: 500 });
  if (memberState.documentQuery.trim()) params.set("q", memberState.documentQuery.trim());
  const payload = await requestJson(`/api/documents?${params.toString()}`);
  return payload.items || [];
}

async function searchTransferDocuments() {
  memberState.loading = true;
  try {
    memberState.documents = await loadDocumentsForTransfer();
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    memberState.loading = false;
  }
}

async function saveMembers() {
  if (!memberState.collection?.id) return;
  memberState.saving = true;
  try {
    const payload = await requestJson(`/api/collections/${encodeURIComponent(memberState.collection.id)}/documents`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_ids: memberState.selectedIds }),
    });
    memberState.selectedIds = (payload.items || []).map((item) => item.id);
    ElMessage.success("Collection 文档已保存");
    await refreshCollections();
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    memberState.saving = false;
  }
}

function transferData() {
  return memberState.documents.map((item) => ({
    key: item.id,
    label: item.original_filename,
    disabled: false,
    status: item.status,
    pages: item.page_count,
    size: item.file_size,
  }));
}
</script>

<template>
  <section class="page collections-page">
    <el-card shadow="never" class="toolbar-card collection-toolbar">
      <div class="toolbar-left">
        <el-input
          v-model="collectionState.newName"
          clearable
          placeholder="新建 Collection 名称"
          @keyup.enter="createCollection"
        />
        <el-button type="primary" :icon="Plus" :loading="collectionState.creating" @click="createCollection">
          新建
        </el-button>
      </div>
      <div class="toolbar-right">
        <el-button :icon="Refresh" @click="refreshCollections">刷新</el-button>
      </div>
    </el-card>

    <el-card shadow="never" class="table-card">
      <el-table
        v-loading="collectionState.loading"
        :data="collectionState.items"
        height="100%"
        border
        stripe
        row-key="id"
        empty-text="暂无 Collection"
      >
        <el-table-column prop="name" label="名称" min-width="260" show-overflow-tooltip />
        <el-table-column prop="document_count" label="文档数" width="120" align="right">
          <template #default="{ row }">{{ row.document_count || 0 }}</template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="230" show-overflow-tooltip />
        <el-table-column prop="updated_at" label="更新时间" width="230" show-overflow-tooltip />
        <el-table-column label="操作" width="310" fixed="right">
          <template #default="{ row }">
            <el-button size="small" :icon="FolderOpened" @click="openMembers(row)">文档</el-button>
            <el-button size="small" :icon="Edit" @click="openEditDialog(row)">重命名</el-button>
            <el-button size="small" type="danger" :icon="Delete" @click="deleteCollection(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </section>

  <el-dialog v-model="editState.visible" title="重命名 Collection" width="520px">
    <el-input v-model="editState.name" clearable placeholder="Collection 名称" @keyup.enter="saveCollectionName" />
    <template #footer>
      <el-button @click="editState.visible = false">取消</el-button>
      <el-button type="primary" :loading="editState.saving" @click="saveCollectionName">保存</el-button>
    </template>
  </el-dialog>

  <el-drawer v-model="memberState.visible" size="720px" title="Collection 文档">
    <div v-loading="memberState.loading" class="drawer-content collection-members">
      <div class="member-header">
        <div>
          <h2>{{ memberState.collection?.name }}</h2>
          <span>{{ memberState.selectedIds.length }} 个已选文档</span>
        </div>
        <el-button type="primary" :loading="memberState.saving" @click="saveMembers">保存成员</el-button>
      </div>

      <div class="member-search">
        <el-input
          v-model="memberState.documentQuery"
          clearable
          placeholder="搜索可分配文档"
          :prefix-icon="Search"
          @keyup.enter="searchTransferDocuments"
          @clear="searchTransferDocuments"
        />
        <el-button :icon="Search" @click="searchTransferDocuments">搜索</el-button>
      </div>

      <el-transfer
        v-model="memberState.selectedIds"
        filterable
        :data="transferData()"
        :titles="['全部文档', 'Collection 文档']"
        target-order="push"
        class="document-transfer"
      >
        <template #default="{ option }">
          <div class="transfer-row">
            <span>{{ option.label }}</span>
            <span>
              <el-tag size="small" :type="documentStatusType(option.status)">{{ option.status }}</el-tag>
              {{ option.pages || 0 }} 页 / {{ formatFileSize(option.size) }}
            </span>
          </div>
        </template>
      </el-transfer>
    </div>
  </el-drawer>
</template>

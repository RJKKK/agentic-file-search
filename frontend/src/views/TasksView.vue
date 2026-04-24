<script setup>
import { Delete, Refresh, View } from "@element-plus/icons-vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { inject, onBeforeUnmount, onMounted, reactive, watch } from "vue";
import {
  buildDbParams,
  documentStatusType,
  formatDuration,
  formatTime,
  requestJson,
} from "../api.js";

const appState = inject("appState");
let pollTimer = null;

const taskList = reactive({
  items: [],
  page: 1,
  pageSize: 20,
  total: 0,
  loading: false,
  filters: {
    status: "",
    taskType: "",
    documentId: "",
  },
});

const detailState = reactive({
  visible: false,
  loading: false,
  task: null,
});

onMounted(async () => {
  await refreshTasks();
  startPolling();
});

watch(
  () => appState.refreshTick,
  () => refreshTasks(1),
);

onBeforeUnmount(() => {
  stopPolling();
});

function startPolling() {
  stopPolling();
  pollTimer = setInterval(async () => {
    await refreshTasks(taskList.page);
    if (detailState.visible && detailState.task?.id && !isTerminal(detailState.task.status)) {
      await loadTaskDetail(detailState.task.id);
    }
  }, 3000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function isTerminal(status) {
  return ["completed", "failed"].includes(status);
}

async function refreshTasks(page = taskList.page) {
  taskList.loading = true;
  taskList.page = page;
  try {
    const params = buildDbParams(appState.dbPath, {
      page,
      page_size: taskList.pageSize,
      status: taskList.filters.status || null,
      task_type: taskList.filters.taskType || null,
      document_id: taskList.filters.documentId || null,
    });
    const payload = await requestJson(`/api/document-parse-tasks?${params.toString()}`);
    taskList.items = payload.items || [];
    taskList.total = payload.total || 0;
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    taskList.loading = false;
  }
}

function handlePageSizeChange(size) {
  taskList.pageSize = size;
  refreshTasks(1);
}

async function loadTaskDetail(taskId) {
  detailState.loading = true;
  try {
    const params = buildDbParams(appState.dbPath);
    const payload = await requestJson(`/api/document-parse-tasks/${encodeURIComponent(taskId)}?${params.toString()}`);
    detailState.task = payload.task;
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    detailState.loading = false;
  }
}

async function openTaskDetail(row) {
  detailState.visible = true;
  await loadTaskDetail(row.id);
}

async function deleteTask(row) {
  try {
    await ElMessageBox.confirm(`确定删除任务 ${row.id} 吗？`, "删除任务", {
      confirmButtonText: "删除",
      cancelButtonText: "取消",
      type: "warning",
    });
  } catch {
    return;
  }
  try {
    const params = buildDbParams(appState.dbPath);
    await requestJson(`/api/document-parse-tasks/${encodeURIComponent(row.id)}?${params.toString()}`, {
      method: "DELETE",
    });
    ElMessage.success("任务已删除");
    if (detailState.task?.id === row.id) {
      detailState.visible = false;
      detailState.task = null;
    }
    await refreshTasks(taskList.page);
  } catch (error) {
    ElMessage.error(error.message);
  }
}
</script>

<template>
  <section class="page documents-page">
    <el-card shadow="never" class="toolbar-card document-toolbar">
      <div class="toolbar-left">
        <el-select v-model="taskList.filters.status" clearable placeholder="状态" @change="refreshTasks(1)">
          <el-option label="queued" value="queued" />
          <el-option label="running" value="running" />
          <el-option label="completed" value="completed" />
          <el-option label="failed" value="failed" />
        </el-select>
        <el-select v-model="taskList.filters.taskType" clearable placeholder="任务类型" @change="refreshTasks(1)">
          <el-option label="upload_parse" value="upload_parse" />
          <el-option label="reparse" value="reparse" />
          <el-option label="embed_only" value="embed_only" />
        </el-select>
        <el-input
          v-model="taskList.filters.documentId"
          clearable
          placeholder="按文档 ID 过滤"
          @keyup.enter="refreshTasks(1)"
          @clear="refreshTasks(1)"
        />
      </div>
      <div class="toolbar-right">
        <el-button :icon="Refresh" @click="refreshTasks(taskList.page)">刷新</el-button>
      </div>
    </el-card>

    <el-card shadow="never" class="table-card">
      <el-table
        v-loading="taskList.loading"
        :data="taskList.items"
        height="100%"
        border
        stripe
        row-key="id"
        empty-text="暂无任务"
      >
        <el-table-column prop="document_filename" label="文档" min-width="240" show-overflow-tooltip />
        <el-table-column prop="task_type" label="任务类型" width="140" />
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="documentStatusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="current_stage" label="当前阶段" width="180">
          <template #default="{ row }">{{ row.current_stage || "-" }}</template>
        </el-table-column>
        <el-table-column prop="progress_percent" label="进度" width="160">
          <template #default="{ row }">
            <el-progress :percentage="Number(row.progress_percent || 0)" :stroke-width="8" />
          </template>
        </el-table-column>
        <el-table-column prop="total_duration_ms" label="总耗时" width="120">
          <template #default="{ row }">{{ formatDuration(row.total_duration_ms) }}</template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="190">
          <template #default="{ row }">{{ row.created_at ? new Date(row.created_at).toLocaleString() : "-" }}</template>
        </el-table-column>
        <el-table-column label="操作" width="160" fixed="right">
          <template #default="{ row }">
            <el-button size="small" :icon="View" @click="openTaskDetail(row)">详情</el-button>
            <el-button
              v-if="isTerminal(row.status)"
              size="small"
              type="danger"
              :icon="Delete"
              @click="deleteTask(row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <div class="pagination-bar">
      <el-pagination
        v-model:current-page="taskList.page"
        v-model:page-size="taskList.pageSize"
        :page-sizes="[10, 20, 50, 100]"
        :total="taskList.total"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="handlePageSizeChange"
        @current-change="refreshTasks"
      />
    </div>
  </section>

  <el-dialog v-model="detailState.visible" title="任务详情" width="760px">
    <div v-loading="detailState.loading" v-if="detailState.task" class="task-detail">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="任务 ID">{{ detailState.task.id }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="documentStatusType(detailState.task.status)">{{ detailState.task.status }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="文档">{{ detailState.task.document_filename }}</el-descriptions-item>
        <el-descriptions-item label="任务类型">{{ detailState.task.task_type }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ detailState.task.created_at ? new Date(detailState.task.created_at).toLocaleString() : "-" }}</el-descriptions-item>
        <el-descriptions-item label="总耗时">{{ formatDuration(detailState.task.total_duration_ms) }}</el-descriptions-item>
      </el-descriptions>

      <el-alert
        v-if="detailState.task.error_message"
        type="error"
        show-icon
        :closable="false"
        :title="detailState.task.error_message"
        class="task-error"
      />

      <h3>任务选项</h3>
      <pre class="page-markdown">{{ JSON.stringify(detailState.task.options || {}, null, 2) }}</pre>

      <h3>阶段详情</h3>
      <el-table :data="detailState.task.stage_timings || []" border stripe>
        <el-table-column prop="label" label="阶段" min-width="180" />
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="documentStatusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="开始时间" width="190">
          <template #default="{ row }">{{ row.started_at ? new Date(row.started_at).toLocaleString() : "-" }}</template>
        </el-table-column>
        <el-table-column label="结束时间" width="190">
          <template #default="{ row }">{{ row.finished_at ? new Date(row.finished_at).toLocaleString() : "-" }}</template>
        </el-table-column>
        <el-table-column label="耗时" width="120">
          <template #default="{ row }">{{ formatDuration(row.duration_ms) }}</template>
        </el-table-column>
      </el-table>
    </div>
  </el-dialog>
</template>

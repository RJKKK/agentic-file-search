<script setup>
import { ChatDotRound, Collection, Document, List, Refresh, Search, Setting } from "@element-plus/icons-vue";
import { provide, reactive } from "vue";
import { useRoute, useRouter } from "vue-router";

const route = useRoute();
const router = useRouter();
const appState = reactive({
  dbPath: "",
  refreshTick: 0,
});

provide("appState", appState);

function selectRoute(path) {
  router.push(path);
}

function refreshData() {
  appState.refreshTick += 1;
}
</script>

<template>
  <el-container class="app-shell">
    <el-header class="app-header">
      <div class="brand">
        <el-icon><Document /></el-icon>
        <span>FsExplorer</span>
      </div>

      <el-menu :default-active="route.path" mode="horizontal" class="top-menu" @select="selectRoute">
        <el-menu-item index="/qa">
          <el-icon><ChatDotRound /></el-icon>
          <span>问答</span>
        </el-menu-item>
        <el-menu-item index="/retrieval-test">
          <el-icon><Search /></el-icon>
          <span>召回测试</span>
        </el-menu-item>
        <el-menu-item index="/documents">
          <el-icon><Document /></el-icon>
          <span>文档</span>
        </el-menu-item>
        <el-menu-item index="/tasks">
          <el-icon><List /></el-icon>
          <span>解析任务</span>
        </el-menu-item>
        <el-menu-item index="/collections">
          <el-icon><Collection /></el-icon>
          <span>Collections</span>
        </el-menu-item>
      </el-menu>

      <el-popover placement="bottom-end" width="420" trigger="click">
        <template #reference>
          <el-button :icon="Setting">设置</el-button>
        </template>
        <el-form label-position="top">
          <el-form-item label="可选 DB Path">
            <el-input v-model="appState.dbPath" clearable placeholder="留空使用默认数据库" />
          </el-form-item>
          <el-button type="primary" :icon="Refresh" @click="refreshData">刷新数据</el-button>
        </el-form>
      </el-popover>
    </el-header>

    <el-main class="app-main">
      <router-view />
    </el-main>
  </el-container>
</template>

import { createRouter, createWebHistory } from "vue-router";
const QaView = ()=>import("./views/QaView.vue");
const RetrievalTestView = ()=>import("./views/RetrievalTestView.vue");
const DocumentsView = ()=>import("./views/DocumentsView.vue");
const CollectionsView = ()=>import("./views/CollectionsView.vue");
const TasksView = ()=>import("./views/TasksView.vue");

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/qa" },
    { path: "/qa", component: QaView },
    { path: "/retrieval-test", component: RetrievalTestView },
    { path: "/documents", component: DocumentsView },
    { path: "/tasks", component: TasksView },
    { path: "/collections", component: CollectionsView },
  ],
});

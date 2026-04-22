import { createRouter, createWebHistory } from "vue-router";
const QaView = ()=>import("./views/QaView.vue");
const DocumentsView = ()=>import("./views/DocumentsView.vue");
const CollectionsView = ()=>import("./views/CollectionsView.vue");

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/qa" },
    { path: "/qa", component: QaView },
    { path: "/documents", component: DocumentsView },
    { path: "/collections", component: CollectionsView },
  ],
});

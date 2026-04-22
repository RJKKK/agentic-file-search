import { createApp } from "vue";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import "github-markdown-css/github-markdown-light.css";
import App from "./App.vue";
import { router } from "./router.js";
import "./styles.css";

createApp(App).use(router).use(ElementPlus).mount("#app");

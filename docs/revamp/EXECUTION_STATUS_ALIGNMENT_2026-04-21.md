# 执行状态对齐（2026-04-21）

## 1. 对齐范围
- 分支：`feat/api-ui-sqlite-openai`
- 最近提交：`dcf00a7 (save)`、`bf079d5 (ws2sse)`
- 基线校验：
  - `python -m pytest -q` -> `123 passed, 2 skipped`
  - 变更相关 `ruff` 检查通过
  - `python -m ruff check .` 仍存在历史遗留问题（主要在 `scripts/` 与 `tests/test_server_streaming.py` 未使用导入）

## 2. 里程碑状态（M0-M5）

| 里程碑 | 状态 | 结论 |
|---|---|---|
| M0 环境与基线 | 已完成 | 系统环境安装与基线验证已落地，文档已在 `docs/revamp` |
| M1 存储迁移 | 已完成（有偏差） | 主链路已切 PostgreSQL；但为兼容测试引入了“非 DSN 本地内存存储路径”兜底 |
| M2 解析与缓存 | 已完成（主链路） | 页级缓存、图片 hash 占位、lazy 语义增强、运行时事件均已接入 |
| M3 API/SSE | 已完成（首版） | 文档 CRUD、上传、解析触发、分页读取、SSE 扩展字段均已可用 |
| M4 前端重写 | 未开始 | 仍是单文件 `ui.html`，未落 Vue3 + Vite |
| M5 回归与发布 | 未开始 | README/迁移说明仍与目标方案存在差异 |

## 3. 与“当前目标任务”对齐结果

### 3.1 已对齐项
1. PostgreSQL 主路径已落地（替代 DuckDB 主路径）
- 证据：`src/fs_explorer/storage/postgres.py`
- 证据：`src/fs_explorer/storage/__init__.py`

2. M2 关键能力已落地（解析缓存 + 图片语义增强）
- 证据：`src/fs_explorer/document_parsing.py`
- 证据：`src/fs_explorer/indexing/pipeline.py`
- 证据：`tests/test_m2_parsing.py`

3. 会话流已从 WS 迁移到 SSE
- 证据：`src/fs_explorer/server.py`（`/api/explore/sessions/{session_id}/events`）
- 证据：`src/fs_explorer/explore_sessions.py`
- 证据：`tests/test_server_streaming.py`

### 3.2 未对齐项（需继续）
1. 仍保留 CLI，对“仅 API + 前端”目标未完成
- 证据：`src/fs_explorer/main.py`
- 证据：`pyproject.toml` 仍有 `explore` script

2. OpenAI 兼容模型配置尚未替换完成
- 当前仍依赖 `GOOGLE_API_KEY`
- 证据：`.env.example`
- 证据：`src/fs_explorer/agent.py`
- 证据：`src/fs_explorer/embeddings.py`
- 证据：`src/fs_explorer/image_semantics.py`

3. 文档 CRUD / 上传托管 / 分页解析查询 API 已按首版原型落地
- 已实现：`GET /api/documents`
- 已实现：`POST /api/documents`
- 已实现：`GET /api/documents/{doc_id}`
- 已实现：`PATCH /api/documents/{doc_id}`
- 已实现：`DELETE /api/documents/{doc_id}`
- 已实现：`POST /api/documents/{doc_id}/parse`
- 已实现：`GET /api/documents/{doc_id}/pages`
- 证据：`src/fs_explorer/server.py`
- 证据：`tests/test_server_documents_api.py`

4. SSE 会话流继续保持兼容，并补齐了 M2/M3 所需扩展字段
- 已透出：`cache_hit`、`image_enhance_started`、`image_enhance_done`
- `complete` 事件已附带 `last_focus_anchor` 与 `context_budget`
- 证据：`src/fs_explorer/server.py`
- 证据：`src/fs_explorer/explore_sessions.py`

5. 前端未重写为 Vue3 + Vite，单端口静态托管仍是内嵌 HTML
- 证据：`src/fs_explorer/ui.html`

### 3.3 设计偏差（建议确认）
1. `PostgresStorage` 为兼容测试，保留了“无 DSN -> 本地内存存储”行为
- 证据：`src/fs_explorer/storage/postgres.py`
- 影响：与“全量 PostgreSQL 单一路径”目标不完全一致
- 建议：M3 或 M5 阶段决定是否移除该兼容层，仅保留测试替身

2. 协议已从 WebSocket 改为 SSE
- 这与当前提交方向一致，但需同步原型文档契约字段

3. 上传 API 采用“上传到现有 corpus 文件夹并立即增量建索引”的落地方式
- 证据：`src/fs_explorer/server.py`
- 影响：满足 M3 可用性目标，但暂未抽象出独立托管目录/租户隔离
- 建议：M4/M5 再决定是否引入专用上传存储根目录

## 4. 下一步建议（按优先级）
1. 配置迁移：落地 OpenAI 兼容 `TEXT_* / VISION_* / EMBEDDING_*` 配置，逐步移除 `GOOGLE_API_KEY` 强依赖。
2. M4 启动：引入 Vue3 + Vite，FastAPI 同源静态托管单端口联调。
3. CLI 下线：移除 `main.py` 对外入口与 `pyproject.toml` 中 CLI script。
4. 质量收口：修复遗留 `ruff` 问题，恢复“lint 全绿”门禁。

## 5. 本次对齐结论
- 代码进度已经从 M1 推进到 M3 首版闭环。
- 当前剩余关键闭环聚焦三块：`CLI 下线`、`OpenAI 兼容配置切换`、`Vue3 + Vite 前端重写`。
- 现阶段可判定项目状态为：**M0-M3 完成，M4/M5 未完成，整体约 75%-80%**。

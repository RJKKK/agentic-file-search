# 原型文档（v1）：PostgreSQL + pgvector 架构规格

## 1. 原型目标与非目标

## 1.1 目标

1. 提供统一的 API + WebSocket 服务，支持文档上传托管、解析、检索与问答。
2. 采用 PostgreSQL + pgvector 作为唯一主存储与向量检索引擎。
3. 提供页级解析缓存与图片语义增强缓存，提升重复查询效率。
4. 前端与后端同源单端口访问，降低部署复杂度。
5. 模型接入兼容 OpenAI 格式，支持文本与视觉模型分离配置。

## 1.2 非目标

1. v1 不实现多租户权限体系。
2. v1 不实现离线批处理调度系统。
3. v1 不实现跨地域多活部署。

## 2. 系统架构原型

## 2.1 组件

1. FastAPI：
   - REST API
   - WebSocket 会话流
   - 前端静态资源托管
2. Vue3 + Vite 前端：
   - 文档管理
   - 解析状态查看
   - 问答和步骤流展示
3. PostgreSQL + pgvector：
   - 文档与缓存数据
   - 分块检索数据
   - 向量相似度检索
4. 模型网关：
   - 文本模型（OpenAI 兼容）
   - 视觉模型（OpenAI 兼容）
   - 可选 embedding 模型（OpenAI 兼容）

## 2.2 部署形态

1. 开发环境与生产环境均对外暴露单端口（默认 `8000`）。
2. 前端通过 FastAPI 提供静态资源与同源 API/WS 访问。
3. PostgreSQL 可通过 Docker Compose 启动（推荐 pgvector 镜像）。

## 3. 数据模型原型（PostgreSQL）

以下为核心数据模型与约束，字段命名采用 snake_case。

## 3.1 `documents`

用途：文档主记录（上传托管与生命周期管理）。

关键字段：

- `id` UUID PK
- `filename` TEXT NOT NULL
- `storage_path` TEXT NOT NULL
- `mime_type` TEXT NOT NULL
- `size_bytes` BIGINT NOT NULL
- `content_sha256` TEXT NOT NULL
- `status` TEXT NOT NULL
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT now()
- `updated_at` TIMESTAMPTZ NOT NULL DEFAULT now()
- `deleted_at` TIMESTAMPTZ NULL

关键约束：

- `UNIQUE (content_sha256, deleted_at)`（逻辑删除后允许重复上传）

## 3.2 `parsed_units`

用途：按页/逻辑页解析缓存。

关键字段：

- `id` UUID PK
- `document_id` UUID NOT NULL REFERENCES documents(id)
- `page_no` INT NOT NULL
- `markdown` TEXT NOT NULL
- `page_hash` TEXT NOT NULL
- `parser_name` TEXT NOT NULL
- `parser_version` TEXT NOT NULL
- `has_images` BOOLEAN NOT NULL DEFAULT FALSE
- `updated_at` TIMESTAMPTZ NOT NULL DEFAULT now()

关键约束：

- `UNIQUE (document_id, page_no, parser_version)`

## 3.3 `image_semantics`

用途：图片语义增强缓存（按 hash 唯一）。

关键字段：

- `id` UUID PK
- `image_hash` TEXT NOT NULL UNIQUE
- `mime_type` TEXT NOT NULL
- `storage_path` TEXT NOT NULL
- `vision_text` TEXT NULL
- `vision_model` TEXT NULL
- `vision_status` TEXT NOT NULL DEFAULT 'pending'
- `updated_at` TIMESTAMPTZ NOT NULL DEFAULT now()

## 3.4 `chunks`（检索支撑）

用途：问答检索候选分块，支持关键词与向量混合检索。

关键字段：

- `id` UUID PK
- `document_id` UUID NOT NULL REFERENCES documents(id)
- `parsed_unit_id` UUID NULL REFERENCES parsed_units(id)
- `position` INT NOT NULL
- `text` TEXT NOT NULL
- `metadata_json` JSONB NOT NULL DEFAULT '{}'::jsonb
- `embedding` VECTOR(768) NULL
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT now()

索引建议：

- `GIN (to_tsvector('simple', text))` 用于关键词召回
- `HNSW/IVFFLAT` 向量索引（基于 pgvector 能力）
- `BTREE (document_id, position)`

## 3.5 DDL 原型片段

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  filename TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  size_bytes BIGINT NOT NULL,
  content_sha256 TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS parsed_units (
  id UUID PRIMARY KEY,
  document_id UUID NOT NULL REFERENCES documents(id),
  page_no INT NOT NULL,
  markdown TEXT NOT NULL,
  page_hash TEXT NOT NULL,
  parser_name TEXT NOT NULL,
  parser_version TEXT NOT NULL,
  has_images BOOLEAN NOT NULL DEFAULT FALSE,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_id, page_no, parser_version)
);

CREATE TABLE IF NOT EXISTS image_semantics (
  id UUID PRIMARY KEY,
  image_hash TEXT NOT NULL UNIQUE,
  mime_type TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  vision_text TEXT,
  vision_model TEXT,
  vision_status TEXT NOT NULL DEFAULT 'pending',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

## 4. API 原型契约（REST）

统一响应头建议返回 `X-Trace-Id` 便于追踪。

## 4.1 文档上传与 CRUD

### POST `/api/documents`

`multipart/form-data` 上传文件。

成功响应：

```json
{
  "id": "6f1d9cf8-6de8-47e8-b2c8-56b6f81ef321",
  "filename": "contract.pdf",
  "status": "uploaded"
}
```

### GET `/api/documents`

查询参数：`page`, `page_size`, `status`, `q`。

### GET `/api/documents/{id}`

返回文档详情与解析概览。

### PATCH `/api/documents/{id}`

允许更新业务标签、备注等元数据字段（不修改文件本体）。

### DELETE `/api/documents/{id}`

逻辑删除文档，并标记关联数据状态。

## 4.2 解析接口

### POST `/api/documents/{id}/parse`

请求体：

```json
{
  "mode": "incremental",
  "force": false
}
```

响应体：

```json
{
  "document_id": "6f1d9cf8-6de8-47e8-b2c8-56b6f81ef321",
  "parsed_pages": 12,
  "cache_hits": 20,
  "images_detected": 8
}
```

### GET `/api/documents/{id}/pages`

查询参数：`page_no`, `page_size`，返回 `parsed_units`。

## 4.3 问答接口

### POST `/api/chat`

请求体：

```json
{
  "question": "请总结该合同的付款条款",
  "document_ids": [
    "6f1d9cf8-6de8-47e8-b2c8-56b6f81ef321"
  ],
  "enable_semantic": true
}
```

响应体（同步模式）：

```json
{
  "answer": "......",
  "citations": [
    {
      "document_id": "6f1d9cf8-6de8-47e8-b2c8-56b6f81ef321",
      "page_no": 4
    }
  ]
}
```

## 5. WebSocket 事件协议原型

地址：`/ws/explore`

## 5.1 客户端发起

```json
{
  "task": "请总结合同中的付款和违约条款",
  "document_ids": ["6f1d9cf8-6de8-47e8-b2c8-56b6f81ef321"],
  "enable_semantic": true
}
```

## 5.2 服务端事件类型

固定事件：

1. `start`
2. `tool_call`
3. `go_deeper`
4. `ask_human`
5. `complete`
6. `error`

新增事件：

1. `cache_hit`
2. `image_enhance_started`
3. `image_enhance_done`

## 5.3 事件 payload schema

### `start`

```json
{
  "type": "start",
  "data": {
    "task": "string",
    "session_id": "string",
    "document_count": 1
  }
}
```

### `tool_call`

```json
{
  "type": "tool_call",
  "data": {
    "step": 3,
    "tool_name": "semantic_search",
    "tool_input": {},
    "reason": "string"
  }
}
```

### `cache_hit`

```json
{
  "type": "cache_hit",
  "data": {
    "document_id": "string",
    "page_no": 8,
    "source": "parsed_units"
  }
}
```

### `image_enhance_started`

```json
{
  "type": "image_enhance_started",
  "data": {
    "image_hash": "string",
    "model": "string"
  }
}
```

### `image_enhance_done`

```json
{
  "type": "image_enhance_done",
  "data": {
    "image_hash": "string",
    "vision_status": "done"
  }
}
```

### `complete`

```json
{
  "type": "complete",
  "data": {
    "final_result": "string",
    "citations": [],
    "stats": {
      "steps": 12,
      "cache_hits": 9
    }
  }
}
```

## 6. 配置原型

## 6.1 数据库配置

- `DB_HOST`（必填）
- `DB_PORT`（默认 `5432`）
- `DB_NAME`（必填）
- `DB_USER`（必填）
- `DB_PASSWORD`（必填）
- `DB_POOL_MIN`（默认 `2`）
- `DB_POOL_MAX`（默认 `20`）

## 6.2 模型配置（OpenAI 兼容）

文本模型：

- `TEXT_MODEL_NAME`（必填）
- `TEXT_API_KEY`（必填）
- `TEXT_BASE_URL`（必填）

视觉模型：

- `VISION_MODEL_NAME`（必填）
- `VISION_API_KEY`（必填）
- `VISION_BASE_URL`（必填）

可选 embedding：

- `EMBEDDING_MODEL_NAME`（可选）
- `EMBEDDING_API_KEY`（可选）
- `EMBEDDING_BASE_URL`（可选）
- `EMBEDDING_DIM`（默认 `768`）

## 6.3 系统安装与镜像源策略（无 venv）

推荐：

```bash
uv pip install --system -e ".[dev]" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

备用：

```bash
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 7. 非功能原型

## 7.1 单端口一致性

1. 开发与生产均由 FastAPI 对外暴露同一端口（默认 `8000`）。
2. 前端资源、REST、WS 全部同源。

## 7.2 幂等与并发

1. 解析接口支持幂等调用（重复请求不会重复写入相同 `parsed_units`）。
2. 图片增强按 `image_hash` 幂等。
3. 同文档并发解析使用文档级锁或事务约束防止重复计算。

## 7.3 错误码与可观测性

统一错误对象：

```json
{
  "error_code": "PARSER_DOC_CONVERT_FAILED",
  "message": "DOC conversion requires LibreOffice",
  "trace_id": "string",
  "details": {}
}
```

日志字段最低要求：

1. `trace_id`
2. `session_id`
3. `document_id`
4. `step`
5. `latency_ms`
6. `model_name`

## 8. 验收映射（接口可测性）

1. DB 约束可映射为迁移测试（唯一键、外键、索引存在性）。
2. API 字段可映射为契约测试（请求缺失、类型错误、状态流转）。
3. WS 事件可映射为时序测试（start -> ... -> complete，包含新增事件）。
4. 缓存与幂等可映射为重复执行测试（解析与图片增强均需命中）。


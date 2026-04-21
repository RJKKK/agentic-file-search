# Agentic File Search

面向文档问答与探索的本地应用。当前主产品流程已经从“本地目录问答”切换为“统一上传文档库 + collection + 临时多选文档问答”。

## 当前能力

- 文档统一上传到共享文档库
- 上传文件先写入对象存储抽象层
  - 首版使用本地目录模拟对象存储
- 问答范围由两部分组成
  - 一个可选 collection
  - 若干临时多选文档
- 搜索、问答、解析、懒索引都只作用于本次选中的文档集合
- 保留单文档详情、分页解析、metadata 编辑、SSE 会话流

## 技术栈

- 后端：FastAPI
- 前端：Vue 3 + Vite
- 索引存储：当前项目内置 `PostgresStorage` 抽象
  - 本地测试场景可直接使用文件路径命名空间
  - 生产可继续接 PostgreSQL / pgvector
- 对象存储：`LocalBlobStore`

## 环境要求

- Python 3.10+
- Node.js 20+

## 安装

```bash
git clone https://github.com/PromtEngineer/agentic-file-search.git
cd agentic-file-search

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -e ".[dev]"
```

首次构建前端：

```bash
cd frontend
npm install
npm run build
cd ..
```

## 配置

在项目根目录创建 `.env`：

```bash
TEXT_MODEL_NAME=gpt-4o-mini
TEXT_API_KEY=your_api_key_here

# 可选：向量检索 / 图片语义增强
# EMBEDDING_MODEL_NAME=text-embedding-3-small
# EMBEDDING_API_KEY=your_embedding_api_key_here
# VISION_MODEL_NAME=gpt-4o-mini
# VISION_API_KEY=your_vision_api_key_here

# 可选：索引数据库
FS_EXPLORER_DB_DSN=postgresql://fs_explorer:devpassword@127.0.0.1:5432/fs_explorer

# 可选：本地对象存储目录
FS_EXPLORER_OBJECT_STORE_DIR=data/object_store
```

说明：

- `FS_EXPLORER_OBJECT_STORE_DIR` 默认为 `data/object_store`
- 上传后的原始文件会写入：
  - `documents/{doc_id}/{sanitized_filename}`

## 启动方式

### 推荐：单端口启动前后端

先构建前端：

```bash
cd frontend
npm run build
cd ..
```

再启动 FastAPI：

```bash
python -c "from fs_explorer.server import run_server; run_server()"
```

默认访问：

- Web UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API: `http://127.0.0.1:8000/api/...`
- SSE: `http://127.0.0.1:8000/api/explore/sessions/{id}/events`

当前开发环境下，前端和后端走同一个端口 `8000`。FastAPI 会优先托管 `frontend/dist/`，因此浏览器只需要访问一个地址。

### 显式指定地址和端口

```bash
python -c "from fs_explorer.server import run_server; run_server(host='127.0.0.1', port=8000)"
```

### 可选：直接使用 `uvicorn`

如果你的环境已经能直接调用 `uvicorn`，可以这样启动：

```bash
uvicorn --app-dir src fs_explorer.server:app --host 127.0.0.1 --port 8000
```

README 不再使用 `python -m uvicorn`。

## 当前 UI 工作流

启动后页面分为三栏：

- 左侧：文档库列表、搜索、上传
- 中间：单文档详情、metadata、分页解析
- 右侧：collection 管理、临时多选文档、问答面板、SSE 时间线

问答请求体语义：

- `document_ids`
- `collection_id`

实际作用域为：

- `document_ids ∪ collection.documents`

如果两者都不传，会返回 `400`。

## 常用开发命令

运行后端测试：

```bash
python -m pytest
```

仅构建前端：

```bash
npm --prefix frontend run build
```

## Database Migration

如果 `fs_explorer` 数据库是空库，先执行一次显式迁移命令：

```bash
python -m fs_explorer migrate
```

如果你想指定 DSN：

```bash
python -m fs_explorer migrate --db-path "postgresql://postgres:root@127.0.0.1:5432/fs_explorer"
```

如果你的环境已经安装了 console script，也可以使用 `explore migrate`。

这个命令会调用项目内置的 schema bootstrap，按幂等方式执行 `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE`，适合空库初始化和升级现有库结构。

## 目录

```text
frontend/
src/fs_explorer/
  agent.py
  blob_store.py
  document_cache.py
  document_library.py
  document_parsing.py
  server.py
  workflow.py
  indexing/
  search/
  storage/
tests/
docs/revamp/
```

## 参考

- [docs/revamp/IMPLEMENTATION_TASK_BREAKDOWN.md](docs/revamp/IMPLEMENTATION_TASK_BREAKDOWN.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## License

MIT

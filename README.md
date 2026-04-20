# Agentic File Search

一个面向文档目录问答的 Agent 检索系统。它会先解析本地文档，再把结构化内容、文本分块和检索索引写入 PostgreSQL，随后通过 Agent 工作流完成检索、追踪引用和回答生成。

当前主执行路径已经以 `PostgreSQL + pgvector` 为中心，安装和开发流程统一使用系统 Python 直接安装，不使用虚拟环境，也不使用 `uv`。

## 功能概览

- 支持目录索引、关键词检索、语义检索和基于 Agent 的问答
- 支持 `PDF / DOCX / DOC / PPTX / XLSX / HTML / Markdown`
- PDF 解析按页落库，支持页级缓存和图片语义占位缓存
- FastAPI 提供 Web UI 和 API
- 保留 CLI 入口，便于本地调试和回归验证

## 环境要求

- Python 3.10+
- PostgreSQL 15+，并安装 `pgvector`
- 可选：LibreOffice
  - 仅在需要解析 `.doc` 文件时使用，命令需要能找到 `soffice`

## 直接安装

以下命令默认直接安装到当前系统 Python 环境。

```bash
git clone https://github.com/PromtEngineer/agentic-file-search.git
cd agentic-file-search

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

如果你在中国大陆网络环境下，建议直接使用清华镜像：

```bash
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

## PostgreSQL 准备

项目运行前需要准备数据库，并确保目标库可创建 `vector` 扩展。

```sql
CREATE DATABASE fs_explorer;
\c fs_explorer
CREATE EXTENSION IF NOT EXISTS vector;
```

默认连接串：

```text
postgresql://fs_explorer:devpassword@127.0.0.1:5432/fs_explorer
```

也可以通过环境变量覆盖：

```bash
FS_EXPLORER_DB_DSN=postgresql://user:password@127.0.0.1:5432/fs_explorer
```

仓库里也提供了 `docker/docker-compose.yml` 作为本地 PostgreSQL/pgvector 的启动参考。

## 配置

在项目根目录创建 `.env`：

```bash
GOOGLE_API_KEY=your_api_key_here
FS_EXPLORER_DB_DSN=postgresql://fs_explorer:devpassword@127.0.0.1:5432/fs_explorer
```

如果需要语义检索或后续视觉增强能力，建议同时准备兼容的模型配置。

## 启动方式

### 1. Web UI

```bash
python -m uvicorn fs_explorer.server:app --host 127.0.0.1 --port 8000
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

### 2. CLI

```bash
python -m fs_explorer.main --help
```

示例：

```bash
python -m fs_explorer.main index data/test_acquisition --discover-schema
python -m fs_explorer.main query --task "Look in data/test_acquisition/. What is the adjusted purchase price?"
```

如果你的环境已经安装了 entry point，也可以直接使用：

```bash
explore --task "Look in data/test_acquisition/. Who is the CTO?"
explore-ui
```

## 开发检查

```bash
python -m pytest
python -m ruff check .
```

## M2 相关实现说明

当前仓库已经开始落地解析链路升级，重点包括：

- PDF 按页解析并清理重复页眉页脚
- `parsed_units` 表缓存页级 Markdown
- 基于内容 hash 的重复索引复用
- `image_semantics` 表缓存图片基础信息，语义字段支持后续懒增强
- 检索命中相关页时可触发图片语义增强，`get_document` 会展示已缓存语义

这部分仍会继续向 API 和前端状态可视化延伸，详细拆分见 [docs/revamp/IMPLEMENTATION_TASK_BREAKDOWN.md](docs/revamp/IMPLEMENTATION_TASK_BREAKDOWN.md)。

## 目录结构

```text
src/fs_explorer/
  agent.py
  document_parsing.py
  fs.py
  server.py
  workflow.py
  indexing/
  search/
  storage/
tests/
docs/revamp/
docker/
```

## 参考文档

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- [docs/revamp/IMPLEMENTATION_TASK_BREAKDOWN.md](docs/revamp/IMPLEMENTATION_TASK_BREAKDOWN.md)

## License

MIT

# M1 进展记录（PostgreSQL 主路径落地）

## 时间与范围
- 更新时间：2026-04-20（Asia/Shanghai）
- 目标里程碑：M1 存储迁移（DuckDB -> PostgreSQL）
- 本次范围：存储实现、配置切换、本机数据库可用性验证、关键测试回归

## 已完成项
1. 存储后端完成 PostgreSQL 化
- 新增 [src/fs_explorer/storage/postgres.py](/C:/projects/agentic-file-search/src/fs_explorer/storage/postgres.py)
- 覆盖 `corpora/documents/chunks/schemas/chunk_embeddings` 的初始化与 CRUD/检索能力
- 保留 DuckDB 兼容别名：`DuckDBStorage = PostgresStorage`

2. 主链路已切换到 PostgresStorage
- [src/fs_explorer/agent.py](/C:/projects/agentic-file-search/src/fs_explorer/agent.py)
- [src/fs_explorer/indexing/pipeline.py](/C:/projects/agentic-file-search/src/fs_explorer/indexing/pipeline.py)
- [src/fs_explorer/search/query.py](/C:/projects/agentic-file-search/src/fs_explorer/search/query.py)
- [src/fs_explorer/server.py](/C:/projects/agentic-file-search/src/fs_explorer/server.py)
- [src/fs_explorer/main.py](/C:/projects/agentic-file-search/src/fs_explorer/main.py)

3. 配置层已改为 DSN 驱动
- [src/fs_explorer/index_config.py](/C:/projects/agentic-file-search/src/fs_explorer/index_config.py)
- 优先级：`FS_EXPLORER_DB_DSN` > `FS_EXPLORER_DB_PATH`(legacy) > 默认 DSN
- `.duckdb` 风格入参做了兼容归一化，避免旧调用直接报错

4. 依赖与初始化脚本已更新
- [pyproject.toml](/C:/projects/agentic-file-search/pyproject.toml)：移除 `duckdb`，增加 `psycopg[binary]`、`pgvector`
- [docker/init.sql](/C:/projects/agentic-file-search/docker/init.sql)：预置 `CREATE EXTENSION IF NOT EXISTS vector`
- [.env.example](/C:/projects/agentic-file-search/.env.example)：增加 `FS_EXPLORER_DB_DSN`

5. 本机 PostgreSQL 验证与项目库初始化已完成
- 已确认服务：`postgresql-x64-18` 运行中
- 使用账号 `postgres/root` 执行初始化
- 已创建：
  - 角色：`fs_explorer` / `devpassword`
  - 数据库：`fs_explorer`（owner=`fs_explorer`）

6. pgvector 缺失时的降级策略已实现
- 当前本机 `CREATE EXTENSION vector` 不可用（扩展未安装）
- 已实现“自动降级到非向量语义路径”：
  - 存储 embeddings 为 JSONB（用于保留数据）
  - 语义检索不可用时自动回退关键词检索
  - 不因缺少 pgvector 导致主流程中断

## 本次附带修复（测试稳定性）
1. 跨平台路径展示统一
- [src/fs_explorer/exploration_trace.py](/C:/projects/agentic-file-search/src/fs_explorer/exploration_trace.py)
- [src/fs_explorer/fs.py](/C:/projects/agentic-file-search/src/fs_explorer/fs.py)
- 修复 Windows 下路径分隔符导致的测试断言失败

## 测试结果
1. 关键 M1 回归
- `pytest tests/test_indexing.py tests/test_search.py tests/test_server_search.py -q`
- 结果：`40 passed`

2. 全量回归
- `pytest -q`
- 结果：`104 passed, 2 skipped`

3. 静态检查
- `ruff check`（本次改动相关文件）通过

## 当前风险与说明
1. 本机未安装 pgvector 数据库扩展
- 现状：PostgreSQL 可用，但 `vector` 扩展不可创建
- 影响：语义向量检索/HNSW 索引在本机降级
- 不影响：文档索引、关键词检索、API 主路径与测试通过

## 建议的下一步（进入 M2 前）
1. 在目标环境安装 pgvector 扩展并验证 `CREATE EXTENSION vector;`
2. 增加一条“扩展已安装”环境门禁检查（启动时提示）
3. 继续推进 M2：页级解析缓存、图片 hash 与 lazy 语义增强、WS 事件扩展

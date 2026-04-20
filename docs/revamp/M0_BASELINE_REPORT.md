# M0 基线检查记录

## 1. 执行时间

- 执行时间：2026-04-20 17:18:41 +08:00
- 执行目录：`C:\projects\agentic-file-search`
- 执行分支：`feat/api-ui-sqlite-openai`

## 2. M0 目标

1. 清理中断安装状态。
2. 恢复 `uv.lock` 到仓库基线。
3. 验证“无虚拟环境 + 中国大陆镜像源（清华源）”安装可行性。
4. 产出 `pytest` 与 `ruff` 基线结果清单。

## 3. 已执行动作

1. 检查并确认无残留 `uv/python` 进程。
2. 执行 `git checkout -- uv.lock`，恢复 lockfile。
3. 通过系统环境安装项目依赖（无 venv）：
   - `uv pip install --system -e ".[dev]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
   - 由于项目使用的是 dependency groups 而非 extras，补装 dev 工具：
   - `uv pip install --system pytest pytest-asyncio ruff pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 4. 环境信息

- Python：`3.14.3`
- uv：`0.11.3`
- 安装目标环境：`C:\Users\41948\AppData\Local\Python\pythoncore-3.14-64`
- 镜像源：清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`

## 5. 基线检查结果

## 5.1 pytest

- 命令：`python -m pytest -q`
- 结果：`5 failed, 99 passed, 2 skipped`
- 失败用例：
  1. `tests/test_exploration_trace.py::test_trace_records_steps_and_documents`
  2. `tests/test_exploration_trace.py::test_trace_records_resolved_document_paths`
  3. `tests/test_fs.py::TestDescribeDirContent::test_valid_directory`
  4. `tests/test_fs.py::TestDescribeDirContent::test_directory_without_subfolders`
  5. `tests/test_fs.py::TestDocumentParsing::test_parse_file_pdf`
- 初判原因：
  1. Windows 路径分隔符与测试断言（`/`）不一致导致 4 个测试失败。
  2. `test_parse_file_pdf` 依赖联网下载模型（Docling/RapidOCR/HuggingFace），在当前网络状态下出现 SSL EOF。

## 5.2 ruff

- 命令：`python -m ruff check .`
- 结果：`4 errors`
- 失败项：
  1. `scripts/generate_large_docs.py:9` `F401` 未使用 `PageBreak`
  2. `scripts/generate_large_docs.py:573` `F541` 无占位符 f-string
  3. `scripts/generate_test_docs.py:10` `F401` 未使用 `PageBreak`
  4. `scripts/generate_test_docs.py:756` `F541` 无占位符 f-string

## 6. M0 结论

1. “不使用虚拟环境 + 清华源”安装路径可行，已验证成功。
2. `uv.lock` 已恢复至基线，当前仓库不再有 lockfile 脏变更。
3. 基线质量状态已明确，后续可按失败清单执行修复或在迁移阶段做兼容处理。

## 7. 后续建议（进入 M1 前）

1. 先决定是否在当前阶段修复 Windows 路径断言测试，或在 PostgreSQL 改造阶段统一调整。
2. 对依赖网络下载模型的测试增加更稳定的 skip/mock 策略，避免 CI/离线环境波动。
3. 在正式实施 M1 前，冻结一次“基线失败清单”作为回归对照。


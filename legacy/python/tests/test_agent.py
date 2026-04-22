"""Tests for the FsExplorerAgent class."""

import io
import pytest
import os
from types import SimpleNamespace
from pathlib import Path

from unittest.mock import patch

import fs_explorer.agent as agent_module
from fs_explorer.blob_store import LocalBlobStore
from fs_explorer.agent import (
    FsExplorerAgent,
    SYSTEM_PROMPT,
    TokenUsage,
    _build_system_prompt,
    _parse_action_response,
    get_document,
    semantic_search,
    set_search_flags,
    set_image_semantic_enhancer,
    set_runtime_event_callback,
    get_search_flags,
    clear_index_context,
)
from fs_explorer.document_pages import page_record_from_manifest
from fs_explorer.models import Action, StopAction, ToolCallAction
from fs_explorer.page_store import StoredPage, render_page_markdown
from google.genai.types import Content, Part
from fs_explorer.storage import (
    ChunkRecord,
    DocumentRecord,
    DuckDBStorage,
    ImageSemanticRecord,
)
from .conftest import MockGenAIClient


def _mock_action_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(
                content=Content(role="model", parts=[Part.from_text(text=text)])
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )


class _SequenceModels:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    async def generate_content(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        if not self._texts:
            raise AssertionError("No more mock model responses configured")
        return _mock_action_response(self._texts.pop(0))


class _SequenceClient:
    def __init__(self, texts: list[str]) -> None:
        self.aio = SimpleNamespace(models=_SequenceModels(texts))


class TestAgentInitialization:
    """Tests for agent initialization."""
    
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    def test_agent_init_with_env_key(self) -> None:
        """Test agent initialization with API key from environment."""
        agent = FsExplorerAgent()
        assert hasattr(agent._client, "aio")
        assert len(agent._chat_history) == 0  # No system prompt in history
        assert isinstance(agent.token_usage, TokenUsage)

    def test_agent_init_with_explicit_key(self) -> None:
        """Test agent initialization with explicit API key."""
        agent = FsExplorerAgent(api_key="explicit-test-key")
        assert hasattr(agent._client, "aio")

    def test_agent_init_without_key_raises(self) -> None:
        """Test that initialization without API key raises ValueError."""
        # Ensure no key in environment
        env = os.environ.copy()
        env.pop("TEXT_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        env.pop("GOOGLE_API_KEY", None)
        
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Text model is not configured"):
                FsExplorerAgent()


class TestAgentConfiguration:
    """Tests for agent task configuration."""
    
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    def test_configure_task_adds_to_history(self) -> None:
        """Test that configure_task adds message to chat history."""
        agent = FsExplorerAgent()
        agent.configure_task("this is a task")
        
        assert len(agent._chat_history) == 1
        assert agent._chat_history[0].role == "user"
        assert agent._chat_history[0].parts[0].text == "this is a task"

    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    def test_multiple_configure_task_calls(self) -> None:
        """Test that multiple configure_task calls accumulate."""
        agent = FsExplorerAgent()
        agent.configure_task("task 1")
        agent.configure_task("task 2")
        
        assert len(agent._chat_history) == 2
        assert agent._chat_history[0].parts[0].text == "task 1"
        assert agent._chat_history[1].parts[0].text == "task 2"


class TestAgentActions:
    """Tests for agent action handling."""
    
    def test_parse_action_response_accepts_tool_json_with_text_context_plan(self) -> None:
        raw = r"""
{
  "action": {
    "tool_name": "read",
    "tool_input": [
      {
        "parameter_name": "file_path",
        "parameter_value": "D:\projects\agent-file-search\data\object_store\documents\report.pdf\pages\page-0033.md"
      }
    ]
  },
  "reason": "Read the most relevant page before answering.",
  "context_plan": "After reading this page, decide whether adjacent pages are needed."
}
"""
        action = _parse_action_response(raw)

        assert action is not None
        assert isinstance(action.action, ToolCallAction)
        assert action.action.tool_name == "read"
        assert action.action.to_fn_args() == {
            "file_path": r"D:\projects\agent-file-search\data\object_store\documents\report.pdf\pages\page-0033.md",
        }
        assert action.context_plan is None

    def test_parse_action_response_accepts_dict_tool_input(self) -> None:
        raw = """
{
  "action": {
    "tool_name": "read",
    "tool_input": {
      "file_path": "pages/page-0033.md"
    }
  },
  "reason": "Read the page."
}
"""
        action = _parse_action_response(raw)

        assert action is not None
        assert isinstance(action.action, ToolCallAction)
        assert action.action.tool_name == "read"
        assert action.action.to_fn_args() == {"file_path": "pages/page-0033.md"}

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    async def test_take_action_returns_action(self) -> None:
        """Test that take_action returns an action from the model."""
        agent = FsExplorerAgent()
        agent.configure_task("this is a task")
        agent._client = MockGenAIClient()
        
        result = await agent.take_action()
        
        assert result is not None
        action, action_type = result
        assert isinstance(action, Action)
        assert isinstance(action.action, StopAction)
        assert action.action.final_result == "this is a final result"
        assert action.reason == "I am done"
        assert action_type == "stop"

    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    def test_reset_clears_history(self) -> None:
        """Test that reset clears chat history and token usage."""
        agent = FsExplorerAgent()
        agent.configure_task("task 1")
        agent.token_usage.api_calls = 5
        
        agent.reset()
        
        assert len(agent._chat_history) == 0
        assert agent.token_usage.api_calls == 0

    def test_parse_action_response_accepts_fenced_json(self) -> None:
        raw = """
我先整理一下。

```json
{"action":{"final_result":"董事会成员有 Alice 和 Bob。"},"reason":"已经找到答案"}
```
"""
        action = _parse_action_response(raw)

        assert action is not None
        assert isinstance(action, Action)
        assert isinstance(action.action, StopAction)
        assert action.action.final_result == "董事会成员有 Alice 和 Bob。"

    def test_parse_action_response_accepts_compact_tool_json(self) -> None:
        raw = """
{
  "action": "semantic_search",
  "parameters": {
    "query": "董事会成员名单 董事 监事 高级管理人员",
    "filters": null,
    "limit": 10
  },
  "reason": "先定位相关章节"
}
"""
        action = _parse_action_response(raw)

        assert action is not None
        assert isinstance(action.action, ToolCallAction)
        assert action.action.tool_name == "semantic_search"
        assert action.action.to_fn_args() == {
            "query": "董事会成员名单 董事 监事 高级管理人员",
            "filters": None,
            "limit": 10,
        }

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    async def test_take_action_accepts_compact_tool_json(self) -> None:
        agent = FsExplorerAgent()
        agent.configure_task("请输出所有董事会成员")
        agent._client = _SequenceClient(
            [
                '{"action":"semantic_search","parameters":{"query":"董事会成员名单","limit":10},"reason":"先查找相关章节"}',
            ]
        )

        result = await agent.take_action()

        assert result is not None
        action, action_type = result
        assert action_type == "toolcall"
        assert isinstance(action.action, ToolCallAction)
        assert action.action.tool_name == "semantic_search"
        assert action.action.to_fn_args() == {
            "query": "董事会成员名单",
            "limit": 10,
        }

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    async def test_take_action_repairs_non_json_reply(self) -> None:
        agent = FsExplorerAgent()
        agent.configure_task("请输出所有董事会成员")
        agent._client = _SequenceClient(
            [
                "我来分析一下这个问题，然后去目录里查找相关信息。",
                '{"action":{"final_result":"董事会成员包括 Alice、Bob。"},"reason":"修复为合法 JSON"}',
            ]
        )

        result = await agent.take_action()

        assert result is not None
        action, action_type = result
        assert action_type == "stop"
        assert isinstance(action.action, StopAction)
        assert action.action.final_result == "董事会成员包括 Alice、Bob。"
        assert agent._client.aio.models.calls == 2

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    async def test_take_action_falls_back_to_stop_for_unstructured_text(self) -> None:
        raw_text = "我来分析一下这个目录，并输出所有董事会成员。"
        agent = FsExplorerAgent()
        agent.configure_task("请输出所有董事会成员")
        agent._client = _SequenceClient([raw_text, raw_text])

        result = await agent.take_action()

        assert result is not None
        action, action_type = result
        assert action_type == "stop"
        assert isinstance(action.action, StopAction)
        assert action.action.final_result == raw_text

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    async def test_take_action_blocks_repeated_identical_toolcall(self) -> None:
        repeated_tool = (
            '{"action":{"tool_name":"glob","tool_input":'
            '[{"parameter_name":"directory","parameter_value":"tests"},'
            '{"parameter_name":"pattern","parameter_value":"*.py"}]},'
            '"reason":"Search the test directory"}'
        )
        stop_after_guard = (
            '{"action":{"final_result":"Best-effort answer after avoiding a loop."},'
            '"reason":"Changed strategy after the loop guard warning."}'
        )

        agent = FsExplorerAgent()
        agent.configure_task("Avoid repeating the same tool call")
        agent._client = _SequenceClient(
            [
                repeated_tool,
                repeated_tool,
                stop_after_guard,
            ]
        )

        first_result = await agent.take_action()
        assert first_result is not None
        assert first_result[1] == "toolcall"

        second_result = await agent.take_action()
        assert second_result is not None
        action, action_type = second_result
        assert action_type == "stop"
        assert isinstance(action.action, StopAction)
        assert action.action.final_result == "Best-effort answer after avoiding a loop."
        assert agent._client.aio.models.calls == 3

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    async def test_take_action_allows_same_tool_with_different_parameters(self) -> None:
        read_first_page = (
            '{"action":{"tool_name":"read","tool_input":'
            '[{"parameter_name":"file_path","parameter_value":"tests/page-0033.md"}]},'
            '"reason":"Read the first candidate page"}'
        )
        read_adjacent_page = (
            '{"action":{"tool_name":"read","tool_input":'
            '[{"parameter_name":"file_path","parameter_value":"tests/page-0034.md"}]},'
            '"reason":"Read the adjacent page because the table may continue"}'
        )

        agent = FsExplorerAgent()
        agent.configure_task("Read adjacent pages when evidence spans pages")
        agent._client = _SequenceClient([read_first_page, read_adjacent_page])

        first_result = await agent.take_action()
        assert first_result is not None
        assert first_result[1] == "toolcall"

        second_result = await agent.take_action()
        assert second_result is not None
        assert second_result[1] == "toolcall"
        assert agent._client.aio.models.calls == 2

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    async def test_repeated_tool_guard_never_returns_generic_failure_message(self) -> None:
        repeated_tool = (
            '{"action":{"tool_name":"glob","tool_input":'
            '[{"parameter_name":"directory","parameter_value":"tests"},'
            '{"parameter_name":"pattern","parameter_value":"*.py"}]},'
            '"reason":"Search the test directory"}'
        )

        agent = FsExplorerAgent()
        agent.configure_task("Answer from current evidence if tools repeat")
        agent._client = _SequenceClient(
            [repeated_tool, repeated_tool, repeated_tool, repeated_tool]
        )

        first_result = await agent.take_action()
        assert first_result is not None
        assert first_result[1] == "toolcall"

        second_result = await agent.take_action()
        assert second_result is not None
        action, action_type = second_result

        assert action_type == "stop"
        assert isinstance(action.action, StopAction)
        assert "I could not make further progress without repeating" not in action.action.final_result
        assert "best available" in action.action.final_result.lower()
        assert agent._client.aio.models.calls == 4

    def test_parse_action_response_rejects_parse_file_tool(self) -> None:
        raw = """
{
  "action": {
    "tool_name": "parse_file",
    "tool_input": {
      "file_path": "report.pdf"
    }
  },
  "reason": "Reparse the original file."
}
"""
        action = _parse_action_response(raw)

        assert action is None


class TestTokenUsage:
    """Tests for TokenUsage tracking."""
    
    def test_add_api_call(self) -> None:
        """Test adding API call metrics."""
        usage = TokenUsage()
        usage.add_api_call(100, 50)
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.api_calls == 1

    def test_add_tool_result_preview_file(self) -> None:
        """Test tracking preview_file tool usage."""
        usage = TokenUsage()
        usage.add_tool_result("document content here", "preview_file")
        
        assert usage.documents_parsed == 1
        assert usage.tool_result_chars == len("document content here")

    def test_add_tool_result_scan_folder(self) -> None:
        """Test tracking scan_folder tool usage."""
        usage = TokenUsage()
        # Simulating scan output with document markers
        result = "│ [1/3] doc1.pdf\n│ [2/3] doc2.pdf\n│ [3/3] doc3.pdf"
        usage.add_tool_result(result, "scan_folder")
        
        assert usage.documents_scanned == 3

    def test_summary_format(self) -> None:
        """Test that summary produces formatted output."""
        usage = TokenUsage()
        usage.add_api_call(1000, 500)
        
        summary = usage.summary()
        
        assert "TOKEN USAGE SUMMARY" in summary
        assert "1,000" in summary  # Formatted prompt tokens
        assert "API Calls:" in summary
        assert "Est. Cost" in summary


class TestSystemPrompt:
    """Tests for system prompt configuration."""
    
    def test_system_prompt_contains_tools(self) -> None:
        """Test that system prompt documents all tools."""
        assert "scan_folder" in SYSTEM_PROMPT
        assert "preview_file" in SYSTEM_PROMPT
        assert "read" in SYSTEM_PROMPT
        assert "grep" in SYSTEM_PROMPT
        assert "glob" in SYSTEM_PROMPT
        assert "parse_file" not in SYSTEM_PROMPT

    def test_system_prompt_contains_strategy(self) -> None:
        """Test that system prompt includes exploration strategy."""
        assert "Three-Phase" in SYSTEM_PROMPT or "PHASE" in SYSTEM_PROMPT
        assert "Parallel Scan" in SYSTEM_PROMPT or "PARALLEL" in SYSTEM_PROMPT
        assert "Backtracking" in SYSTEM_PROMPT or "BACKTRACK" in SYSTEM_PROMPT
        assert "previous page and the next page" in SYSTEM_PROMPT
        assert "best-effort answer from existing evidence" in SYSTEM_PROMPT
        assert "tool name and all parameters are identical" in SYSTEM_PROMPT

    def test_system_prompt_contains_index_tools(self) -> None:
        """Test that system prompt documents index-aware tools."""
        assert "semantic_search" in SYSTEM_PROMPT
        assert "get_document" in SYSTEM_PROMPT
        assert "list_indexed_documents" in SYSTEM_PROMPT


class TestSearchFlags:
    """Tests for search flag state and dynamic system prompt."""

    def setup_method(self) -> None:
        clear_index_context()

    def teardown_method(self) -> None:
        clear_index_context()

    def test_set_and_get_search_flags(self) -> None:
        assert get_search_flags() == (False, False)
        set_search_flags(enable_semantic=True, enable_metadata=False)
        assert get_search_flags() == (True, False)
        set_search_flags(enable_semantic=False, enable_metadata=False)
        assert get_search_flags() == (False, False)

    def test_clear_index_context_resets_flags(self) -> None:
        set_search_flags(enable_semantic=True, enable_metadata=True)
        clear_index_context()
        assert get_search_flags() == (False, False)

    def test_build_system_prompt_no_index(self) -> None:
        prompt = _build_system_prompt(False, False)
        assert prompt == SYSTEM_PROMPT

    def test_build_system_prompt_semantic_only(self) -> None:
        prompt = _build_system_prompt(True, False)
        assert "Semantic Only" in prompt
        assert "WITHOUT the `filters`" in prompt

    def test_build_system_prompt_metadata_only(self) -> None:
        prompt = _build_system_prompt(False, True)
        assert "Metadata Only" in prompt
        assert "metadata filtering" in prompt

    def test_build_system_prompt_both(self) -> None:
        prompt = _build_system_prompt(True, True)
        assert "Semantic + Metadata" in prompt

    @patch.dict(os.environ, {"TEXT_API_KEY": "test-api-key"})
    def test_all_tools_always_available(self) -> None:
        """Filesystem and indexed tools are never blocked."""
        set_search_flags(enable_semantic=False, enable_metadata=False)
        agent = FsExplorerAgent()
        agent.configure_task("test")
        agent.call_tool("glob", {"directory": "/tmp", "pattern": "*.md"})

        last = agent._chat_history[-1]
        assert "not available" not in last.parts[0].text


class TestImageSemanticEnhancement:
    def teardown_method(self) -> None:
        clear_index_context()

    @staticmethod
    def _seed_indexed_pdf(tmp_path):
        storage = DuckDBStorage(str(tmp_path / "index.duckdb"))
        object_store = LocalBlobStore(tmp_path / "object_store")
        corpus_id = storage.get_or_create_corpus(str(tmp_path.resolve()))
        doc_id = storage.make_document_id(corpus_id, "report.pdf")
        document = DocumentRecord(
            id=doc_id,
            corpus_id=corpus_id,
            relative_path="report.pdf",
            absolute_path=str(tmp_path / "report.pdf"),
            content="Quarterly review includes revenue chart and commentary.",
            metadata_json="{}",
            file_mtime=0.0,
            file_size=128,
            content_sha256="sha-report",
            original_filename="report.pdf",
            pages_prefix="documents/report.pdf/pages",
        )
        chunk = ChunkRecord(
            id=storage.make_chunk_id(doc_id, 0, 0, 56),
            doc_id=doc_id,
            text="Quarterly review includes revenue chart and commentary.",
            position=0,
            start_char=0,
            end_char=56,
        )
        storage.upsert_document(document, [chunk])
        storage.sync_document_pages(
            document_id=doc_id,
            pages=[
                page_record_from_manifest(
                    StoredPage(
                        page_no=1,
                        object_key="documents/report.pdf/pages/page-0001.md",
                        heading="Quarterly review",
                        source_locator="page-1",
                        content_hash="page-1-hash",
                        char_count=56,
                        is_synthetic_page=False,
                    ),
                    document_id=doc_id,
                )
            ],
        )
        object_store.put(
            object_key="documents/report.pdf/pages/page-0001.md",
            data=io.BytesIO(
                render_page_markdown(
                    document_id=doc_id,
                    original_filename="report.pdf",
                    page_no=1,
                    page_label="1",
                    content_type="application/pdf",
                    source_locator="page-1",
                    heading="Quarterly review",
                    body="Quarterly review includes revenue chart and commentary.",
                ).encode("utf-8")
            ),
        )
        storage.upsert_image_semantics(
            images=[
                ImageSemanticRecord(
                    image_hash="img-1",
                    source_document_id=doc_id,
                    source_page_no=1,
                    source_image_index=1,
                    mime_type="image/png",
                    width=320,
                    height=200,
                )
            ]
        )
        return storage, object_store, corpus_id, doc_id

    def test_semantic_search_lazily_enhances_hit_images(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        storage, object_store, corpus_id, _ = self._seed_indexed_pdf(tmp_path)

        class StubEnhancer:
            def __init__(self) -> None:
                self.calls = 0

            def describe_image(self, **kwargs):  # noqa: ANN003
                self.calls += 1
                return "Revenue bar chart with quarterly labels.", "stub-vision"

        enhancer = StubEnhancer()
        monkeypatch.setattr(
            agent_module,
            "_get_index_storage_and_corpus",
            lambda: (storage, corpus_id, None),
        )
        monkeypatch.setattr(agent_module, "_PAGE_BLOB_STORE", object_store)
        monkeypatch.setattr(
            "fs_explorer.document_parsing._extract_pdf_images",
            lambda file_path, page_no, include_bytes=False: [
                {
                    "image_hash": "img-1",
                    "image_index": 1,
                    "mime_type": "image/png",
                    "width": 320,
                    "height": 200,
                    "image_bytes": b"fake-image",
                }
            ],
        )
        set_search_flags(enable_semantic=True, enable_metadata=False)
        set_image_semantic_enhancer(enhancer)

        first = semantic_search("revenue chart")
        second = semantic_search("revenue chart")

        semantics = storage.get_image_semantics(image_hashes=["img-1"])
        assert enhancer.calls == 1
        assert semantics["img-1"]["semantic_text"] == "Revenue bar chart with quarterly labels."
        assert "enhanced 1 images" in first
        assert "reused cached semantics for 1 pages" in second
        storage.close()

    def test_semantic_search_emits_runtime_cache_and_enhancement_events(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        storage, object_store, corpus_id, doc_id = self._seed_indexed_pdf(tmp_path)

        class StubEnhancer:
            def describe_image(self, **kwargs):  # noqa: ANN003
                return "Revenue bar chart with quarterly labels.", "stub-vision"

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            agent_module,
            "_get_index_storage_and_corpus",
            lambda: (storage, corpus_id, None),
        )
        monkeypatch.setattr(agent_module, "_PAGE_BLOB_STORE", object_store)
        monkeypatch.setattr(
            "fs_explorer.document_parsing._extract_pdf_images",
            lambda file_path, page_no, include_bytes=False: [
                {
                    "image_hash": "img-1",
                    "image_index": 1,
                    "mime_type": "image/png",
                    "width": 320,
                    "height": 200,
                    "image_bytes": b"fake-image",
                }
            ],
        )
        set_search_flags(enable_semantic=True, enable_metadata=False)
        set_image_semantic_enhancer(StubEnhancer())
        set_runtime_event_callback(lambda event_type, data: events.append((event_type, dict(data))))

        semantic_search("revenue chart")
        semantic_search("revenue chart")

        assert [event_type for event_type, _ in events] == [
            "image_enhance_started",
            "image_enhance_done",
            "cache_hit",
        ]
        assert events[0][1]["doc_id"] == doc_id
        assert events[0][1]["page_no"] == 1
        assert events[0][1]["pending_images"] == 1
        assert events[1][1]["enhanced_images"] == 1
        assert events[2][1]["cache_kind"] == "image_semantics"
        assert events[2][1]["cached_images"] == 1
        storage.close()

    def test_get_document_includes_cached_image_semantics(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        storage, object_store, corpus_id, doc_id = self._seed_indexed_pdf(tmp_path)
        storage.update_image_semantic(
            image_hash="img-1",
            semantic_text="Revenue bar chart with quarterly labels.",
            semantic_model="stub-vision",
        )
        monkeypatch.setattr(
            agent_module,
            "_get_index_storage_and_corpus",
            lambda: (storage, corpus_id, None),
        )
        monkeypatch.setattr(agent_module, "_PAGE_BLOB_STORE", object_store)

        rendered = get_document(doc_id)

        assert "Image semantics cache:" in rendered
        assert "Page 1 (Quarterly review):" in rendered
        assert "image 1: Revenue bar chart with quarterly labels." in rendered
        storage.close()

    def test_get_document_falls_back_to_page_blobs_when_content_is_empty(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        storage = DuckDBStorage(str(tmp_path / "index.duckdb"))
        blob_store = LocalBlobStore(tmp_path / "object_store")
        corpus_id = storage.get_or_create_corpus(str(tmp_path.resolve()))
        doc_id = storage.make_document_id(corpus_id, "report.pdf")
        document = DocumentRecord(
            id=doc_id,
            corpus_id=corpus_id,
            relative_path="report.pdf",
            absolute_path=str(tmp_path / "report.pdf"),
            content="",
            metadata_json="{}",
            file_mtime=0.0,
            file_size=128,
            content_sha256="sha-report",
            original_filename="report.pdf",
            pages_prefix="documents/report.pdf/pages",
        )
        storage.upsert_document_stub(document)
        storage.sync_document_pages(
            document_id=doc_id,
            pages=[
                page_record_from_manifest(
                    StoredPage(
                        page_no=1,
                        object_key="documents/report.pdf/pages/page-0001.md",
                        heading="Summary",
                        source_locator="page-1",
                        content_hash="page-1",
                        char_count=17,
                        is_synthetic_page=False,
                    ),
                    document_id=doc_id,
                ),
                page_record_from_manifest(
                    StoredPage(
                        page_no=2,
                        object_key="documents/report.pdf/pages/page-0002.md",
                        heading="Details",
                        source_locator="page-2",
                        content_hash="page-2",
                        char_count=17,
                        is_synthetic_page=False,
                    ),
                    document_id=doc_id,
                ),
            ],
        )
        blob_store.put(
            object_key="documents/report.pdf/pages/page-0001.md",
            data=io.BytesIO(b"Page one summary.\n"),
        )
        blob_store.put(
            object_key="documents/report.pdf/pages/page-0002.md",
            data=io.BytesIO(b"Page two details.\n"),
        )
        monkeypatch.setattr(agent_module, "_PAGE_BLOB_STORE", blob_store)
        monkeypatch.setattr(
            agent_module,
            "_get_index_storage_and_corpus",
            lambda: (storage, corpus_id, None),
        )

        rendered = get_document(doc_id)

        assert "Page one summary." in rendered
        assert "Page two details." in rendered
        storage.close()

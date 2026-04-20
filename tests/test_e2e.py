import os

import pytest

from workflows.testing import WorkflowTestRunner


SKIP_IF, SKIP_REASON = (
    not os.getenv("RUN_REAL_E2E_TESTS") or not os.getenv("GOOGLE_API_KEY"),
    "Set RUN_REAL_E2E_TESTS=1 and GOOGLE_API_KEY to enable the real end-to-end workflow test.",
)


@pytest.mark.asyncio
@pytest.mark.skipif(condition=SKIP_IF, reason=SKIP_REASON)
async def test_e2e() -> None:
    from fs_explorer.workflow import (
        ExplorationEndEvent,
        GoDeeperEvent,
        InputEvent,
        ToolCallEvent,
        workflow,
    )

    start_event = InputEvent(
        task="Starting from the current directory, individuate the python file responsible for file system operations and explain what it does"
    )
    runner = WorkflowTestRunner(workflow=workflow)
    result = await runner.run(start_event=start_event)
    assert isinstance(result.result, ExplorationEndEvent)
    assert result.result.error is None
    assert result.result.final_result is not None
    assert len(result.collected) > 1
    assert ToolCallEvent in result.event_types or GoDeeperEvent in result.event_types

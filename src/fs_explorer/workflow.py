"""
Workflow orchestration for the FsExplorer agent.

This module defines the event-driven workflow that coordinates the agent's
exploration of the filesystem, handling tool calls, directory navigation,
and human interaction.
"""

import contextvars
import os

from workflows import Workflow, Context, step
from workflows.events import (
    StartEvent,
    StopEvent,
    Event,
    InputRequiredEvent,
    HumanResponseEvent,
)
from workflows.resource import Resource
from pydantic import BaseModel
from typing import Annotated, cast, Any

from .agent import FsExplorerAgent
from .models import GoDeeperAction, ToolCallAction, StopAction, AskHumanAction, Action
from .fs import describe_dir_content

# Per-asyncio-task agent storage — each exploration session gets its own.
_AGENT_VAR: contextvars.ContextVar[FsExplorerAgent | None] = contextvars.ContextVar(
    "_AGENT_VAR", default=None
)


def get_agent() -> FsExplorerAgent:
    """Get or create the agent instance for the current context."""
    agent = _AGENT_VAR.get()
    if agent is None:
        agent = FsExplorerAgent()
        _AGENT_VAR.set(agent)
    return agent


def reset_agent() -> None:
    """Reset the agent instance for the current context."""
    _AGENT_VAR.set(None)


class WorkflowState(BaseModel):
    """State maintained throughout the workflow execution."""

    initial_task: str = ""
    document_labels: list[str] = []
    collection_name: str | None = None
    enable_semantic: bool = False
    enable_metadata: bool = False


class InputEvent(StartEvent):
    """Initial event containing the user's task."""

    task: str
    document_ids: list[str] = []
    document_labels: list[str] = []
    collection_name: str | None = None
    enable_semantic: bool = False
    enable_metadata: bool = False


class GoDeeperEvent(Event):
    """Event triggered when navigating into a subdirectory."""

    directory: str
    reason: str
    context_plan: dict[str, Any] | None = None


class ToolCallEvent(Event):
    """Event triggered when executing a tool."""

    tool_name: str
    tool_input: dict[str, Any]
    reason: str
    context_plan: dict[str, Any] | None = None


class AskHumanEvent(InputRequiredEvent):
    """Event triggered when human input is required."""

    question: str
    reason: str
    context_plan: dict[str, Any] | None = None


class HumanAnswerEvent(HumanResponseEvent):
    """Event containing the human's response."""

    response: str


class ExplorationEndEvent(StopEvent):
    """Event signaling the end of exploration."""

    final_result: str | None = None
    error: str | None = None


# Type alias for the union of possible workflow events
WorkflowEvent = ExplorationEndEvent | GoDeeperEvent | ToolCallEvent | AskHumanEvent


def _handle_action_result(
    action: Action,
    action_type: str,
    ctx: Context[WorkflowState],
) -> WorkflowEvent:
    """
    Convert an action result into the appropriate workflow event.

    This helper extracts the common logic for handling agent action results,
    reducing code duplication across workflow steps.

    Args:
        action: The action returned by the agent
        action_type: The type of action ("godeeper", "toolcall", "askhuman", "stop")
        ctx: The workflow context for state updates and event streaming

    Returns:
        The appropriate workflow event based on the action type
    """
    if action_type == "godeeper":
        godeeper = cast(GoDeeperAction, action.action)
        event = GoDeeperEvent(
            directory=godeeper.directory,
            reason=action.reason,
            context_plan=(
                action.context_plan.model_dump(exclude_none=True)
                if action.context_plan is not None
                else None
            ),
        )
        ctx.write_event_to_stream(event)
        return event

    elif action_type == "toolcall":
        toolcall = cast(ToolCallAction, action.action)
        event = ToolCallEvent(
            tool_name=toolcall.tool_name,
            tool_input=toolcall.to_fn_args(),
            reason=action.reason,
            context_plan=(
                action.context_plan.model_dump(exclude_none=True)
                if action.context_plan is not None
                else None
            ),
        )
        ctx.write_event_to_stream(event)
        return event

    elif action_type == "askhuman":
        askhuman = cast(AskHumanAction, action.action)
        # InputRequiredEvent is written to the stream by default
        return AskHumanEvent(
            question=askhuman.question,
            reason=action.reason,
            context_plan=(
                action.context_plan.model_dump(exclude_none=True)
                if action.context_plan is not None
                else None
            ),
        )

    else:  # stop
        stopaction = cast(StopAction, action.action)
        return ExplorationEndEvent(final_result=stopaction.final_result)


async def _process_agent_action(
    agent: FsExplorerAgent,
    ctx: Context[WorkflowState],
    update_directory: bool = False,
) -> WorkflowEvent:
    """
    Process the agent's next action and return the appropriate event.

    Args:
        agent: The agent instance
        ctx: The workflow context
        update_directory: Whether to update the current directory on godeeper action

    Returns:
        The appropriate workflow event
    """
    result = await agent.take_action()

    if result is None:
        return ExplorationEndEvent(error="Could not produce action to take")

    action, action_type = result

    # Update directory state if needed for godeeper actions
    if update_directory and action_type == "godeeper":
        godeeper = cast(GoDeeperAction, action.action)
        async with ctx.store.edit_state() as state:
            state.current_directory = godeeper.directory

    return _handle_action_result(action, action_type, ctx)


class FsExplorerWorkflow(Workflow):
    """
    Event-driven workflow for filesystem exploration.

    Coordinates the agent's actions through a series of steps:
    - start_exploration: Initial task processing
    - go_deeper_action: Directory navigation
    - tool_call_action: Tool execution
    - receive_human_answer: Human interaction handling
    """

    @step
    async def start_exploration(
        self,
        ev: InputEvent,
        ctx: Context[WorkflowState],
        agent: Annotated[FsExplorerAgent, Resource(get_agent)],
    ) -> WorkflowEvent:
        """Initialize exploration with the user's task."""
        async with ctx.store.edit_state() as state:
            state.initial_task = ev.task
            state.document_labels = list(ev.document_labels)
            state.collection_name = ev.collection_name
            state.enable_semantic = ev.enable_semantic
            state.enable_metadata = ev.enable_metadata

        scope_name = (
            f"collection '{ev.collection_name}'"
            if ev.collection_name
            else "selected documents"
        )
        document_summary = "\n".join(f"- {label}" for label in ev.document_labels) or "- (none)"
        if ev.enable_semantic and ev.enable_metadata:
            index_hint = (
                "Use `semantic_search` first, optionally with metadata filters, "
                "then use `get_document` or focused `parse_file` on the returned files."
            )
        elif ev.enable_semantic:
            index_hint = (
                "Use `semantic_search` first for similarity retrieval, then drill in."
            )
        elif ev.enable_metadata:
            index_hint = (
                "Use `semantic_search` with filters first, then drill in."
            )
        else:
            index_hint = (
                "Only work inside the selected document set and prefer focused reads."
            )
        agent.configure_task(
            f"You can only access the current {scope_name}. The available documents are:\n\n"
            f"```text\n{document_summary}\n```\n\n"
            f"The user task is: '{ev.task}'. "
            f"What action should you take first? {index_hint}"
        )

        return await _process_agent_action(agent, ctx, update_directory=False)

    @step
    async def go_deeper_action(
        self,
        ev: GoDeeperEvent,
        ctx: Context[WorkflowState],
        agent: Annotated[FsExplorerAgent, Resource(get_agent)],
    ) -> WorkflowEvent:
        """Handle navigation into a subdirectory."""
        state = await ctx.store.get_state()
        document_summary = "\n".join(f"- {label}" for label in state.document_labels) or "- (none)"

        agent.configure_task(
            f"Stay within the selected documents below:\n\n```text\n{document_summary}\n```\n\n"
            f"The user task is still: '{state.initial_task}'. "
            f"Based on what you learned, what action should you take next?"
        )

        return await _process_agent_action(agent, ctx, update_directory=False)

    @step
    async def receive_human_answer(
        self,
        ev: HumanAnswerEvent,
        ctx: Context[WorkflowState],
        agent: Annotated[FsExplorerAgent, Resource(get_agent)],
    ) -> WorkflowEvent:
        """Process the human's response to a question."""
        state = await ctx.store.get_state()

        agent.configure_task(
            f"Human response to your question: {ev.response}\n\n"
            f"Based on it, proceed with your exploration based on the "
            f"original task: {state.initial_task}"
        )

        return await _process_agent_action(agent, ctx, update_directory=False)

    @step
    async def tool_call_action(
        self,
        ev: ToolCallEvent,
        ctx: Context[WorkflowState],
        agent: Annotated[FsExplorerAgent, Resource(get_agent)],
    ) -> WorkflowEvent:
        """Process the result of a tool call."""
        state = await ctx.store.get_state()
        agent.configure_task(
            f"The user task is still: '{state.initial_task}'. "
            "Given the tool result and structured context you just received, "
            "choose the next action. Do not assume the previously active pages still contain the answer. "
            "If the current range is stale or insufficient, run a fresh search or parse a genuinely new range."
        )

        return await _process_agent_action(agent, ctx, update_directory=True)


# Workflow timeout for complex multi-document analysis (5 minutes)
WORKFLOW_TIMEOUT_SECONDS = 3000

workflow = FsExplorerWorkflow(timeout=WORKFLOW_TIMEOUT_SECONDS)

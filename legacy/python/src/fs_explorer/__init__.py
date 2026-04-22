"""
FsExplorer package namespace.

Keep package imports lightweight so submodules can be used without eagerly
loading the full agent stack and its optional runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "FsExplorerAgent",
    "TokenUsage",
    "workflow",
    "FsExplorerWorkflow",
    "InputEvent",
    "ExplorationEndEvent",
    "ToolCallEvent",
    "GoDeeperEvent",
    "AskHumanEvent",
    "HumanAnswerEvent",
    "get_agent",
    "reset_agent",
    "Action",
    "ActionType",
    "Tools",
]


def __getattr__(name: str):
    if name in {"FsExplorerAgent", "TokenUsage"}:
        module = import_module(".agent", __name__)
        return getattr(module, name)
    if name in {
        "workflow",
        "FsExplorerWorkflow",
        "InputEvent",
        "ExplorationEndEvent",
        "ToolCallEvent",
        "GoDeeperEvent",
        "AskHumanEvent",
        "HumanAnswerEvent",
        "get_agent",
        "reset_agent",
    }:
        module = import_module(".workflow", __name__)
        return getattr(module, name)
    if name in {"Action", "ActionType", "Tools"}:
        module = import_module(".models", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

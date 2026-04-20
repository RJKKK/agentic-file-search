"""
Configuration helpers for index storage.
"""

from __future__ import annotations

import os


DEFAULT_DB_PATH = "postgresql://fs_explorer:devpassword@127.0.0.1:5432/fs_explorer"
ENV_DB_DSN = "FS_EXPLORER_DB_DSN"
ENV_DB_PATH_LEGACY = "FS_EXPLORER_DB_PATH"


def resolve_db_path(override_path: str | None = None) -> str:
    """
    Resolve the PostgreSQL DSN from request override, env var, or default.

    Precedence:
    1) explicit override_path
    2) FS_EXPLORER_DB_DSN
    3) FS_EXPLORER_DB_PATH (legacy alias)
    4) default DSN
    """
    return (
        override_path
        or os.getenv(ENV_DB_DSN)
        or os.getenv(ENV_DB_PATH_LEGACY)
        or DEFAULT_DB_PATH
    )

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
    2) FS_EXPLORER_DB_PATH when it points to a local file path
    3) FS_EXPLORER_DB_DSN
    4) default DSN
    """
    legacy_path = os.getenv(ENV_DB_PATH_LEGACY)
    if override_path:
        return override_path
    if legacy_path and "://" not in legacy_path:
        return legacy_path
    return (
        os.getenv(ENV_DB_DSN)
        or legacy_path
        or DEFAULT_DB_PATH
    )

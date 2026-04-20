"""
Backward-compatible storage shim.

DuckDB has been replaced by PostgreSQL + pgvector for the primary execution path.
Keep this module to avoid import breakage while the rest of the codebase migrates.
"""

from .postgres import PostgresStorage

DuckDBStorage = PostgresStorage

__all__ = ["DuckDBStorage"]


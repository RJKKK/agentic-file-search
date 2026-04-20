"""Storage backends for FsExplorer indexing."""

from .base import ChunkRecord, DocumentRecord, SchemaRecord, StorageBackend
from .postgres import PostgresStorage

# Backward-compatible alias while imports are migrated.
DuckDBStorage = PostgresStorage

__all__ = [
    "ChunkRecord",
    "DocumentRecord",
    "SchemaRecord",
    "StorageBackend",
    "PostgresStorage",
    "DuckDBStorage",
]

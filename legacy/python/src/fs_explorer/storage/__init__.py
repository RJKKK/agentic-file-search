"""Storage backends for FsExplorer indexing."""

from .base import (
    ChunkRecord,
    CollectionRecord,
    DocumentRecord,
    DocumentPageRecord,
    ImageSemanticRecord,
    ParsedUnitRecord,
    SchemaRecord,
    StorageBackend,
)
from .postgres import PostgresStorage

# Backward-compatible alias while imports are migrated.
DuckDBStorage = PostgresStorage

__all__ = [
    "ChunkRecord",
    "CollectionRecord",
    "DocumentRecord",
    "DocumentPageRecord",
    "ParsedUnitRecord",
    "ImageSemanticRecord",
    "SchemaRecord",
    "StorageBackend",
    "PostgresStorage",
    "DuckDBStorage",
]

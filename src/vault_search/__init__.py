"""vault-search: vault-agnostic local-first hybrid retrieval."""

from vault_search.chunk import chunk_vault
from vault_search.embed import Embedder
from vault_search.index import Index, build_index
from vault_search.models import Chunk, SearchHit, VaultProfile
from vault_search.search import search

__version__ = "0.2.0"

__all__ = [
    "VaultProfile", "Chunk", "SearchHit", "Embedder",
    "Index", "build_index", "chunk_vault", "search", "__version__",
]

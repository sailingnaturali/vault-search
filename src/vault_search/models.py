"""Core dataclasses: Chunk, VaultProfile, SearchHit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    doc_path: str
    ordinal: int
    text: str
    embed_text: str
    metadata: dict
    citation: str

    @staticmethod
    def make(doc_path: str, ordinal: int, text: str, embed_text: str,
             metadata: dict, citation: str) -> "Chunk":
        raw = f"{doc_path}#{ordinal}".encode("utf-8")
        cid = hashlib.sha1(raw).hexdigest()[:16]
        return Chunk(id=cid, doc_path=doc_path, ordinal=ordinal, text=text,
                     embed_text=embed_text, metadata=metadata, citation=citation)


@dataclass
class VaultProfile:
    glob: str
    front_matter_fields: list[str]
    chunk_strategy: str           # "whole_file" | "headings"
    breadcrumb: str               # str.format template over front-matter fields
    citation: str                 # str.format template over front-matter fields
    max_tokens: int = 512         # word-count proxy for "headings" strategy
    overlap_tokens: int = 64


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    retriever: str = "hybrid"     # "bm25" | "vector" | "hybrid"

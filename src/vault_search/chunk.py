"""Markdown -> Chunk list. Strategy driven by VaultProfile."""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from vault_search.models import Chunk, VaultProfile


def _meta(post: frontmatter.Post, profile: VaultProfile) -> dict:
    return {f: str(post.get(f, "")) for f in profile.front_matter_fields}


def _render(template: str, meta: dict) -> str:
    """Render a str.format template against meta; returns the raw template unchanged if a referenced field is missing."""
    try:
        return template.format(**meta)
    except (KeyError, IndexError):
        return template


def chunk_document(path: Path, vault_root: Path, profile: VaultProfile) -> list[Chunk]:
    post = frontmatter.load(str(path))
    meta = _meta(post, profile)
    rel = str(path.relative_to(vault_root))
    citation = _render(profile.citation, meta)
    breadcrumb = _render(profile.breadcrumb, meta)
    body = post.content.strip()

    if profile.chunk_strategy == "whole_file":
        return [Chunk.make(doc_path=rel, ordinal=0, text=body,
                           embed_text=f"{breadcrumb}\n\n{body}",
                           metadata=meta, citation=citation)]
    if profile.chunk_strategy == "headings":
        return _chunk_headings(body, rel, breadcrumb, citation, meta, profile)
    raise ValueError(f"unknown chunk_strategy: {profile.chunk_strategy}")


_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)


def _split_sections(body: str) -> list[str]:
    starts = [m.start() for m in _HEADING.finditer(body)]
    if not starts:
        return [body] if body.strip() else []
    if starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [len(body)]
    out = []
    for i in range(len(starts)):
        seg = body[bounds[i]:bounds[i + 1]].strip()
        if seg:
            out.append(seg)
    return out


def _window(section: str, max_tokens: int, overlap: int) -> list[str]:
    words = section.split()
    if len(words) <= max_tokens:
        return [section]
    step = max(1, max_tokens - overlap)
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i + max_tokens]))
        if i + max_tokens >= len(words):
            break
        i += step
    return out


def _chunk_headings(body, rel, breadcrumb, citation, meta, profile) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for section in _split_sections(body):
        for piece in _window(section, profile.max_tokens, profile.overlap_tokens):
            chunks.append(Chunk.make(
                doc_path=rel, ordinal=ordinal, text=piece,
                embed_text=f"{breadcrumb}\n\n{piece}",
                metadata=meta, citation=citation))
            ordinal += 1
    return chunks


def chunk_vault(vault_root: Path, profile: VaultProfile) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(vault_root.glob(profile.glob)):
        chunks.extend(chunk_document(path, vault_root, profile))
    return chunks

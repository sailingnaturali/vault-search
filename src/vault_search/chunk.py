"""Markdown -> Chunk list. Strategy driven by VaultProfile."""

from __future__ import annotations

from pathlib import Path

import frontmatter

from vault_search.models import Chunk, VaultProfile


def _meta(post: frontmatter.Post, profile: VaultProfile) -> dict:
    return {f: str(post.get(f, "")) for f in profile.front_matter_fields}


def _render(template: str, meta: dict) -> str:
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
    raise ValueError(f"unknown chunk_strategy: {profile.chunk_strategy}")

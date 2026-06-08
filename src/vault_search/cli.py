"""vault-search CLI: build / query / eval."""

from __future__ import annotations

from pathlib import Path

import typer

from vault_search.chunk import chunk_vault
from vault_search.embed import Embedder
from vault_search.index import Index, build_index
from vault_search.profiles import PROFILES
from vault_search.search import search

app = typer.Typer(help="Local hybrid retrieval over markdown vaults.")


@app.command()
def build(vault: str, profile: str = "colregs", db: str = "vault.db") -> None:
    """Chunk + embed a vault into a SQLite index."""
    prof = PROFILES[profile]
    chunks = chunk_vault(Path(vault), prof)
    build_index(Path(db), chunks, Embedder())
    typer.echo(f"indexed {len(chunks)} chunks -> {db}")


@app.command()
def query(text: str, db: str = "vault.db", limit: int = 5) -> None:
    """Run a hybrid query against a built index."""
    emb = Embedder()
    idx = Index.open(Path(db), emb)
    for hit in search(idx, emb, text, limit=limit):
        typer.echo(f"[{hit.score:.4f}] {hit.chunk.citation}")
        typer.echo(f"    {hit.chunk.text[:120].strip()}…")


@app.command()
def evaluate(golden: str, vault: str, profile: str = "colregs",
             db: str = "vault.db") -> None:
    """Score keyword vs vector vs hybrid on a golden query set."""
    from vault_search.eval import run_eval
    run_eval(Path(golden), Path(vault), PROFILES[profile], Path(db))

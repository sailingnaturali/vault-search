from pathlib import Path

from vault_search.chunk import chunk_vault
from vault_search.embed import Embedder
from vault_search.index import Index, build_index
from vault_search.models import VaultProfile

VAULT = Path(__file__).parent / "fixtures" / "vault"

COLREGS = VaultProfile(
    glob="rules/**/*.md",
    front_matter_fields=["number", "regime", "title", "source_pdf"],
    chunk_strategy="whole_file",
    breadcrumb="Rule {number} ({regime}) — {title}",
    citation="Rule {number} ({regime})",
)


def test_build_then_bm25_and_knn_find_the_rule(tmp_path):
    db = tmp_path / "colregs.db"
    emb = Embedder()
    build_index(db, chunk_vault(VAULT, COLREGS), emb)

    idx = Index.open(db, emb)
    bm = idx.bm25("sidelights sternlight", n=5)
    assert bm                                  # rowids, best first
    assert idx.get_chunk(bm[0]).metadata["number"] == "25"

    kn = idx.knn(emb.encode(["lights for a boat under sail"])[0], n=5)
    assert kn
    assert idx.get_chunk(kn[0]).metadata["number"] == "25"


def test_rebuild_is_idempotent(tmp_path):
    db = tmp_path / "c.db"
    emb = Embedder()
    chunks = chunk_vault(VAULT, COLREGS)
    build_index(db, chunks, emb)
    build_index(db, chunks, emb)               # second build must not error or double-count
    idx = Index.open(db, emb)
    assert idx.count() == len(chunks)

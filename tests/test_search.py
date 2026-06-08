from pathlib import Path

from vault_search.chunk import chunk_vault
from vault_search.embed import Embedder
from vault_search.index import Index, build_index
from vault_search.models import VaultProfile
from vault_search.search import rrf_fuse, search

VAULT = Path(__file__).parent / "fixtures" / "vault"
COLREGS = VaultProfile(
    glob="rules/**/*.md", front_matter_fields=["number", "regime", "title", "source_pdf"],
    chunk_strategy="whole_file", breadcrumb="Rule {number} ({regime}) — {title}",
    citation="Rule {number} ({regime})")


def test_rrf_fuse_known_order():
    # two rankings of rowids; RRF with k=60 sums 1/(k+rank), rank starts at 1
    bm25 = [1, 2, 3]
    vector = [2, 1, 4]
    fused = rrf_fuse([bm25, vector], k=60)
    ids = [rowid for rowid, _ in fused]
    # 1: 1/61 + 1/62 ; 2: 1/62 + 1/61  -> tie, broken by smaller rowid -> 1 before 2
    assert ids[0] == 1
    assert ids[1] == 2
    assert set(ids[2:]) == {3, 4}


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([[5, 9], [5, 7]], k=60)   # 5 appears in both, top of each
    assert fused[0][0] == 5


def test_search_returns_cited_hits(tmp_path):
    db = tmp_path / "s.db"
    emb = Embedder()
    build_index(db, chunk_vault(VAULT, COLREGS), emb)
    idx = Index.open(db)
    hits = search(idx, emb, "lights for a boat under sail", limit=3)
    assert hits
    assert isinstance(hits[0].score, float)
    assert hits[0].chunk.citation == "Rule 25 (international)"

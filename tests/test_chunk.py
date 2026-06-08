from pathlib import Path

from vault_search.chunk import chunk_document
from vault_search.models import VaultProfile

VAULT = Path(__file__).parent / "fixtures" / "vault"

COLREGS = VaultProfile(
    glob="rules/**/*.md",
    front_matter_fields=["number", "regime", "title", "source_pdf"],
    chunk_strategy="whole_file",
    breadcrumb="Rule {number} ({regime}) — {title}",
    citation="Rule {number} ({regime})",
)


def test_whole_file_yields_single_chunk():
    chunks = chunk_document(VAULT / "rules" / "rule-25.md", VAULT, COLREGS)
    assert len(chunks) == 1


def test_whole_file_lifts_front_matter_and_breadcrumb():
    c = chunk_document(VAULT / "rules" / "rule-25.md", VAULT, COLREGS)[0]
    assert c.metadata["number"] == "25"
    assert c.doc_path == "rules/rule-25.md"
    assert c.citation == "Rule 25 (international)"
    assert c.embed_text.startswith("Rule 25 (international) — Sailing Vessels Underway")
    assert "sidelights" in c.text          # body only, no front-matter
    assert "number:" not in c.text

from pathlib import Path

from vault_search.chunk import chunk_document
from vault_search.models import VaultProfile

PILOT = VaultProfile(
    glob="**/*.md",
    front_matter_fields=["title"],
    chunk_strategy="headings",
    breadcrumb="{title}",
    citation="{title}",
    max_tokens=20,
    overlap_tokens=4,
)

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


def _write(tmp_path, body):
    p = tmp_path / "doc.md"
    p.write_text("---\ntitle: Boundary Pass\n---\n" + body, encoding="utf-8")
    return p


def test_headings_splits_on_headings(tmp_path):
    from vault_search.chunk import chunk_document
    body = "## Approach\nEnter from the south.\n\n## Anchorage\nGood holding in mud.\n"
    chunks = chunk_document(_write(tmp_path, body), tmp_path, PILOT)
    assert len(chunks) == 2
    assert "Approach" in chunks[0].embed_text
    assert "Anchorage" in chunks[1].embed_text


def test_headings_size_cap_splits_long_section(tmp_path):
    from vault_search.chunk import chunk_document
    words = " ".join(f"w{i}" for i in range(50))   # 50 words, cap is 20
    body = f"## Long\n{words}\n"
    chunks = chunk_document(_write(tmp_path, body), tmp_path, PILOT)
    assert len(chunks) >= 3
    # overlap: last 4 words of chunk 0 reappear at the start of chunk 1's body
    tail = chunks[0].text.split()[-4:]
    assert tail == chunks[1].text.split()[: len(tail)]


def test_chunk_vault_walks_glob():
    from vault_search.chunk import chunk_vault
    chunks = chunk_vault(VAULT, COLREGS)
    assert any(c.doc_path == "rules/rule-25.md" for c in chunks)

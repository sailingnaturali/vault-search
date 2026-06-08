from vault_search.models import Chunk, VaultProfile


def test_chunk_id_is_stable():
    a = Chunk.make(doc_path="rules/r.md", ordinal=0, text="t", embed_text="e",
                   metadata={}, citation="c")
    b = Chunk.make(doc_path="rules/r.md", ordinal=0, text="DIFFERENT", embed_text="x",
                   metadata={}, citation="c")
    # id is derived from (doc_path, ordinal) only -> stable across body edits
    assert a.id == b.id


def test_chunk_id_differs_by_ordinal():
    a = Chunk.make(doc_path="rules/r.md", ordinal=0, text="t", embed_text="e",
                   metadata={}, citation="c")
    b = Chunk.make(doc_path="rules/r.md", ordinal=1, text="t", embed_text="e",
                   metadata={}, citation="c")
    assert a.id != b.id


def test_vault_profile_defaults():
    p = VaultProfile(glob="**/*.md", front_matter_fields=["title"],
                     chunk_strategy="whole_file", breadcrumb="{title}",
                     citation="{title}")
    assert p.max_tokens == 512
    assert p.overlap_tokens == 64

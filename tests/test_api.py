def test_top_level_reexports():
    import vault_search as vs
    for name in ["VaultProfile", "Chunk", "SearchHit", "Embedder",
                 "Index", "build_index", "chunk_vault", "search"]:
        assert hasattr(vs, name), f"vault_search.{name} missing"

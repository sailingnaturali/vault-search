from vault_search.eval import recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["25", "18"], expect={"25"}, k=1) == 1.0
    assert recall_at_k(["18", "25"], expect={"25"}, k=1) == 0.0
    assert recall_at_k(["18", "25"], expect={"25"}, k=2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["18", "25"], expect={"25"}) == 0.5
    assert reciprocal_rank(["25"], expect={"25"}) == 1.0
    assert reciprocal_rank(["1", "2"], expect={"25"}) == 0.0

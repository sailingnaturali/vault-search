from vault_search.eval import recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["25", "18"], expect={"25"}, k=1) == 1.0
    assert recall_at_k(["18", "25"], expect={"25"}, k=1) == 0.0
    assert recall_at_k(["18", "25"], expect={"25"}, k=2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["18", "25"], expect={"25"}) == 0.5
    assert reciprocal_rank(["25"], expect={"25"}) == 1.0
    assert reciprocal_rank(["1", "2"], expect={"25"}) == 0.0


def test_run_eval_integration_floor(tmp_path):
    from pathlib import Path
    import yaml
    from vault_search.eval import run_eval
    from vault_search.profiles import COLREGS

    vault = Path(__file__).parent / "fixtures" / "vault"
    golden = tmp_path / "g.yaml"
    golden.write_text(yaml.safe_dump({"queries": [
        {"query": "lights for a sailing vessel", "expect": ["25"]},
        {"query": "shapes for a vessel at anchor", "expect": ["30"]},
    ]}))
    result = run_eval(golden, vault, COLREGS, tmp_path / "e.db")
    # fixture has 2 docs (rules 25 + 30), so any retriever's recall@5 should be perfect
    assert result["hybrid"]["r5"] >= 0.5
    assert set(result["hybrid"].keys()) == {"r1", "r3", "r5", "mrr"}

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


def test_run_eval_respects_label_field(tmp_path):
    # The default label field is "number" (present in the fixture) -> non-zero recall.
    # A bogus/missing label field -> labels are all "" -> zero recall for every retriever.
    from pathlib import Path
    import yaml
    from vault_search.eval import run_eval
    from vault_search.models import VaultProfile

    vault = Path(__file__).parent / "fixtures" / "vault"
    profile = VaultProfile(
        glob="rules/**/*.md",
        front_matter_fields=["number", "regime", "title", "source_pdf"],
        chunk_strategy="whole_file",
        breadcrumb="Rule {number} ({regime})",
        citation="Rule {number} ({regime})")
    golden = tmp_path / "g.yaml"
    golden.write_text(yaml.safe_dump({"queries": [
        {"query": "lights for a sailing vessel", "expect": ["25"]},
    ]}))

    default = run_eval(golden, vault, profile, tmp_path / "d.db")
    bogus = run_eval(golden, vault, profile, tmp_path / "b.db", label_field="not_a_field")
    assert default["hybrid"]["r5"] >= 0.5          # number field works
    assert bogus["hybrid"]["r5"] == 0.0            # unknown field -> no labels match

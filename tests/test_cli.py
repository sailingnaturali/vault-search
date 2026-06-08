from pathlib import Path

from typer.testing import CliRunner

from vault_search.cli import app

VAULT = Path(__file__).parent / "fixtures" / "vault"
runner = CliRunner()


def test_build_then_query(tmp_path):
    db = tmp_path / "c.db"
    r = runner.invoke(app, ["build", str(VAULT), "--profile", "colregs", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert db.exists()

    r = runner.invoke(app, ["query", "lights under sail", "--db", str(db), "--limit", "2"])
    assert r.exit_code == 0, r.output
    assert "Rule 25 (international)" in r.output


def test_build_unknown_profile_errors(tmp_path):
    r = runner.invoke(app, ["build", str(VAULT), "--profile", "nope",
                            "--db", str(tmp_path / "x.db")])
    assert r.exit_code == 1
    assert "unknown profile" in r.output

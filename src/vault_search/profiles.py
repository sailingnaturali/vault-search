"""Worked-example vault profiles."""

from vault_search.models import VaultProfile

COLREGS = VaultProfile(
    glob="rules/**/*.md",
    front_matter_fields=["number", "regime", "title", "source_pdf"],
    chunk_strategy="whole_file",
    breadcrumb="Rule {number} ({regime}) — {title}",
    citation="Rule {number} ({regime})",
)

PROFILES = {"colregs": COLREGS}

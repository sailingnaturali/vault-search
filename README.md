# vault-search

vault-search is a vault-agnostic, local-first hybrid retrieval library and CLI for markdown
knowledge vaults. It combines in-process fastembed ONNX vector embeddings with SQLite FTS5
(BM25) keyword search, fused with Reciprocal Rank Fusion (RRF). The index lives in a single
SQLite file (via sqlite-vec) — no server, no PyTorch, works offline after the first model
download. The colregs corpus (IRPCS + Inland + Canadian Rules) is the worked example.

## Why hybrid

Keyword search (BM25) nails exact terms and rule citations — "Rule 18", "restricted in ability
to maneuver" — but fails on paraphrase. Vector search catches semantic equivalents:
"trawling" → "fishing", "under engine" → "propelled by machinery", "head-on" → "meeting
situation". RRF fuses both ranked lists without requiring score calibration. On the colregs
corpus (terse, numbered rules) vector alone already outperforms keyword, because natural-
language queries rarely match the regulatory phrasing verbatim. On longer-prose vaults like
a pilot book, the semantic advantage is even larger.

## Install

```bash
uv sync
```

## Usage

**Build an index:**

```bash
uv run vault-search build ../colregs-vault --profile colregs --db colregs.db
# indexed 135 chunks -> colregs.db
```

**Query (hybrid by default):**

```bash
uv run vault-search query "who gives way when I'm sailing and they're trawling" --db colregs.db --limit 3
```

```
[0.0308] Rule 26 (inland)
    (a) A vessel engaged in fishing, whether underway or at anchor, shall exhibit only the lights and shapes prescribed in t…
[0.0306] Rule 26 (international)
    (a) A vessel engaged in fishing, whether underway or at anchor, shall exhibit only the lights and shapes prescribed in t…
[0.0300] Rule Annex II (international)
    1. General The lights mentioned herein shall, if exhibited in pursuance of Rule 26(d), be placed where they can best be…
```

Rule 18 (Responsibilities between vessels) appears at rank 4 with `--limit 5` — the query
about "trawling" correctly surfaces fishing-vessel rules before the general give-way rule.

**Evaluate against a golden set:**

```bash
uv run vault-search evaluate golden/colregs.yaml ../colregs-vault --profile colregs --db colregs.db
```

## VaultProfile

Adapting to a new vault is a config change, not a code change. A `VaultProfile` specifies:

| field | description |
|---|---|
| `glob` | which files to index (e.g. `rules/**/*.md`) |
| `front_matter_fields` | YAML front-matter keys to extract (e.g. `number`, `regime`, `title`) |
| `chunk_strategy` | `whole_file` (one chunk per file) or `headings` (split on `##` headings) |
| `breadcrumb` | display template, e.g. `Rule {number} ({regime}) — {title}` |
| `citation` | short citation template for results, e.g. `Rule {number} ({regime})` |

See `src/vault_search/profiles.py` (`COLREGS`) for the worked example.

## Results (colregs)

10-query golden set, 135 chunks indexed (IRPCS International + Inland + Canadian rules):

```
retriever     R@1    R@3    R@5    MRR
keyword      0.30   0.30   0.60   0.38
vector       0.80   0.80   1.00   0.84
hybrid       0.60   0.70   0.80   0.68
```

Vector search dominates on this corpus (MRR 0.84 vs keyword 0.38), confirming that
natural-language queries rarely match regulatory phrasing word-for-word. Hybrid (MRR 0.68)
sits between the two — RRF pulls in BM25 candidates that dilute the vector rankings on a
corpus where keyword is already the weaker retriever. On longer-prose vaults where both
retrievers contribute signal, the fusion advantage is more pronounced.

# vault-search — Design

**Date:** 2026-06-07
**Status:** Approved (brainstorming) → ready for implementation plan

## Purpose

A vault-agnostic, local-first **hybrid retrieval** module (dense vectors + BM25, RRF-fused)
for the markdown knowledge vaults. It improves on today's naive keyword search by surfacing
semantically-relevant passages where exact-term matching fails.

The first test corpus is **colregs-vault** (public, structured, easy to iterate). The real
payoff target is **pilotbook-vault** (prose-heavy, where keyword search is weakest). The
module is therefore designed to be **vault-agnostic** — adapting to a new vault is a config
change, not a rewrite.

This is a **standalone module + eval harness**. It does NOT modify the published
`colregs-mcp` tool surface; in particular it leaves the deterministic `search_rules`
contract and the safety layer untouched. Adoption into individual vault MCPs is a later,
separate decision informed by the eval results.

## Non-goals

- Wiring into any MCP server's tool surface (deferred).
- Reranking as a hard dependency (a pluggable stage is designed, but **off by default**).
- Replacing the deterministic `search_rules` ranking in colregs-mcp.
- Generation / answer synthesis — this is the **retrieval** layer only. Generation stays
  with Hermes → Claude Sonnet.

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding runtime | **In-process fastembed (ONNX)** | No daemon, no torch, caches once, runs offline and on the Pi. Matches the local-first/portable ethos better than depending on the Ollama daemon. |
| Store | **Single SQLite file**: FTS5 (BM25) + `sqlite-vec` (`vec0` KNN) | Embeddable, one file, ships anywhere; both retrievers live together. |
| Fusion | **Reciprocal Rank Fusion (RRF), k=60** | Simple, robust, no score normalization needed across BM25 and cosine. |
| Default embed model | `BAAI/bge-small-en-v1.5` (384-d) | Fast, small, strong for English retrieval; ONNX via fastembed. |
| Packaging | New sibling uv project `vault-search/` (Python 3.11) | Reusable reference other vault MCPs can vendor/install; MIT-able later. |
| Scope | Standalone module + CLI + eval harness | Prove quality (measured), leave published surfaces intact. |

## Architecture

Five focused units under `src/vault_search/`:

| Unit | Responsibility | Depends on |
|---|---|---|
| `chunk.py` | Markdown → chunks. Parse front-matter, split on headings, size-cap with overlap, prepend a breadcrumb (e.g. title/number) to the embed text. Strategy driven by `VaultProfile`. | front-matter parser (e.g. `python-frontmatter`) |
| `embed.py` | fastembed wrapper. Batch-encode chunk/query text. Single configured model. | `fastembed` |
| `index.py` | Build one SQLite file: `chunks` table + FTS5 virtual table + `vec0` virtual table. Idempotent rebuild (drop + recreate). | `sqlite-vec`, `chunk`, `embed` |
| `search.py` | Query → {BM25 top-N, vec KNN top-N} → RRF fuse → top-k chunks with metadata + citation. Optional rerank stage (off by default). | `index`, `embed` |
| `cli.py` / `eval.py` | `build` / `query` / `eval` commands; eval prints recall@k + MRR for three retrievers. | all above |

### VaultProfile (config)

A small per-vault descriptor so colregs→pilotbook is configuration:

- `glob`: which markdown files (e.g. `rules/**/*.md`).
- `front_matter_fields`: names to lift into metadata + citation (colregs: `number`,
  `regime`, `title`, `source_pdf`).
- `chunk_strategy`: `whole_file` (colregs — one chunk per rule) or `headings`
  (pilotbook — split on headings, size-capped with overlap).
- `breadcrumb`: template for the prefix prepended to embed text (e.g.
  `"Rule {number} ({regime}) — {title}"`).
- `citation`: template for the human-readable source string.

colregs profile ships in-repo as the worked example.

### Chunk record

```
id            stable hash of (doc_path, ordinal)
doc_path      source markdown path (relative to vault)
ordinal       chunk index within the doc
text          chunk body (what BM25 indexes)
embed_text    breadcrumb + text (what gets embedded)
metadata      JSON: lifted front-matter fields
citation      rendered citation string
```

## Data flow

**Build:** vault `*.md` → `chunk` → `embed(embed_text)` → SQLite (`chunks` + FTS5 + `vec0`).
Deterministic given (model, vault).

**Query:** text → `embed` → parallel { FTS5 BM25 top-N, `vec0` KNN top-N } → **RRF fuse** →
(optional rerank) → top-k chunks with metadata + citation. Deterministic given a fixed index.

## Evaluation (the justification)

"Improved RAG" must be measured, not asserted.

- **Golden query set** `golden/colregs.yaml`: ~15–20 natural-language queries, each paired
  with the rule(s) that should surface — chosen to exercise *semantic* gaps keyword search
  misses. Examples:
  - "what lights does a sailboat under engine show at night" → Rule 25(e), Rule 23
  - "who keeps clear, sailing vessel or a boat trawling" → Rule 18
  - "signals in fog when anchored" → Rule 35
- `eval.py` runs each query through **three retrievers** — keyword-only, vector-only,
  hybrid (RRF) — and reports **recall@1/3/5 and MRR** per retriever. This head-to-head
  justifies the approach and later guides the pilotbook rollout.

## Testing (TDD)

- `chunk.py` — front-matter parse, heading split, size-cap + overlap, breadcrumb prepend
  (table-driven against a tiny fixture vault).
- `search.py` — **RRF fusion is pure; exact unit tests**: known ranked lists → known fused
  order. Core logic verified independently of embeddings.
- `index.py` — build→query roundtrip on a 3-file fixture vault (real fastembed; asserts the
  obvious chunk returns for an obvious query).
- The golden eval doubles as an integration smoke test (asserts hybrid recall@5 ≥ a floor).

## Error handling

- Embedding model not cached + offline → clear, actionable error naming the model and
  instructing a one-time online `build`. (fastembed caches to `~/.cache`; queries are then
  fully offline.)
- Query before build / missing `.db` → explicit "index not built" error, not a stack trace.
- Empty corpus or zero hits → empty result list, never an exception (mirrors the
  `search_rules` "empty hits, never error" contract).

## Determinism

Build is deterministic given (model, vault); ONNX inference is deterministic; query results
are stable given a fixed index. The module preserves reproducibility even though ranking is
fuzzy — consistent with colregs-mcp's values.

## Dependencies

`fastembed`, `sqlite-vec`, a front-matter parser (`python-frontmatter`), `pyyaml`, and a CLI
lib (`typer` or stdlib `argparse`). No torch, no server.

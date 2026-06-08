# vault-search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vault-agnostic, local-first hybrid retrieval module (dense vectors + BM25, RRF-fused) with a measurable eval harness, proven against colregs-vault.

**Architecture:** A uv Python package. A markdown-aware chunker (driven by a per-vault `VaultProfile`) feeds an in-process fastembed (ONNX) embedder. Chunks + BM25 (FTS5) + vectors (sqlite-vec `vec0`) live in one SQLite file. Queries run BM25 and KNN in parallel, fuse with Reciprocal Rank Fusion, and return cited chunks. An eval harness scores keyword-only vs vector-only vs hybrid on a golden query set.

**Tech Stack:** Python 3.11, uv, fastembed, sqlite-vec, python-frontmatter, pyyaml, typer, pytest.

---

## File Structure

```
vault-search/
  pyproject.toml
  src/vault_search/
    __init__.py
    models.py        # Chunk, VaultProfile, SearchHit dataclasses
    chunk.py         # markdown -> Chunk list (whole_file + headings strategies)
    embed.py         # Embedder: fastembed wrapper
    index.py         # build_index + Index (FTS5 + vec0 over one SQLite file)
    search.py        # rrf_fuse + search()
    profiles.py      # COLREGS profile (worked example)
    cli.py           # typer app: build / query / eval
    eval.py          # recall@k + MRR for three retrievers
  golden/
    colregs.yaml     # golden query set
  tests/
    fixtures/vault/  # tiny 3-file markdown fixture
    test_chunk.py
    test_embed.py
    test_index.py
    test_search.py
    test_eval.py
```

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/vault_search/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "vault-search"
version = "0.1.0"
description = "Vault-agnostic local-first hybrid retrieval for markdown knowledge vaults"
requires-python = ">=3.11"
dependencies = [
    "fastembed>=0.3",
    "sqlite-vec>=0.1.6",
    "python-frontmatter>=1.1",
    "pyyaml>=6",
    "typer>=0.12",
]

[project.scripts]
vault-search = "vault_search.cli:app"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the package marker**

`src/vault_search/__init__.py`:

```python
"""vault-search: vault-agnostic local-first hybrid retrieval."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write a smoke test**

`tests/test_smoke.py`:

```python
def test_package_imports():
    import vault_search

    assert vault_search.__version__ == "0.1.0"
```

- [ ] **Step 4: Sync deps and run the smoke test**

Run: `uv sync && uv run pytest tests/test_smoke.py -v`
Expected: PASS (1 passed). First sync downloads fastembed/sqlite-vec wheels.

- [ ] **Step 5: Add `.gitignore` and commit**

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.db
```

```bash
git add pyproject.toml uv.lock src/vault_search/__init__.py tests/test_smoke.py .gitignore
git commit -m "chore: scaffold vault-search package"
```

---

## Task 2: Data models

**Files:**
- Create: `src/vault_search/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_search.models'`

- [ ] **Step 3: Write the models**

`src/vault_search/models.py`:

```python
"""Core dataclasses: Chunk, VaultProfile, SearchHit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    doc_path: str
    ordinal: int
    text: str
    embed_text: str
    metadata: dict
    citation: str

    @staticmethod
    def make(doc_path: str, ordinal: int, text: str, embed_text: str,
             metadata: dict, citation: str) -> "Chunk":
        raw = f"{doc_path}#{ordinal}".encode("utf-8")
        cid = hashlib.sha1(raw).hexdigest()[:16]
        return Chunk(id=cid, doc_path=doc_path, ordinal=ordinal, text=text,
                     embed_text=embed_text, metadata=metadata, citation=citation)


@dataclass
class VaultProfile:
    glob: str
    front_matter_fields: list[str]
    chunk_strategy: str           # "whole_file" | "headings"
    breadcrumb: str               # str.format template over front-matter fields
    citation: str                 # str.format template over front-matter fields
    max_tokens: int = 512         # word-count proxy for "headings" strategy
    overlap_tokens: int = 64


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    retriever: str = "hybrid"     # "bm25" | "vector" | "hybrid"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vault_search/models.py tests/test_models.py
git commit -m "feat: core dataclasses (Chunk, VaultProfile, SearchHit)"
```

---

## Task 3: Chunker — whole_file strategy

**Files:**
- Create: `src/vault_search/chunk.py`
- Create: `tests/fixtures/vault/rules/rule-25.md`
- Test: `tests/test_chunk.py`

- [ ] **Step 1: Create one fixture rule file**

`tests/fixtures/vault/rules/rule-25.md`:

```markdown
---
number: '25'
regime: international
title: Sailing Vessels Underway
source_pdf: navrules.pdf p.69
---
A sailing vessel underway shall exhibit sidelights and a sternlight.
A vessel proceeding under sail when also propelled by machinery shall
exhibit forward a conical shape, apex downwards.
```

- [ ] **Step 2: Write the failing test**

`tests/test_chunk.py`:

```python
from pathlib import Path

from vault_search.chunk import chunk_document
from vault_search.models import VaultProfile

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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_search.chunk'`

- [ ] **Step 4: Write the chunker (whole_file branch)**

`src/vault_search/chunk.py`:

```python
"""Markdown -> Chunk list. Strategy driven by VaultProfile."""

from __future__ import annotations

from pathlib import Path

import frontmatter

from vault_search.models import Chunk, VaultProfile


def _meta(post: frontmatter.Post, profile: VaultProfile) -> dict:
    return {f: str(post.get(f, "")) for f in profile.front_matter_fields}


def _render(template: str, meta: dict) -> str:
    try:
        return template.format(**meta)
    except (KeyError, IndexError):
        return template


def chunk_document(path: Path, vault_root: Path, profile: VaultProfile) -> list[Chunk]:
    post = frontmatter.load(path)
    meta = _meta(post, profile)
    rel = str(path.relative_to(vault_root))
    citation = _render(profile.citation, meta)
    breadcrumb = _render(profile.breadcrumb, meta)
    body = post.content.strip()

    if profile.chunk_strategy == "whole_file":
        return [Chunk.make(doc_path=rel, ordinal=0, text=body,
                           embed_text=f"{breadcrumb}\n\n{body}",
                           metadata=meta, citation=citation)]
    raise ValueError(f"unknown chunk_strategy: {profile.chunk_strategy}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/vault_search/chunk.py tests/test_chunk.py tests/fixtures/vault/rules/rule-25.md
git commit -m "feat: markdown chunker (whole_file strategy)"
```

---

## Task 4: Chunker — headings strategy with size cap + overlap

**Files:**
- Modify: `src/vault_search/chunk.py`
- Modify: `tests/test_chunk.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/test_chunk.py`:

```python
PILOT = VaultProfile(
    glob="**/*.md",
    front_matter_fields=["title"],
    chunk_strategy="headings",
    breadcrumb="{title}",
    citation="{title}",
    max_tokens=20,
    overlap_tokens=4,
)


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
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: FAIL — `whole_file`-only chunker raises `ValueError: unknown chunk_strategy: headings`

- [ ] **Step 3: Implement the headings strategy**

In `src/vault_search/chunk.py`, replace the `raise ValueError(...)` line with a dispatch to a new helper and add the helper:

```python
    if profile.chunk_strategy == "whole_file":
        return [Chunk.make(doc_path=rel, ordinal=0, text=body,
                           embed_text=f"{breadcrumb}\n\n{body}",
                           metadata=meta, citation=citation)]
    if profile.chunk_strategy == "headings":
        return _chunk_headings(body, rel, breadcrumb, citation, meta, profile)
    raise ValueError(f"unknown chunk_strategy: {profile.chunk_strategy}")
```

Add these helpers at the end of the file:

```python
import re

_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)


def _split_sections(body: str) -> list[str]:
    starts = [m.start() for m in _HEADING.finditer(body)]
    if not starts:
        return [body] if body.strip() else []
    if starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [len(body)]
    out = []
    for i in range(len(starts)):
        seg = body[bounds[i]:bounds[i + 1]].strip()
        if seg:
            out.append(seg)
    return out


def _window(section: str, max_tokens: int, overlap: int) -> list[str]:
    words = section.split()
    if len(words) <= max_tokens:
        return [section]
    step = max(1, max_tokens - overlap)
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i + max_tokens]))
        if i + max_tokens >= len(words):
            break
        i += step
    return out


def _chunk_headings(body, rel, breadcrumb, citation, meta, profile) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for section in _split_sections(body):
        for piece in _window(section, profile.max_tokens, profile.overlap_tokens):
            chunks.append(Chunk.make(
                doc_path=rel, ordinal=ordinal, text=piece,
                embed_text=f"{breadcrumb}\n\n{piece}",
                metadata=meta, citation=citation))
            ordinal += 1
    return chunks
```

- [ ] **Step 4: Run to verify all chunk tests pass**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Add `chunk_vault` and a test**

Add to `src/vault_search/chunk.py`:

```python
def chunk_vault(vault_root: Path, profile: VaultProfile) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(vault_root.glob(profile.glob)):
        chunks.extend(chunk_document(path, vault_root, profile))
    return chunks
```

Add to `tests/test_chunk.py`:

```python
def test_chunk_vault_walks_glob():
    from vault_search.chunk import chunk_vault
    chunks = chunk_vault(VAULT, COLREGS)
    assert any(c.doc_path == "rules/rule-25.md" for c in chunks)
```

- [ ] **Step 6: Run, then commit**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: PASS (5 passed)

```bash
git add src/vault_search/chunk.py tests/test_chunk.py
git commit -m "feat: headings chunk strategy with size cap + overlap, chunk_vault"
```

---

## Task 5: Embedder

**Files:**
- Create: `src/vault_search/embed.py`
- Test: `tests/test_embed.py`

- [ ] **Step 1: Write the failing test**

`tests/test_embed.py`:

```python
from vault_search.embed import Embedder


def test_embed_dim_and_determinism():
    emb = Embedder()                       # default bge-small-en-v1.5
    vecs = emb.encode(["sailing vessel lights", "sailing vessel lights"])
    assert len(vecs) == 2
    assert len(vecs[0]) == emb.dim == 384
    # identical text -> identical vector (ONNX is deterministic)
    assert vecs[0] == vecs[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_search.embed'`

- [ ] **Step 3: Write the embedder**

`src/vault_search/embed.py`:

```python
"""In-process embeddings via fastembed (ONNX). No torch, no daemon."""

from __future__ import annotations

from functools import cached_property

from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DIMS = {"BAAI/bge-small-en-v1.5": 384}


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self) -> TextEmbedding:
        return TextEmbedding(model_name=self.model_name)

    @property
    def dim(self) -> int:
        return _DIMS.get(self.model_name, 384)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(list(texts))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embed.py -v`
Expected: PASS (1 passed). First run downloads the ONNX model to `~/.cache/fastembed` (needs network once).

- [ ] **Step 5: Commit**

```bash
git add src/vault_search/embed.py tests/test_embed.py
git commit -m "feat: fastembed Embedder (bge-small-en-v1.5)"
```

---

## Task 6: Index — build + BM25 + KNN over one SQLite file

**Files:**
- Create: `src/vault_search/index.py`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write the failing test**

`tests/test_index.py`:

```python
from pathlib import Path

from vault_search.chunk import chunk_vault
from vault_search.embed import Embedder
from vault_search.index import Index, build_index
from vault_search.models import VaultProfile

VAULT = Path(__file__).parent / "fixtures" / "vault"

COLREGS = VaultProfile(
    glob="rules/**/*.md",
    front_matter_fields=["number", "regime", "title", "source_pdf"],
    chunk_strategy="whole_file",
    breadcrumb="Rule {number} ({regime}) — {title}",
    citation="Rule {number} ({regime})",
)


def test_build_then_bm25_and_knn_find_the_rule(tmp_path):
    db = tmp_path / "colregs.db"
    emb = Embedder()
    build_index(db, chunk_vault(VAULT, COLREGS), emb)

    idx = Index.open(db, emb)
    bm = idx.bm25("sidelights sternlight", n=5)
    assert bm                                  # rowids, best first
    assert idx.get_chunk(bm[0]).metadata["number"] == "25"

    kn = idx.knn(emb.encode(["lights for a boat under sail"])[0], n=5)
    assert kn
    assert idx.get_chunk(kn[0]).metadata["number"] == "25"


def test_rebuild_is_idempotent(tmp_path):
    db = tmp_path / "c.db"
    emb = Embedder()
    chunks = chunk_vault(VAULT, COLREGS)
    build_index(db, chunks, emb)
    build_index(db, chunks, emb)               # second build must not error or double-count
    idx = Index.open(db, emb)
    assert idx.count() == len(chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_search.index'`

- [ ] **Step 3: Write the index**

`src/vault_search/index.py`:

```python
"""One SQLite file holding chunks + FTS5 (BM25) + sqlite-vec (vec0) KNN."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import sqlite_vec

from vault_search.embed import Embedder
from vault_search.models import Chunk

_WORD = re.compile(r"[A-Za-z0-9]+")


def _connect(db_path: Path, dim: int) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            rowid INTEGER PRIMARY KEY,
            id TEXT UNIQUE, doc_path TEXT, ordinal INTEGER,
            text TEXT, embed_text TEXT, metadata TEXT, citation TEXT
        )""")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(text)")
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])")
    return conn


def build_index(db_path: Path, chunks: list[Chunk], embedder: Embedder) -> None:
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()                       # idempotent: rebuild from scratch
    conn = _connect(db_path, embedder.dim)
    vectors = embedder.encode([c.embed_text for c in chunks])
    with conn:
        for rowid, (c, vec) in enumerate(zip(chunks, vectors), start=1):
            conn.execute(
                "INSERT INTO chunks(rowid,id,doc_path,ordinal,text,embed_text,metadata,citation)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (rowid, c.id, c.doc_path, c.ordinal, c.text, c.embed_text,
                 json.dumps(c.metadata), c.citation))
            conn.execute("INSERT INTO fts_chunks(rowid,text) VALUES(?,?)", (rowid, c.text))
            conn.execute("INSERT INTO vec_chunks(rowid,embedding) VALUES(?,?)",
                         (rowid, sqlite_vec.serialize_float32(vec)))
    conn.close()


def _match_query(query: str) -> str:
    terms = _WORD.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in terms)


class Index:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, db_path: Path, embedder: Embedder) -> "Index":
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"index not built: {db_path} — run `vault-search build` first")
        return cls(_connect(Path(db_path), embedder.dim))

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def bm25(self, query: str, n: int) -> list[int]:
        match = _match_query(query)
        if not match:
            return []
        rows = self._conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?"
            " ORDER BY bm25(fts_chunks) LIMIT ?", (match, n)).fetchall()
        return [r[0] for r in rows]

    def knn(self, query_vec: list[float], n: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT rowid FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
            " ORDER BY distance", (sqlite_vec.serialize_float32(query_vec), n)).fetchall()
        return [r[0] for r in rows]

    def get_chunk(self, rowid: int) -> Chunk:
        r = self._conn.execute(
            "SELECT id,doc_path,ordinal,text,embed_text,metadata,citation"
            " FROM chunks WHERE rowid=?", (rowid,)).fetchone()
        return Chunk(id=r[0], doc_path=r[1], ordinal=r[2], text=r[3], embed_text=r[4],
                     metadata=json.loads(r[5]), citation=r[6])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_index.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vault_search/index.py tests/test_index.py
git commit -m "feat: SQLite index with FTS5 BM25 + sqlite-vec KNN"
```

---

## Task 7: Search — RRF fusion (pure) then hybrid query

**Files:**
- Create: `src/vault_search/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing pure-fusion test**

`tests/test_search.py`:

```python
from vault_search.search import rrf_fuse


def test_rrf_fuse_known_order():
    # two rankings of rowids; RRF with k=60 sums 1/(k+rank), rank starts at 1
    bm25 = [1, 2, 3]
    vector = [2, 1, 4]
    fused = rrf_fuse([bm25, vector], k=60)
    ids = [rowid for rowid, _ in fused]
    # 1: 1/61 + 1/62 ; 2: 1/62 + 1/61  -> tie, broken by smaller rowid -> 1 before 2
    assert ids[0] == 1
    assert ids[1] == 2
    assert set(ids[2:]) == {3, 4}


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([[5, 9], [5, 7]], k=60)   # 5 appears in both, top of each
    assert fused[0][0] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_search.search'`

- [ ] **Step 3: Write `rrf_fuse`**

`src/vault_search/search.py`:

```python
"""Reciprocal Rank Fusion + the hybrid search entry point."""

from __future__ import annotations

from vault_search.embed import Embedder
from vault_search.index import Index
from vault_search.models import SearchHit


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, rowid in enumerate(ranking, start=1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank)
    # sort by score desc, ties broken by smaller rowid for determinism
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the hybrid `search()` test**

Append to `tests/test_search.py`:

```python
from pathlib import Path

from vault_search.chunk import chunk_vault
from vault_search.embed import Embedder
from vault_search.index import Index, build_index
from vault_search.models import VaultProfile
from vault_search.search import search

VAULT = Path(__file__).parent / "fixtures" / "vault"
COLREGS = VaultProfile(
    glob="rules/**/*.md", front_matter_fields=["number", "regime", "title", "source_pdf"],
    chunk_strategy="whole_file", breadcrumb="Rule {number} ({regime}) — {title}",
    citation="Rule {number} ({regime})")


def test_search_returns_cited_hits(tmp_path):
    db = tmp_path / "s.db"
    emb = Embedder()
    build_index(db, chunk_vault(VAULT, COLREGS), emb)
    idx = Index.open(db, emb)
    hits = search(idx, emb, "lights for a boat under sail", limit=3)
    assert hits
    assert isinstance(hits[0].score, float)
    assert hits[0].chunk.citation == "Rule 25 (international)"
```

- [ ] **Step 6: Implement `search()`**

Append to `src/vault_search/search.py`:

```python
def search(index: Index, embedder: Embedder, query: str, limit: int = 5,
           k: int = 60, pool: int = 20) -> list[SearchHit]:
    bm25 = index.bm25(query, n=pool)
    vector = index.knn(embedder.encode([query])[0], n=pool)
    fused = rrf_fuse([bm25, vector], k=k)
    return [SearchHit(chunk=index.get_chunk(rowid), score=score, retriever="hybrid")
            for rowid, score in fused[:limit]]
```

- [ ] **Step 7: Run, then commit**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS (3 passed)

```bash
git add src/vault_search/search.py tests/test_search.py
git commit -m "feat: RRF fusion + hybrid search()"
```

---

## Task 8: colregs profile + CLI

**Files:**
- Create: `src/vault_search/profiles.py`
- Create: `src/vault_search/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the colregs profile**

`src/vault_search/profiles.py`:

```python
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
```

- [ ] **Step 2: Write the failing CLI test**

`tests/test_cli.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_search.cli'`

- [ ] **Step 4: Write the CLI**

`src/vault_search/cli.py`:

```python
"""vault-search CLI: build / query / eval."""

from __future__ import annotations

from pathlib import Path

import typer

from vault_search.chunk import chunk_vault
from vault_search.embed import Embedder
from vault_search.index import Index, build_index
from vault_search.profiles import PROFILES
from vault_search.search import search

app = typer.Typer(help="Local hybrid retrieval over markdown vaults.")


@app.command()
def build(vault: str, profile: str = "colregs", db: str = "vault.db") -> None:
    """Chunk + embed a vault into a SQLite index."""
    prof = PROFILES[profile]
    chunks = chunk_vault(Path(vault), prof)
    build_index(Path(db), chunks, Embedder())
    typer.echo(f"indexed {len(chunks)} chunks -> {db}")


@app.command()
def query(text: str, db: str = "vault.db", limit: int = 5) -> None:
    """Run a hybrid query against a built index."""
    emb = Embedder()
    idx = Index.open(Path(db), emb)
    for hit in search(idx, emb, text, limit=limit):
        typer.echo(f"[{hit.score:.4f}] {hit.chunk.citation}")
        typer.echo(f"    {hit.chunk.text[:120].strip()}…")


@app.command()
def evaluate(golden: str, vault: str, profile: str = "colregs",
             db: str = "vault.db") -> None:
    """Score keyword vs vector vs hybrid on a golden query set."""
    from vault_search.eval import run_eval
    run_eval(Path(golden), Path(vault), PROFILES[profile], Path(db))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add src/vault_search/profiles.py src/vault_search/cli.py tests/test_cli.py
git commit -m "feat: colregs profile and build/query CLI"
```

---

## Task 9: Eval harness (the justification)

**Files:**
- Create: `src/vault_search/eval.py`
- Create: `golden/colregs.yaml`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the golden query set**

`golden/colregs.yaml`. Each entry: a natural-language query and the rule number(s) that should surface. `expect` matches against each hit chunk's `metadata["number"]`.

```yaml
queries:
  - query: what lights does a sailboat under engine show at night
    expect: ["25"]
  - query: who keeps clear, a sailing vessel or a boat trawling
    expect: ["18"]
  - query: sound signals in fog when at anchor
    expect: ["35"]
  - query: how do I overtake another boat safely
    expect: ["13"]
  - query: lights for a vessel not under command
    expect: ["27"]
  - query: when two power boats meet head on who turns
    expect: ["14"]
  - query: crossing situation give way vessel
    expect: ["15"]
  - query: shapes shown by a vessel restricted in ability to maneuver
    expect: ["27"]
  - query: anchor ball day shape
    expect: ["30"]
  - query: lookout requirement
    expect: ["5"]
```

- [ ] **Step 2: Write the failing test**

`tests/test_eval.py`:

```python
from vault_search.eval import recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["25", "18"], expect={"25"}, k=1) == 1.0
    assert recall_at_k(["18", "25"], expect={"25"}, k=1) == 0.0
    assert recall_at_k(["18", "25"], expect={"25"}, k=2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["18", "25"], expect={"25"}) == 0.5
    assert reciprocal_rank(["25"], expect={"25"}) == 1.0
    assert reciprocal_rank(["1", "2"], expect={"25"}) == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_search.eval'`

- [ ] **Step 4: Write the eval module**

`src/vault_search/eval.py`:

```python
"""Score keyword-only / vector-only / hybrid retrievers on a golden set."""

from __future__ import annotations

from pathlib import Path

import yaml

from vault_search.chunk import chunk_vault
from vault_search.embed import Embedder
from vault_search.index import Index, build_index
from vault_search.models import VaultProfile
from vault_search.search import rrf_fuse


def recall_at_k(labels: list[str], expect: set[str], k: int) -> float:
    return 1.0 if expect & set(labels[:k]) else 0.0


def reciprocal_rank(labels: list[str], expect: set[str]) -> float:
    for i, label in enumerate(labels, start=1):
        if label in expect:
            return 1.0 / i
    return 0.0


def _labels(index: Index, rowids: list[int]) -> list[str]:
    return [index.get_chunk(r).metadata.get("number", "") for r in rowids]


def _retrievers(index: Index, emb: Embedder, query: str, pool: int = 20):
    bm = index.bm25(query, n=pool)
    vec = index.knn(emb.encode([query])[0], n=pool)
    hybrid = [rid for rid, _ in rrf_fuse([bm, vec])]
    return {"keyword": bm, "vector": vec, "hybrid": hybrid}


def run_eval(golden: Path, vault: Path, profile: VaultProfile, db: Path) -> dict:
    emb = Embedder()
    build_index(db, chunk_vault(vault, profile), emb)
    index = Index.open(db, emb)
    queries = yaml.safe_load(golden.read_text())["queries"]

    names = ["keyword", "vector", "hybrid"]
    agg = {n: {"r1": 0.0, "r3": 0.0, "r5": 0.0, "mrr": 0.0} for n in names}
    for q in queries:
        expect = set(str(x) for x in q["expect"])
        rets = _retrievers(index, emb, q["query"])
        for n in names:
            labels = _labels(index, rets[n])
            agg[n]["r1"] += recall_at_k(labels, expect, 1)
            agg[n]["r3"] += recall_at_k(labels, expect, 3)
            agg[n]["r5"] += recall_at_k(labels, expect, 5)
            agg[n]["mrr"] += reciprocal_rank(labels, expect)

    total = len(queries)
    print(f"\n{'retriever':<10} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
    for n in names:
        m = {k: v / total for k, v in agg[n].items()}
        print(f"{n:<10} {m['r1']:>6.2f} {m['r3']:>6.2f} {m['r5']:>6.2f} {m['mrr']:>6.2f}")
    return agg
```

- [ ] **Step 5: Run unit test to verify it passes**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/vault_search/eval.py golden/colregs.yaml tests/test_eval.py
git commit -m "feat: eval harness (recall@k + MRR) and colregs golden set"
```

---

## Task 10: End-to-end run against the real colregs-vault

**Files:**
- Create: `README.md`

- [ ] **Step 1: Build the index against the real vault**

Run: `uv run vault-search build ../colregs-vault --profile colregs --db colregs.db`
Expected: `indexed 135 chunks -> colregs.db` (count may differ as the vault evolves)

- [ ] **Step 2: Spot-check a semantic query keyword search would miss**

Run: `uv run vault-search query "who gives way when I'm sailing and they're trawling" --db colregs.db --limit 3`
Expected: Rule 18 appears in the top hits with a citation line.

- [ ] **Step 3: Run the head-to-head eval**

Run: `uv run vault-search evaluate golden/colregs.yaml ../colregs-vault --profile colregs --db colregs.db`
Expected: a table printing R@1/R@3/R@5/MRR for keyword, vector, and hybrid. **Success criterion:** hybrid MRR ≥ keyword MRR (hybrid should match or beat keyword; the win is expected to be larger on prose vaults like pilotbook).

- [ ] **Step 4: Write the README**

`README.md` documents: what it is (vault-agnostic local hybrid retrieval), install (`uv sync`), the three CLI commands with the examples above, the `VaultProfile` config, and a one-line note that the colregs eval is the worked example and pilotbook is the next target. Keep the "why hybrid" rationale ≤8 lines.

- [ ] **Step 5: Record the eval numbers and commit**

Paste the eval table from Step 3 into the README under a "Results (colregs)" heading, then:

```bash
git add README.md
git commit -m "docs: README with CLI usage and colregs eval results"
```

- [ ] **Step 6: Push**

```bash
git push
```

---

## Notes for the implementer

- **First `uv sync` and first `Embedder()` need network** (wheels + the ONNX model download to `~/.cache/fastembed`). After that, build and query are fully offline.
- **`*.db` is gitignored** — indexes are build artifacts, never committed.
- **fastembed model downloads** can take a minute on first use; the embed test will appear to hang briefly the first time.
- The colregs vault lives at `../colregs-vault` relative to this repo on the dev machine.

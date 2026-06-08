"""One SQLite file holding chunks + FTS5 (BM25) + sqlite-vec (vec0) KNN."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import sqlite_vec

from vault_search.models import Chunk

_WORD = re.compile(r"[A-Za-z0-9]+")


def _open_conn(db_path: Path) -> sqlite3.Connection:
    """Open a connection and load the sqlite_vec extension. No DDL."""
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _create_schema(conn: sqlite3.Connection, dim: int) -> None:
    # Invariant: chunks, fts_chunks, and vec_chunks are kept rowid-aligned —
    # build_index inserts into all three tables under the same rowid.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            rowid INTEGER PRIMARY KEY,
            id TEXT UNIQUE, doc_path TEXT, ordinal INTEGER,
            text TEXT, embed_text TEXT, metadata TEXT, citation TEXT
        )""")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(text)")
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])")


def build_index(db_path: Path, chunks: list[Chunk], embedder: Embedder) -> None:
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()                       # idempotent: rebuild from scratch
    conn = _open_conn(db_path)
    try:
        _create_schema(conn, embedder.dim)
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
    finally:
        conn.close()


def _match_query(query: str) -> str:
    terms = _WORD.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in terms)


class Index:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, db_path: Path) -> "Index":
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"index not built: {db_path} — run `vault-search build` first")
        return cls(_open_conn(Path(db_path)))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

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
        if r is None:
            raise KeyError(rowid)
        return Chunk(id=r[0], doc_path=r[1], ordinal=r[2], text=r[3], embed_text=r[4],
                     metadata=json.loads(r[5]), citation=r[6])

"""Temporary per-namespace SQLite storage for parsed CSV and Excel tables."""

import contextlib
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from app.contracts.ingestion import ParsedTable, TextChunk

_NS_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_QUOTED_LITERAL_RE = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
_FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _make_schema_chunk_id(document_id: str, table_name: str) -> str:
    raw = f"{document_id}:schema:{table_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _validate_readonly_sql(sql: str) -> str:
    """Ensure SQL query is a single read-only SELECT or WITH statement."""
    cleaned = sql.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        raise ValueError("SQL query must not be empty.")

    without_literals = _QUOTED_LITERAL_RE.sub(" ", cleaned)
    if ";" in without_literals:
        raise ValueError("Multiple SQL statements are not allowed.")
    if "--" in without_literals or "/*" in without_literals:
        raise ValueError("SQL comments are not allowed in generated queries.")

    first_token = without_literals.lstrip().split(maxsplit=1)[0].lower()
    if first_token not in {"select", "with"}:
        raise ValueError("Only read-only SELECT or WITH queries are permitted.")
    if _FORBIDDEN_SQL_RE.search(without_literals):
        raise ValueError("Write or schema-mutating SQL keywords are not permitted.")
    return cleaned


class LocalSQLiteTableStore:
    """Persists parsed CSV/Excel tables locally in per-namespace SQLite databases."""

    def __init__(self, base_dir: str | Path = "./data/sqlite") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _safe_db_path(self, namespace: str) -> Path:
        safe_ns = _NS_RE.sub("_", namespace.strip()) or "default"
        candidate = (self.base_dir / f"{safe_ns}.db").resolve()
        if not candidate.is_relative_to(self.base_dir):
            raise ValueError(f"Namespace '{namespace}' escapes SQLite storage root.")
        return candidate

    @staticmethod
    def _ensure_registry(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _rag_table_registry (
                table_name TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source TEXT NOT NULL,
                sheet_name TEXT NOT NULL,
                columns_json TEXT NOT NULL,
                column_types_json TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )

    def save_tables(
        self,
        namespace: str,
        document_id: str,
        tables: list[ParsedTable],
    ) -> list[str]:
        """Create/replace SQLite tables for parsed tabular datasets inside a transaction."""
        if not tables:
            return []

        db_path = self._safe_db_path(namespace)
        loaded_tables: list[str] = []
        now = time.time()

        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            with conn:
                self._ensure_registry(conn)
                for table in tables:
                    quoted_table = _quote_ident(table.table_name)
                    col_defs = ", ".join(
                        f"{_quote_ident(col)} {table.column_types.get(col, 'TEXT')}"
                        for col in table.columns
                    )
                    conn.execute(f"DROP TABLE IF EXISTS {quoted_table}")
                    conn.execute(f"CREATE TABLE {quoted_table} ({col_defs})")

                    if table.rows:
                        placeholders = ", ".join("?" for _ in table.columns)
                        col_list = ", ".join(_quote_ident(col) for col in table.columns)
                        conn.executemany(
                            f"INSERT INTO {quoted_table} ({col_list}) VALUES ({placeholders})",
                            table.rows,
                        )

                    conn.execute(
                        """
                        INSERT INTO _rag_table_registry (
                            table_name, namespace, document_id, source, sheet_name,
                            columns_json, column_types_json, row_count, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(table_name) DO UPDATE SET
                            namespace = excluded.namespace,
                            document_id = excluded.document_id,
                            source = excluded.source,
                            sheet_name = excluded.sheet_name,
                            columns_json = excluded.columns_json,
                            column_types_json = excluded.column_types_json,
                            row_count = excluded.row_count,
                            created_at = excluded.created_at
                        """,
                        (
                            table.table_name,
                            namespace,
                            document_id,
                            table.source,
                            table.sheet_name,
                            json.dumps(table.columns),
                            json.dumps(table.column_types),
                            table.row_count,
                            now,
                        ),
                    )
                    loaded_tables.append(table.table_name)

        return loaded_tables

    def list_tables(self, namespace: str) -> list[dict[str, Any]]:
        """Return metadata for all SQLite tables registered in the given namespace."""
        db_path = self._safe_db_path(namespace)
        if not db_path.exists():
            return []

        with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_rag_table_registry'"
            ).fetchone()
            if exists is None:
                return []
            rows = conn.execute(
                """
                SELECT table_name, namespace, document_id, source, sheet_name,
                       columns_json, column_types_json, row_count, created_at
                FROM _rag_table_registry
                ORDER BY created_at ASC, rowid ASC
                """
            ).fetchall()

        return [
            {
                "table_name": row["table_name"],
                "namespace": row["namespace"],
                "document_id": row["document_id"],
                "source": row["source"],
                "sheet_name": row["sheet_name"],
                "columns": json.loads(row["columns_json"]),
                "column_types": json.loads(row["column_types_json"]),
                "row_count": row["row_count"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def query_readonly(self, namespace: str, sql: str) -> list[dict[str, Any]]:
        """Execute a validated read-only SELECT/WITH SQL query against the namespace DB."""
        safe_sql = _validate_readonly_sql(sql)
        db_path = self._safe_db_path(namespace)
        if not db_path.exists():
            raise FileNotFoundError(f"No tabular SQLite database exists for namespace '{namespace}'.")

        with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            if hasattr(conn, "setconfig") and hasattr(sqlite3, "SQLITE_DBCONFIG_DQS_DML"):
                conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(safe_sql).fetchall()
            return [dict(row) for row in rows]

    def get_table_samples(
        self,
        namespace: str,
        table_name: str,
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        """Fetch up to N sample rows from a registered SQLite table in read-only mode."""
        if limit <= 0:
            return []
        registered = {item["table_name"] for item in self.list_tables(namespace)}
        if table_name not in registered:
            return []
        return self.query_readonly(
            namespace,
            f"SELECT * FROM {_quote_ident(table_name)} LIMIT {int(limit)}",
        )

    def describe_schema(self, namespace: str, sample_rows: int = 2) -> str:
        """Format all registered SQLite tables and sample rows for router/SQL prompts."""
        tables = self.list_tables(namespace)
        if not tables:
            return ""
        blocks: list[str] = []
        for meta in tables:
            cols = ", ".join(
                f'"{col}" {meta["column_types"].get(col, "TEXT")}'
                for col in meta["columns"]
            )
            samples = self.get_table_samples(namespace, meta["table_name"], limit=sample_rows)
            sample_json = json.dumps(samples, ensure_ascii=False)
            blocks.append(
                f'Table "{meta["table_name"]}" (file: {meta["source"]}, sheet: {meta["sheet_name"]}, '
                f'rows: {meta["row_count"]}):\n  Columns: {cols}\n  Sample rows: {sample_json}'
            )
        return "\n".join(blocks)

    def delete_document_tables(self, namespace: str, document_id: str) -> int:
        """Drop temporary SQLite tables belonging to a specific document."""
        db_path = self._safe_db_path(namespace)
        if not db_path.exists():
            return 0

        dropped = 0
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            with conn:
                self._ensure_registry(conn)
                rows = conn.execute(
                    "SELECT table_name FROM _rag_table_registry WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
                for (table_name,) in rows:
                    conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(table_name)}")
                    dropped += 1
                conn.execute(
                    "DELETE FROM _rag_table_registry WHERE document_id = ?",
                    (document_id,),
                )
        return dropped

    def clear_namespace(self, namespace: str) -> None:
        """Delete the entire temporary SQLite DB file for a namespace."""
        db_path = self._safe_db_path(namespace)
        if db_path.exists():
            db_path.unlink()

    @staticmethod
    def build_schema_chunks(
        document_id: str,
        tables: list[ParsedTable],
        max_sample_rows: int = 3,
    ) -> list[TextChunk]:
        """Create schema summary chunks for vector indexing so retrieval discovers SQLite tables."""
        chunks: list[TextChunk] = []
        for idx, table in enumerate(tables):
            col_summary = ", ".join(
                f"{col} ({table.column_types.get(col, 'TEXT')})"
                for col in table.columns
            )
            sample_dicts = [
                dict(zip(table.columns, row, strict=True))
                for row in table.rows[:max_sample_rows]
            ]
            sample_json = json.dumps(sample_dicts, ensure_ascii=False)
            summary_text = (
                f"Tabular SQLite Table '{table.table_name}' from file '{table.source}' "
                f"(sheet '{table.sheet_name}', rows: {table.row_count}). "
                f"Columns: {col_summary}. Sample rows: {sample_json}"
            )
            metadata = {
                "record_type": "tabular_schema",
                "document_id": document_id,
                "source": table.source,
                "sheet_name": table.sheet_name,
                "table_name": table.table_name,
                "row_count": table.row_count,
                "chunk_index": idx,
                "text": summary_text,
            }
            chunks.append(
                TextChunk(
                    chunk_id=_make_schema_chunk_id(document_id, table.table_name),
                    text=summary_text,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )
        return chunks

"""Tests for CSV/Excel tabular parsing, type inference, and temporary SQLite storage."""

from datetime import date
from io import BytesIO
from pathlib import Path

from openpyxl import Workbook
import pytest

from app.ingestion.parser import (
    UnsupportedDocumentError,
    parse_document_bytes,
    parse_uploaded_document,
)
from app.ingestion.sqlite_store import LocalSQLiteTableStore
from app.ingestion.tabular_parser import parse_csv_bytes, parse_xlsx_bytes


def _make_sample_xlsx_bytes() -> bytes:
    wb = Workbook()
    ws_rev = wb.active
    assert ws_rev is not None
    ws_rev.title = "Q4 Revenue"
    # Include a leading blank row to verify trimming
    ws_rev.append([None, None, None])
    ws_rev.append(["Region Name", "Units Sold", "Revenue", "Order Date"])
    ws_rev.append(["North", 10, 1250.5, date(2026, 10, 1)])
    ws_rev.append(["South", 25, 3000, date(2026, 10, 15)])

    ws_meta = wb.create_sheet("Owners")
    ws_meta.append(["Region Name", "Lead", "Lead"])
    ws_meta.append(["North", "Alice", "Bob"])

    # Empty trailing sheet should be skipped cleanly
    wb.create_sheet("BlankTab")

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_parse_csv_bytes_infers_types_and_preserves_leading_zeros() -> None:
    csv_bytes = (
        "\ufeffAccount ID,Amount,Amount,Flag,Notes\n"
        "00123,10,15.5,true,First row\n"
        "00456,20,24.5,false,Second row\n"
    ).encode("utf-8")

    tables = parse_csv_bytes("Accounts (2026).csv", csv_bytes, document_id="doc-csv-01")
    assert len(tables) == 1
    table = tables[0]
    assert table.table_name.startswith("tbl_accounts_2026_")
    assert table.columns == ["account_id", "amount", "amount_2", "flag", "notes"]
    assert table.column_types == {
        "account_id": "TEXT",
        "amount": "INTEGER",
        "amount_2": "REAL",
        "flag": "TEXT",
        "notes": "TEXT",
    }
    assert table.row_count == 2
    assert table.rows[0] == ["00123", 10, 15.5, "true", "First row"]


def test_parse_xlsx_bytes_handles_multiple_sheets_and_skips_blank_tabs() -> None:
    xlsx_bytes = _make_sample_xlsx_bytes()
    tables = parse_xlsx_bytes("Financials.xlsx", xlsx_bytes, document_id="doc-xl-99")
    assert len(tables) == 2

    rev, owners = tables
    assert rev.sheet_name == "Q4 Revenue"
    assert rev.columns == ["region_name", "units_sold", "revenue", "order_date"]
    assert rev.column_types == {
        "region_name": "TEXT",
        "units_sold": "INTEGER",
        "revenue": "REAL",
        "order_date": "TEXT",
    }
    assert rev.rows[0] == ["North", 10, 1250.5, "2026-10-01"]
    assert rev.rows[1] == ["South", 25, 3000.0, "2026-10-15"]

    assert owners.sheet_name == "Owners"
    assert owners.columns == ["region_name", "lead", "lead_2"]
    assert owners.row_count == 1


def test_parser_dispatch_separates_text_and_tabular_flows() -> None:
    with pytest.raises(UnsupportedDocumentError, match="tabular"):
        parse_document_bytes("data.csv", b"a,b\n1,2\n")

    parsed_md = parse_uploaded_document("d1", "readme.md", b"# Spec\nDetails")
    assert parsed_md.kind == "text"
    assert "Details" in parsed_md.text
    assert parsed_md.tables == []

    parsed_csv = parse_uploaded_document("d2", "metrics.csv", b"col_a,col_b\n1,2\n")
    assert parsed_csv.kind == "tabular"
    assert len(parsed_csv.tables) == 1


def test_sqlite_store_saves_queries_readonly_and_cleans_up(tmp_path: Path) -> None:
    store = LocalSQLiteTableStore(base_dir=tmp_path)
    tables = parse_xlsx_bytes("sales.xlsx", _make_sample_xlsx_bytes(), document_id="doc-sql-1")

    loaded = store.save_tables(namespace="finance", document_id="doc-sql-1", tables=tables)
    assert len(loaded) == 2

    registry = store.list_tables("finance")
    assert [item["sheet_name"] for item in registry] == ["Q4 Revenue", "Owners"]

    rev_table = loaded[0]
    owners_table = loaded[1]
    rows = store.query_readonly(
        "finance",
        (
            f'SELECT r.region_name, r.revenue, o.lead '
            f'FROM "{rev_table}" r JOIN "{owners_table}" o '
            f'ON r.region_name = o.region_name'
        ),
    )
    assert rows == [{"region_name": "North", "revenue": 1250.5, "lead": "Alice"}]

    agg = store.query_readonly("finance", f'SELECT SUM(units_sold) AS total FROM "{rev_table}"')
    assert agg == [{"total": 35}]

    schema_chunks = store.build_schema_chunks(document_id="doc-sql-1", tables=tables)
    assert len(schema_chunks) == 2
    assert schema_chunks[0].metadata["record_type"] == "tabular_schema"
    assert rev_table in schema_chunks[0].text

    with pytest.raises(ValueError, match="SELECT or WITH"):
        store.query_readonly("finance", f'DROP TABLE "{rev_table}"')
    with pytest.raises(ValueError, match="Multiple SQL statements"):
        store.query_readonly("finance", f'SELECT * FROM "{rev_table}"; DROP TABLE "{rev_table}"')
    with pytest.raises(ValueError, match="Write or schema-mutating"):
        store.query_readonly("finance", f'WITH x AS (SELECT 1) DELETE FROM "{rev_table}"')

    dropped = store.delete_document_tables("finance", "doc-sql-1")
    assert dropped == 2
    assert store.list_tables("finance") == []

    store.clear_namespace("finance")
    assert not (tmp_path / "finance.db").exists()

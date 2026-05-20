"""Agent tools. Each function is a unit the LLM can call.

Kept framework-agnostic so we can bind them to LangGraph (or anything else)
later without rewriting the SQL safety logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.db import DUCKDB_PATH, connect, describe_table, list_tables

MAX_ROWS = 100
FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|ATTACH|COPY|PRAGMA|EXPORT)\b",
    re.IGNORECASE,
)


@dataclass
class SQLResult:
    columns: list[str]
    rows: list[tuple]
    row_count: int
    truncated: bool

    def to_markdown(self) -> str:
        df = pd.DataFrame(self.rows, columns=self.columns)
        head = df.to_markdown(index=False)
        footer = f"\n_{self.row_count} row(s) returned"
        if self.truncated:
            footer += f", showing first {MAX_ROWS}"
        footer += "._"
        return head + footer


def _require_db() -> None:
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse not built yet at {DUCKDB_PATH}. "
            "Place the Olist CSVs in data/raw/ and run: python scripts/load_data.py"
        )


def get_schema() -> str:
    """Return a compact text description of all tables and columns."""
    if not DUCKDB_PATH.exists():
        return "(no warehouse yet — run scripts/load_data.py after placing CSVs in data/raw/)"
    with connect(read_only=True) as con:
        tables = list_tables(con)
        if not tables:
            return "(warehouse is empty — run scripts/load_data.py)"
        parts: list[str] = []
        for t in tables:
            cols = describe_table(con, t)
            col_str = ", ".join(f"{name} {dtype}" for name, dtype in cols)
            parts.append(f"{t}({col_str})")
    return "\n".join(parts)


def run_sql(query: str) -> SQLResult:
    """Run a read-only SQL query against the warehouse.

    Raises ValueError if the query contains a write/DDL statement.
    Raises FileNotFoundError if the warehouse hasn't been built yet.
    """
    if FORBIDDEN.search(query):
        raise ValueError(
            "Only read-only SELECT queries are allowed. "
            "Forbidden keywords: INSERT, UPDATE, DELETE, DROP, TRUNCATE, "
            "ALTER, CREATE, ATTACH, COPY, PRAGMA, EXPORT."
        )
    _require_db()
    with connect(read_only=True) as con:
        cur = con.execute(query)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    row_count = len(rows)
    truncated = row_count > MAX_ROWS
    return SQLResult(
        columns=columns,
        rows=rows[:MAX_ROWS],
        row_count=row_count,
        truncated=truncated,
    )

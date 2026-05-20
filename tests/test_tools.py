"""Smoke tests for the SQL safety guard. No data required."""

import pytest

from src.tools import FORBIDDEN, run_sql


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE orders",
        "DELETE FROM orders WHERE 1=1",
        "INSERT INTO orders VALUES (1)",
        "UPDATE orders SET id=1",
        "ALTER TABLE orders ADD COLUMN x INT",
        "CREATE TABLE bad (x INT)",
        "PRAGMA threads=4",
    ],
)
def test_forbidden_keywords_match(query: str) -> None:
    assert FORBIDDEN.search(query) is not None


def test_run_sql_rejects_write() -> None:
    with pytest.raises(ValueError, match="read-only"):
        run_sql("DROP TABLE orders")

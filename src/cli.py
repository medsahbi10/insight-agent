"""Manual SQL runner — lets us verify the data + tools work before adding the LLM.

Usage:
    python -m src.cli --schema
    python -m src.cli "SELECT COUNT(*) FROM orders"
"""

from __future__ import annotations

import argparse
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from src.tools import get_schema, run_sql  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SQL against the Olist warehouse.")
    parser.add_argument("query", nargs="?", help="SQL query to execute")
    parser.add_argument("--schema", action="store_true", help="Print full schema and exit")
    args = parser.parse_args()

    if args.schema:
        print(get_schema())
        return 0

    if not args.query:
        parser.print_help()
        return 1

    try:
        result = run_sql(args.query)
    except ValueError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"SQL error: {exc}", file=sys.stderr)
        return 3

    print(result.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

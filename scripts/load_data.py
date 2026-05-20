"""Load Olist CSVs from data/raw/ into DuckDB.

Expects the 9 Kaggle CSVs in data/raw/. Each CSV becomes one DuckDB table
named after the file (without the `olist_` prefix and `_dataset` suffix).

    olist_orders_dataset.csv               -> orders
    olist_order_items_dataset.csv          -> order_items
    olist_order_payments_dataset.csv       -> order_payments
    olist_order_reviews_dataset.csv        -> order_reviews
    olist_products_dataset.csv             -> products
    olist_customers_dataset.csv            -> customers
    olist_sellers_dataset.csv              -> sellers
    olist_geolocation_dataset.csv          -> geolocation
    product_category_name_translation.csv  -> product_category_translation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import PROJECT_ROOT, connect  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"

TABLE_MAP = {
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_products_dataset.csv": "products",
    "olist_customers_dataset.csv": "customers",
    "olist_sellers_dataset.csv": "sellers",
    "olist_geolocation_dataset.csv": "geolocation",
    "product_category_name_translation.csv": "product_category_translation",
}


def main() -> int:
    missing = [name for name in TABLE_MAP if not (RAW / name).exists()]
    if missing:
        print(
            "Missing CSVs in data/raw/:\n  - "
            + "\n  - ".join(missing)
            + "\n\nDownload the Olist dataset from "
            "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
            "and unzip its CSVs into data/raw/.",
            file=sys.stderr,
        )
        return 1

    with connect(read_only=False) as con:
        for csv_name, table in TABLE_MAP.items():
            path = RAW / csv_name
            print(f"Loading {csv_name} -> {table}")
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
                [str(path)],
            )
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {count:,} rows")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

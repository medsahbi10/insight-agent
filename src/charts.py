"""Chart rendering from SQL queries. Saves PNG to charts/.

Kept agent-agnostic — the LangChain tool wrapper lives in src/agent.py.
"""

from __future__ import annotations

import uuid
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # headless backend, no GUI required
import matplotlib.pyplot as plt  # noqa: E402

from src.db import PROJECT_ROOT  # noqa: E402
from src.tools import run_sql  # noqa: E402

CHARTS_DIR = PROJECT_ROOT / "charts"

ChartKind = Literal["bar", "line", "pie", "scatter"]


def render_chart(
    sql: str,
    kind: ChartKind = "bar",
    title: str = "",
    x_col: str | None = None,
    y_col: str | None = None,
) -> dict:
    """Run the SQL, render a chart, save PNG. Return metadata about the chart."""
    CHARTS_DIR.mkdir(exist_ok=True)

    result = run_sql(sql)  # may raise ValueError if SQL is forbidden
    cols = result.columns
    if not cols:
        raise ValueError("Query returned no columns to chart.")
    if not result.rows:
        raise ValueError("Query returned zero rows to chart.")

    x_col = x_col or cols[0]
    y_col = y_col or (cols[1] if len(cols) > 1 else cols[0])
    if x_col not in cols:
        raise ValueError(f"x_col {x_col!r} not in result columns {cols}")
    if y_col not in cols:
        raise ValueError(f"y_col {y_col!r} not in result columns {cols}")

    xi, yi = cols.index(x_col), cols.index(y_col)
    xs = [r[xi] for r in result.rows]
    ys = [r[yi] for r in result.rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    if kind == "bar":
        ax.bar(range(len(xs)), ys)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")
    elif kind == "line":
        ax.plot(range(len(xs)), ys, marker="o")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")
    elif kind == "pie":
        ax.pie(ys, labels=[str(x) for x in xs], autopct="%1.1f%%")
    elif kind == "scatter":
        ax.scatter(xs, ys)
    else:
        plt.close(fig)
        raise ValueError(f"Unknown chart kind: {kind!r}")

    if title:
        ax.set_title(title)
    if kind != "pie":
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    fig.tight_layout()
    path = CHARTS_DIR / f"chart_{uuid.uuid4().hex[:8]}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": str(path),
        "rows": result.row_count,
        "x_col": x_col,
        "y_col": y_col,
        "kind": kind,
    }

"""LangGraph agent that answers questions over the Olist warehouse.

Graph shape:

    START → agent ─(tool_calls?)─► tools ─► agent ─► … ─► END

The agent node calls the LLM with two tools bound: get_schema and run_sql.
The tools node executes whatever the LLM asked for. The loop continues until
the LLM emits a response with no tool calls.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.charts import render_chart as _render_chart_impl
from src.llm import build_chat_model
from src.tools import get_schema as _get_schema_impl
from src.tools import run_sql as _run_sql_impl

SYSTEM_PROMPT = """You are a careful data analyst working with the Olist Brazilian e-commerce warehouse (DuckDB).

Tools:
- get_schema(): returns all tables and their columns. Call this FIRST if you do not already know the schema.
- run_sql(query): executes a read-only SELECT query. INSERT/UPDATE/DELETE/DDL are blocked. Use this when a tabular or numeric answer is enough.
- make_chart(sql, kind, title, x_col, y_col): runs a SELECT and renders a PNG chart.
    kind ∈ {bar, line, pie, scatter}. Pick:
      - bar: comparing categories (e.g. revenue by state, count by category)
      - line: trends over time (e.g. orders per month)
      - pie: parts of a whole when there are <=6 slices
      - scatter: relationships between two numeric variables
    The SQL should return the data to plot, typically two columns: x then y.
    Defaults: x_col = first column, y_col = second column.

Workflow:
1. If you do not know the schema yet, call get_schema().
2. Decide if the user wants a chart (words like "plot", "chart", "visualize", "graph", "show me", or comparisons across many categories) or a tabular answer.
3. Write a single SQL query. Prefer simple, correct SQL over clever SQL.
4. If charting: call make_chart(sql=..., kind=..., title=..., x_col=..., y_col=...).
   Otherwise: call run_sql(query=...).
5. If a tool returns a string starting with "ERROR", READ the error, fix the query, and retry.
6. Once results are available, give a concise English answer. Quote the SQL you ran. If a chart was produced, mention it was saved.

SQL notes:
- Dialect is DuckDB. Use EXTRACT(YEAR FROM ts), DATE_TRUNC('year', ts), etc.
- Timestamps are ISO strings; treat them as TIMESTAMP.
- For charts, LIMIT to the top N rows that fit visually (e.g. LIMIT 10 for bar charts of categories).
"""


@tool
def get_schema() -> str:
    """Return the full database schema as text (tables and their columns)."""
    return _get_schema_impl()


@tool
def run_sql(query: str) -> str:
    """Run a read-only SELECT query against the warehouse.

    Returns a Markdown table on success, or a string starting with "ERROR" on failure.
    """
    try:
        result = _run_sql_impl(query)
    except ValueError as exc:
        return f"ERROR (refused): {exc}"
    except Exception as exc:  # noqa: BLE001 - surface any DuckDB error to the LLM
        return f"ERROR (sql): {exc}"
    return result.to_markdown()


@tool
def make_chart(
    sql: str,
    kind: Literal["bar", "line", "pie", "scatter"] = "bar",
    title: str = "",
    x_col: str | None = None,
    y_col: str | None = None,
) -> str:
    """Render a chart from a SELECT query as a PNG image.

    Args:
        sql: SELECT statement returning the data to plot (typically 2 columns: x, y).
        kind: One of bar, line, pie, scatter. Pick the type that fits the question.
        title: Chart title.
        x_col: Column name for the x axis. Defaults to the first column.
        y_col: Column name for the y axis. Defaults to the second column.

    Returns a status string. On success it includes "Chart saved: <path>" so the UI
    can render it inline. On failure it returns a string starting with "ERROR".
    """
    try:
        info = _render_chart_impl(sql, kind=kind, title=title, x_col=x_col, y_col=y_col)
    except ValueError as exc:
        return f"ERROR (chart): {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR (chart): {exc}"
    return (
        f"Chart saved: {info['path']}. "
        f"{info['kind'].capitalize()} chart of {info['x_col']} vs {info['y_col']}, "
        f"{info['rows']} row(s)."
    )


TOOLS = [get_schema, run_sql, make_chart]


def build_graph():
    llm_with_tools = build_chat_model().bind_tools(TOOLS)

    def call_llm(state: MessagesState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def route(state: MessagesState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_llm)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_question(question: str, recursion_limit: int = 25) -> dict:
    """Run a single user question through the agent. Returns the final state."""
    graph = build_graph()
    initial = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
    }
    return graph.invoke(initial, config={"recursion_limit": recursion_limit})

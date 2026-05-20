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

from src.llm import build_chat_model
from src.tools import get_schema as _get_schema_impl
from src.tools import run_sql as _run_sql_impl

SYSTEM_PROMPT = """You are a careful data analyst working with the Olist Brazilian e-commerce warehouse (DuckDB).

Tools:
- get_schema(): returns all tables and their columns. Call this FIRST if you do not already know the schema.
- run_sql(query): executes a read-only SELECT query. INSERT/UPDATE/DELETE/DDL are blocked.

Workflow:
1. If you do not know the schema yet, call get_schema().
2. Write a single SQL query that answers the question. Prefer simple, correct SQL over clever SQL.
3. Call run_sql(query).
4. If run_sql returns an error message starting with "ERROR", READ the error, fix the query, and call run_sql again. Typical fixes: wrong column name, wrong table, missing join.
5. Once results are available, give a concise English answer and quote the SQL you used.

SQL notes:
- Dialect is DuckDB. Use EXTRACT(YEAR FROM ts), DATE_TRUNC('year', ts), etc.
- Timestamps are ISO strings; treat them as TIMESTAMP.
- Always LIMIT large result sets.
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


TOOLS = [get_schema, run_sql]


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

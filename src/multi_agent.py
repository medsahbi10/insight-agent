"""Multi-agent orchestration: Planner → Executor → Critic.

Three specialized roles, each with its own system prompt:
- Planner: produces a short numbered plan (no tools)
- Executor: runs the plan against the warehouse (reuses the M1 graph as a subgraph)
- Critic: reviews the answer; either approves or asks for one revision

Graph shape:

    START → planner → executor → critic ─ approve? ─► END
                          ▲                  │
                          └──── revise ──────┘

The critic loop is capped at MAX_REVISIONS to keep cost and latency bounded.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.agent import SYSTEM_PROMPT as EXEC_PROMPT
from src.agent import build_graph as build_executor_graph
from src.llm import build_chat_model

MAX_REVISIONS = 1  # one revision round after the first answer


PLANNER_PROMPT = """You are a planning agent for a data analyst working with the Olist Brazilian e-commerce warehouse (DuckDB).

Given the user's question, produce a SHORT numbered plan (3 to 5 steps) describing how to answer it.

Tables available (high level):
- orders, order_items, order_payments, order_reviews
- products, customers, sellers
- geolocation, product_category_translation

Rules:
- Do NOT write SQL. Do NOT call tools. Outline the approach only.
- Mention which tables to use and what to join on.
- If the question asks for a chart, say which kind (bar / line / pie / scatter) and why.

Format:
1. <step>
2. <step>
3. <step>
"""

CRITIC_PROMPT = """You are a reviewer for a data analyst agent.

Given the user's question and the executor's final answer, decide if the answer is correct, complete, and clear.

Respond with EXACTLY one of:
- `OK` — if the answer is good.
- `REVISE: <one sentence describing what specifically needs to change>` — if the answer needs improvement (wrong column, missing detail, unclear explanation, etc.).

Do not call tools. Do not write code. Be terse.
"""


class MultiAgentState(MessagesState):
    plan: str
    critique: str
    critic_rounds: int


def build_multi_agent_graph():
    llm = build_chat_model()
    executor_graph = build_executor_graph()

    def planner_node(state: MultiAgentState) -> dict:
        question = state["messages"][0].content
        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=question),
            ]
        )
        return {"plan": response.content}

    def executor_node(state: MultiAgentState) -> dict:
        question = state["messages"][0].content
        plan = state.get("plan", "")
        critique = state.get("critique", "") or ""

        sys_msg = f"{EXEC_PROMPT}\n\nPlan to follow:\n{plan}"
        if critique and not critique.strip().upper().startswith("OK"):
            sys_msg += (
                "\n\nIMPORTANT — Reviewer feedback on your previous attempt:\n"
                f"{critique}\nAddress this feedback in your new answer."
            )

        initial = {
            "messages": [
                SystemMessage(content=sys_msg),
                HumanMessage(content=question),
            ]
        }
        result = executor_graph.invoke(initial, config={"recursion_limit": 25})

        final = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final = msg
                break
        return {"messages": [final] if final else []}

    def critic_node(state: MultiAgentState) -> dict:
        question = state["messages"][0].content
        answer = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                answer = msg.content
                break

        response = llm.invoke(
            [
                SystemMessage(content=CRITIC_PROMPT),
                HumanMessage(content=f"Question:\n{question}\n\nAnswer:\n{answer}"),
            ]
        )
        return {
            "critique": response.content,
            "critic_rounds": state.get("critic_rounds", 0) + 1,
        }

    def route_after_critic(state: MultiAgentState) -> Literal["executor", "__end__"]:
        critique = (state.get("critique") or "").strip()
        rounds = state.get("critic_rounds", 0)
        if critique.upper().startswith("OK"):
            return END
        if rounds > MAX_REVISIONS:
            return END
        return "executor"

    graph = StateGraph(MultiAgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"executor": "executor", END: END}
    )
    return graph.compile()


def run_multi_agent(question: str) -> dict:
    graph = build_multi_agent_graph()
    return graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "plan": "",
            "critique": "",
            "critic_rounds": 0,
        }
    )

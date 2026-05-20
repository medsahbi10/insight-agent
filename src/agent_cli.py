"""Run one question through the agent and print the trace + final answer.

Usage:
    python -m src.agent_cli "How many orders were delivered in 2018?"
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent import run_question


def _truncate(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} more chars]"


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m src.agent_cli "your question"', file=sys.stderr)
        return 1

    question = " ".join(sys.argv[1:])
    print(f"Q: {question}\n")

    state = run_question(question)

    for msg in state["messages"]:
        if isinstance(msg, (SystemMessage, HumanMessage)):
            continue
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_preview = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                    print(f"[agent] -> {tc['name']}({args_preview})")
            elif msg.content:
                print(f"\n[answer]\n{msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"[tool] {_truncate(msg.content)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

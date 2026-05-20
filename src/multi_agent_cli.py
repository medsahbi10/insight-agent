"""CLI: run one question through the Planner → Executor → Critic chain.

Usage:
    python -m src.multi_agent_cli "Which 5 sellers had the highest revenue in 2018 and from which state?"
"""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from langchain_core.messages import AIMessage  # noqa: E402

from src.multi_agent import run_multi_agent  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m src.multi_agent_cli "your question"', file=sys.stderr)
        return 1

    question = " ".join(sys.argv[1:])
    print(f"Q: {question}\n")

    state = run_multi_agent(question)

    print("=== PLAN ===")
    print(state.get("plan") or "(no plan produced)")
    print()

    print("=== EXECUTOR ANSWER ===")
    answer = None
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            answer = msg.content
    print(answer or "(no answer produced)")
    print()

    print("=== CRITIC ===")
    critique = (state.get("critique") or "").strip()
    rounds = state.get("critic_rounds", 0)
    print(critique or "(no critique)")
    print(f"(rounds: {rounds})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Eval harness: golden Q&A + LLM-as-judge.

Run an agent (single or multi) over the golden set, score each answer with
a structured-output judge, save raw results to evals/results/, and print
a summary table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent import run_question as run_single_agent
from src.llm import build_chat_model
from src.multi_agent import run_multi_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "evals" / "golden.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"


class JudgeScore(BaseModel):
    """Structured score returned by the LLM-as-judge."""

    correctness: Literal[0, 1] = Field(
        description="1 if the agent's answer is factually correct (numbers match within ~2% tolerance, references the right entities), 0 otherwise."
    )
    completeness: Literal[0, 1] = Field(
        description="1 if the agent fully answered every part of the question, 0 if anything is missing."
    )
    clarity: Literal[0, 1] = Field(
        description="1 if the answer is clearly written and well-formatted, 0 if confusing."
    )
    notes: str = Field(
        description="One short sentence explaining the score, especially if any axis is 0."
    )


JUDGE_SYSTEM = """You are an evaluator for a data analyst agent.

You will be given a question, a reference answer (ground truth), and the agent's actual answer.

Score the agent's answer on three binary axes:
- correctness: 1 if numbers are essentially correct (allow ±2% tolerance, accept slightly different rounding); 0 if clearly wrong.
- completeness: 1 if all parts of the question are addressed; 0 if anything is missing.
- clarity: 1 if the answer is well-formatted and easy to read; 0 otherwise.

Be lenient on phrasing differences but strict on factual accuracy. Output a structured JSON via the tool you are given.
"""


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_final_answer(state: dict) -> str:
    for msg in reversed(state.get("messages", [])):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", None)
        ):
            return msg.content
    return ""


def chart_was_produced(state: dict) -> bool:
    for msg in state.get("messages", []):
        content = getattr(msg, "content", "") or ""
        if "Chart saved:" in content:
            return True
    return False


def judge(question: str, reference: str, actual: str) -> JudgeScore:
    model = build_chat_model().with_structured_output(JudgeScore)
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"AGENT'S ANSWER:\n{actual}\n"
    )
    return model.invoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user)]
    )


def run_eval(
    agent_name: Literal["single", "multi"], questions: list[dict]
) -> list[dict]:
    """Run an agent over the questions and judge each result."""
    runner = run_single_agent if agent_name == "single" else run_multi_agent

    results: list[dict] = []
    for item in questions:
        qid = item["id"]
        print(f"[{agent_name}] Q{qid}: {item['question'][:70]}")

        t0 = time.perf_counter()
        try:
            state = runner(item["question"])
            actual = extract_final_answer(state)
            chart_ok = chart_was_produced(state) if item.get("expects_chart") else None
            err = None
        except Exception as exc:  # noqa: BLE001
            state = {"messages": []}
            actual = ""
            chart_ok = False if item.get("expects_chart") else None
            err = str(exc)
        latency = round(time.perf_counter() - t0, 2)

        if err is None:
            try:
                score = judge(item["question"], item["reference_answer"], actual)
            except Exception as exc:  # noqa: BLE001
                score = JudgeScore(
                    correctness=0,
                    completeness=0,
                    clarity=0,
                    notes=f"Judge crashed: {exc}",
                )
        else:
            score = JudgeScore(
                correctness=0,
                completeness=0,
                clarity=0,
                notes=f"Agent crashed: {err}",
            )

        print(
            f"  correctness={score.correctness} "
            f"completeness={score.completeness} "
            f"clarity={score.clarity}  ({latency}s)"
        )

        results.append(
            {
                "id": qid,
                "category": item["category"],
                "question": item["question"],
                "actual": actual,
                "score": score.model_dump(),
                "latency_seconds": latency,
                "error": err,
                "chart_produced": chart_ok,
            }
        )

    return results


def summarize(results: list[dict]) -> dict:
    n = len(results) or 1
    correct = sum(r["score"]["correctness"] for r in results)
    complete = sum(r["score"]["completeness"] for r in results)
    clear = sum(r["score"]["clarity"] for r in results)
    total_latency = sum(r["latency_seconds"] for r in results)
    return {
        "n": len(results),
        "correctness_pct": round(100 * correct / n, 1),
        "completeness_pct": round(100 * complete / n, 1),
        "clarity_pct": round(100 * clear / n, 1),
        "correctness_raw": f"{correct}/{n}",
        "completeness_raw": f"{complete}/{n}",
        "clarity_raw": f"{clear}/{n}",
        "avg_latency_seconds": round(total_latency / n, 2),
        "total_latency_seconds": round(total_latency, 2),
    }


def save_results(results: list[dict], agent_name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{agent_name}_{timestamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path

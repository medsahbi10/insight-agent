"""CLI: run the eval harness against the golden Q&A set.

Usage:
    python -m src.evals_cli                       # all questions, single agent
    python -m src.evals_cli --agent multi          # all questions, multi-agent
    python -m src.evals_cli --agent both           # both, prints comparison
    python -m src.evals_cli --limit 3              # first 3 questions only (smoke)
"""

from __future__ import annotations

import argparse
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from src.evals import load_golden, run_eval, save_results, summarize  # noqa: E402


def print_summary(name: str, summary: dict, save_path) -> None:
    print()
    print(f"=== {name.upper()} AGENT — SUMMARY ===")
    print(f"n questions     : {summary['n']}")
    print(
        f"correctness     : {summary['correctness_raw']}  ({summary['correctness_pct']}%)"
    )
    print(
        f"completeness    : {summary['completeness_raw']}  ({summary['completeness_pct']}%)"
    )
    print(f"clarity         : {summary['clarity_raw']}  ({summary['clarity_pct']}%)")
    print(f"avg latency / Q : {summary['avg_latency_seconds']} s")
    print(f"total latency   : {summary['total_latency_seconds']} s")
    print(f"results saved   : {save_path}")


def print_comparison(single: dict, multi: dict) -> None:
    print()
    print("=== COMPARISON: single vs multi ===")
    print(f"{'metric':<22} {'single':>12} {'multi':>12}")
    print("-" * 50)
    rows = [
        ("correctness %", single["correctness_pct"], multi["correctness_pct"]),
        ("completeness %", single["completeness_pct"], multi["completeness_pct"]),
        ("clarity %", single["clarity_pct"], multi["clarity_pct"]),
        ("avg latency (s)", single["avg_latency_seconds"], multi["avg_latency_seconds"]),
        (
            "total latency (s)",
            single["total_latency_seconds"],
            multi["total_latency_seconds"],
        ),
    ]
    for label, s, m in rows:
        print(f"{label:<22} {s:>12} {m:>12}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent eval against golden Q&A set.")
    parser.add_argument("--agent", choices=["single", "multi", "both"], default="single")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N questions"
    )
    args = parser.parse_args()

    questions = load_golden()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Running {len(questions)} question(s) against {args.agent} agent(s)...\n")

    single_summary = single_path = None
    multi_summary = multi_path = None

    if args.agent in ("single", "both"):
        results = run_eval("single", questions)
        single_path = save_results(results, "single")
        single_summary = summarize(results)

    if args.agent in ("multi", "both"):
        results = run_eval("multi", questions)
        multi_path = save_results(results, "multi")
        multi_summary = summarize(results)

    if single_summary is not None:
        print_summary("single", single_summary, single_path)
    if multi_summary is not None:
        print_summary("multi", multi_summary, multi_path)
    if args.agent == "both":
        print_comparison(single_summary, multi_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

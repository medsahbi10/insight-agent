"""Phoenix + OpenTelemetry instrumentation for agent runs.

Calling init_phoenix() starts an in-process Phoenix server (default :6006)
and auto-instruments LangChain / LangGraph so every LLM call, tool call,
and graph step shows up as a trace.

The function is idempotent — safe to call multiple times in the same process.
Set DISABLE_PHOENIX=1 in .env to skip everything (useful for tests).
"""

from __future__ import annotations

import os
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

_LOCK = threading.Lock()
_STARTED = False
_PHOENIX_URL: str | None = None


def init_phoenix(project_name: str = "insight-agent") -> str | None:
    """Start Phoenix and wire OpenTelemetry. Returns the Phoenix UI URL.

    Returns None if Phoenix is disabled or fails to start.
    """
    global _STARTED, _PHOENIX_URL
    with _LOCK:
        if _STARTED:
            return _PHOENIX_URL
        if os.getenv("DISABLE_PHOENIX") == "1":
            _STARTED = True
            return None

        # Windows default console is cp1252; Phoenix's startup banner contains
        # emoji that crash that codec. Force UTF-8 before importing phoenix.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass

        try:
            import phoenix as px
            from phoenix.otel import register

            session = px.launch_app()
            _PHOENIX_URL = getattr(session, "url", "http://localhost:6006")

            register(project_name=project_name, auto_instrument=True)
            _STARTED = True
            return _PHOENIX_URL
        except Exception as exc:  # noqa: BLE001 — observability is best-effort
            print(f"[observability] Phoenix init failed: {exc}")
            _STARTED = True
            return None


def phoenix_url() -> str | None:
    return _PHOENIX_URL

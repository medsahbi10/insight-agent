"""Phoenix + OpenTelemetry instrumentation for agent runs.

Calling init_phoenix() starts an in-process Phoenix server (default :6006)
and auto-instruments LangChain / LangGraph so every LLM call, tool call,
and graph step shows up as a trace.

The function is idempotent — safe to call multiple times in the same process.
Set DISABLE_PHOENIX=1 in .env to skip everything (useful for tests).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

_LOCK = threading.Lock()
_STARTED = False
_PHOENIX_URL: str | None = None
_LAST_ERROR: str | None = None


def init_phoenix(project_name: str = "insight-agent") -> str | None:
    """Start Phoenix and wire OpenTelemetry. Returns the Phoenix UI URL.

    Returns None if Phoenix is disabled or fails to start.
    Use phoenix_error() to retrieve the failure reason in the latter case.
    """
    global _STARTED, _PHOENIX_URL, _LAST_ERROR
    with _LOCK:
        if _STARTED:
            return _PHOENIX_URL
        if os.getenv("DISABLE_PHOENIX") == "1":
            _LAST_ERROR = None
            _STARTED = True
            return None

        # Phoenix prints an emoji-bearing banner during launch. On Windows
        # consoles or under Streamlit's stdout wrapper, that emoji crashes
        # the default cp1252 codec. Redirect Phoenix's chatter to a buffer
        # so encoding issues cannot kill init.
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                import phoenix as px
                from phoenix.otel import register

                session = px.launch_app()
                register(project_name=project_name, auto_instrument=True)

            _PHOENIX_URL = getattr(session, "url", None) or "http://localhost:6006"
            _LAST_ERROR = None
            _STARTED = True
            return _PHOENIX_URL
        except Exception as exc:  # noqa: BLE001 — observability is best-effort
            _LAST_ERROR = f"{type(exc).__name__}: {exc}"
            # Surface the captured Phoenix logs alongside the exception so
            # whoever calls phoenix_error() sees the full picture.
            logs = sink.getvalue().strip()
            if logs:
                _LAST_ERROR += f"\n--- phoenix logs ---\n{logs[-800:]}"
            print(f"[observability] Phoenix init failed: {_LAST_ERROR}", file=sys.stderr)
            _STARTED = True
            return None


def phoenix_url() -> str | None:
    return _PHOENIX_URL


def phoenix_error() -> str | None:
    """Return the last init failure reason, or None if Phoenix is healthy/disabled."""
    return _LAST_ERROR

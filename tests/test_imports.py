"""Import smoke tests — fail fast if any module breaks on import.

Catches missing dependencies, broken refactors, and circular imports
without needing a warehouse, API key, or network.
"""

import importlib

import pytest

MODULES = [
    "src.db",
    "src.tools",
    "src.cli",
    "src.llm",
    "src.charts",
    "src.agent",
    "src.agent_cli",
    "src.multi_agent",
    "src.multi_agent_cli",
    "src.observability",
    "src.evals",
    "src.evals_cli",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)

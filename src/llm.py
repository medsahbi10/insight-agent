"""LLM provider factory. Reads GROQ_API_KEY and GROQ_MODEL from .env."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def build_chat_model(temperature: float = 0.0) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=temperature,
    )

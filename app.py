"""Streamlit chat UI for the Insight Agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import re

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

CHART_PATH_RE = re.compile(r"Chart saved: (.+?\.png)")

from src.agent import SYSTEM_PROMPT, build_graph
from src.observability import init_phoenix, phoenix_error

st.set_page_config(page_title="Insight Agent", page_icon="📊", layout="wide")

PHOENIX_URL = init_phoenix()


# ---------- session state ----------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": str, "content": str}
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()


# ---------- sidebar ----------
with st.sidebar:
    st.title("📊 Insight Agent")
    st.caption(
        "Conversational data analyst over the Olist Brazilian e-commerce "
        "warehouse. Ask about orders, products, reviews, payments, customers, "
        "or sellers."
    )
    st.divider()
    st.subheader("Observability")
    if PHOENIX_URL:
        st.markdown(f"[Open Phoenix traces ↗]({PHOENIX_URL})")
        st.caption("Every question produces a full LLM + tool-call trace.")
    else:
        err = phoenix_error()
        if err:
            st.error("Phoenix init failed")
            st.caption(err)
        else:
            st.caption("Phoenix is disabled.")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ---------- replay prior turns ----------
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])


# ---------- new turn ----------
question = st.chat_input("Ask a question about the warehouse…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=True)
        final_answer = ""
        seen = 0
        initial = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        }
        try:
            for state in st.session_state.graph.stream(
                initial, config={"recursion_limit": 25}, stream_mode="values"
            ):
                messages = state["messages"]
                for msg in messages[seen:]:
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                args = tc.get("args", {})
                                with status:
                                    if tc["name"] == "run_sql":
                                        st.markdown("**→ run_sql**")
                                        st.code(args.get("query", ""), language="sql")
                                    else:
                                        st.markdown(f"**→ {tc['name']}**")
                                        if args:
                                            st.code(repr(args))
                        elif msg.content:
                            final_answer = msg.content
                    elif isinstance(msg, ToolMessage):
                        content = msg.content or ""
                        preview = (
                            content if len(content) < 800
                            else content[:800] + " …(truncated)"
                        )
                        with status:
                            st.markdown(f"```\n{preview}\n```")
                            chart_match = CHART_PATH_RE.search(content)
                            if chart_match:
                                st.image(chart_match.group(1))
                seen = len(messages)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            status.update(label=f"Agent error: {exc}", state="error", expanded=True)
            final_answer = f"_Agent error: {exc}_"

        st.markdown(final_answer)
        st.session_state.history.append({"role": "assistant", "content": final_answer})

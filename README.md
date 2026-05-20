# Insight Agent

A conversational data-analyst agent that answers business questions over an e-commerce database. Built on LangGraph + Ollama + DuckDB, fully open source.

## What it does

Ask a business question in natural language → the agent inspects the schema, plans a query, writes SQL with safety guards, executes it, self-corrects on errors, generates a chart, and explains the answer with citations to the SQL it ran.

Example:
```
> Which product categories had the highest review scores in 2018?
[agent inspects schema -> writes SQL -> runs it -> charts top 10]
```

## Stack

- **Agent framework**: LangGraph
- **LLM**: Llama 3.3 70B (open-weights, Llama 3 License) served via Groq Cloud free tier. Swap to Ollama / vLLM with a one-line provider change.
- **Database**: DuckDB (analytical, single-file)
- **UI**: Streamlit
- **Observability**: Arize Phoenix (OpenInference / OpenTelemetry traces)
- **Container**: Docker Compose
- **CI**: GitHub Actions

## Dataset

[Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — ~100K orders across 9 tables (orders, order_items, products, customers, sellers, reviews, payments, geolocation, category translations), 2016–2018.

## Quickstart

```bash
# 1. Install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the Olist dataset from Kaggle and unzip into data/raw/
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

# 3. Load CSVs into DuckDB
python scripts/load_data.py

# 4. Verify with the CLI (no LLM yet)
python -m src.cli "SELECT COUNT(*) FROM orders"

# 5. Get a free Groq API key at https://console.groq.com,
#    copy .env.example to .env, and paste the key.
```

LLM integration and Streamlit UI are added in the next milestone.

## Project structure

```
insight-agent/
├── data/
│   ├── raw/           # Olist CSVs (gitignored)
│   └── duckdb/        # built warehouse (gitignored)
├── scripts/
│   └── load_data.py   # CSV -> DuckDB ingestion
├── src/
│   ├── db.py          # DuckDB connection + schema helpers
│   ├── tools.py       # agent tools: get_schema, run_sql
│   └── cli.py         # manual SQL runner (pre-agent)
└── tests/
```

## Milestones

- [x] M0 — Scaffold + data loader + SQL tools (current)
- [ ] M1 — LangGraph agent with `get_schema` + `run_sql` + reflection loop
- [ ] M2 — Streamlit chat UI + Phoenix tracing
- [ ] M3 — Chart-generation tool (`make_chart`)
- [ ] M4 — Planner / executor / critic multi-agent split
- [ ] M5 — Evaluation harness (golden Q&A set, LLM-as-judge)
- [ ] M6 — Docker Compose + GitHub Actions CI
- [ ] M7 — Semantic layer / metric definitions

## License

MIT

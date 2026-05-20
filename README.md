# Insight Agent

A conversational data-analyst agent that answers business questions over an e-commerce database. Built on LangGraph + Groq + DuckDB, fully open source.

## What it does

Ask a business question in natural language → the agent inspects the schema, plans a query, writes SQL with safety guards, executes it, self-corrects on errors, optionally renders a chart, and explains the answer with citations to the SQL it ran.

Example:
```
> Plot the top 8 product categories by total revenue in 2018 as a bar chart
[agent] -> get_schema()
[agent] -> make_chart(sql=..., kind='bar', title='Top 8 product categories by revenue (2018)')
[tool]    Chart saved: charts/chart_xxxxxxxx.png. Bar chart of category vs revenue, 8 rows.
[answer]  Here is the bar chart you asked for. SQL used: ...
```

## Sample outputs

| Bar — categories by revenue | Line — orders per month |
|---|---|
| ![Top categories](docs/screenshots/bar_top_categories_2018.png) | ![Orders per month](docs/screenshots/line_orders_per_month_2018.png) |

| Pie — payment types | Scatter — price vs freight |
|---|---|
| ![Payment types](docs/screenshots/pie_payment_types.png) | ![Price vs freight](docs/screenshots/scatter_price_vs_freight.png) |

Each image is produced by the same `make_chart` tool the agent calls at runtime.

## Stack

- **Agent framework**: LangGraph
- **LLM**: OpenAI gpt-oss-120B (open-weights, Apache 2.0) served via Groq Cloud free tier. Swap to Ollama / vLLM / Llama / Qwen with a one-line provider change.
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

# 6. Ask the agent from the CLI
python -m src.agent_cli "How many orders were delivered in 2018?"

# 7. Or launch the chat UI with traces
streamlit run app.py
# Chat: http://localhost:8501   Traces: http://localhost:6006
```

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

- [x] M0 — Scaffold + data loader + SQL tools
- [x] M1 — LangGraph agent with `get_schema` + `run_sql` + reflection loop
- [x] M2 — Streamlit chat UI + Phoenix tracing
- [ ] M3 — Chart-generation tool (`make_chart`)
- [ ] M4 — Planner / executor / critic multi-agent split
- [ ] M5 — Evaluation harness (golden Q&A set, LLM-as-judge)
- [ ] M6 — Docker Compose + GitHub Actions CI
- [ ] M7 — Semantic layer / metric definitions

## License

MIT

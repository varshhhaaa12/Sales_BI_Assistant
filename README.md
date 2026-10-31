# Mini AI Business Intelligence Assistant

Ask questions about sales data in plain English and get answers backed by real database queries. Every answer shows the figures behind it and the SQL that produced them.

Without an API key, the app answers a wide range of questions offline: any metric × filter × grouping × highest/lowest/top-N. With an API key, Claude also plans queries and writes validated SQL for questions outside that range.

## Quick start

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Start application
uvicorn app.main:app --reload

# Run tests
python -m pytest -q

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload        # open http://127.0.0.1:8000
pytest -q                            # run the test suite
```

**Optional LLM (any provider).** The LLM is configured only through environment variables, or a `.env` file. The app talks to any **OpenAI-compatible Chat Completions endpoint**, so switching between Gemini, OpenAI, Claude, Groq, OpenRouter or a local Ollama model needs no code change. Without configuration the app runs fully offline on the rule-based interpreter.

```bash
cp .env.example .env        # then edit: e.g. LLM_PROVIDER=gemini, LLM_MODEL=gemini-3.5-flash, LLM_API_KEY=...
uvicorn app.main:app --reload
curl localhost:8000/api/health   # {"interpreter": "gemini/gemini-3.5-flash+rules"}
```

| Env var | Purpose |
|---|---|
| `LLM_PROVIDER` | Preset that fills in the base URL: `gemini`, `openai`, `anthropic`, `groq`, `openrouter`, `ollama` |
| `LLM_MODEL` | Model name (required) |
| `LLM_API_KEY` | API key. Falls back to the preset's own variable (`GEMINI_API_KEY`, `OPENAI_API_KEY`, …) |
| `LLM_BASE_URL` | Any OpenAI-compatible endpoint; overrides the preset |
| `LLM_JSON_MODE` | `none` (default), `json_object` or `json_schema`, for providers with a native JSON mode |
| `LLM_TIMEOUT` | Request timeout in seconds (default 30) |
| `BI_USE_LLM` | `0` forces rules-only mode |
| `BI_CSV_PATH` / `BI_DB_PATH` | Input file and generated database |

Real environment variables take precedence over `.env`. A misconfiguration (missing model or key, unknown provider) is logged, and the app keeps running on the rules.

## Architecture

```
 Browser (static/index.html)
        │  POST /api/ask {"question": "..."}
        ▼
 FastAPI (app/main.py) ── validates the request (non-empty, ≤500 chars) → 400 on bad input
        │
        ▼
 Interpreter (app/interpreter.py)
   ├─ LLMInterpreter  (if configured) → any LLM via app/llm.py returns a JSON plan,
   │                                     validated with Pydantic (1 retry on invalid JSON)
   └─ RuleInterpreter (always available, also the fallback on any LLM error)
        │  Intent                                   "unsupported" → 422 + reason/suggestions
        ▼
 Analytics (app/analytics.py) - three levels, most constrained first:
   1. named intents       hand-written SQL (total, top product/region, trend, comparison, breakdown)
   2. custom_query        QuerySpec(metric, filters, group_by, order, limit) compiled from
                          whitelisted SQL fragments; values checked against the data
   3. sql_query           LLM-written SQL → strict untrusted mode (below); on rejection the
                          question is retried with the rules interpreter
        │
        ▼
 ReadOnlyDB (app/db.py)
   all queries:   SELECT-only check · read-only connection · authorizer denies writes
   untrusted SQL: + only tables sales/revenue_sales · only whitelisted functions
                  + max 200 rows · execution-step watchdog
        │
        ▼
 (sql_query only) LLM phrases the answer → accepted only if every number in it
                  appears in the query result; otherwise a template answer is used
        │
        ▼
 SQLite: sales · rejected_records · revenue_sales (VIEW)
```

The main design choice is **structured query planning first, text-to-SQL only as a last resort**. Most questions become a validated query spec, so the SQL is built from whitelisted pieces. Free-form SQL is used only when the spec can't express the question, and it runs under much tighter limits. Every number comes from SQLite. Neither the rules nor the LLM do arithmetic.

## Data management

On startup `load_csv()` rebuilds `data/sales.db` from the CSV. Rebuilding is idempotent, so a restart always reflects the current file.

| Table / view | Contents |
|---|---|
| `sales` | Every row that passed validation (all statuses). Dates stored as ISO `YYYY-MM-DD` plus a precomputed `order_month`. `CHECK` constraints enforce quantity > 0, price > 0, discount ≥ 0. |
| `rejected_records` | Rows that failed validation: CSV line number, transaction id, reason, raw JSON. Visible at `GET /api/data-quality`. |
| `revenue_sales` (view) | Only `status = 'Completed'` rows, plus `gross_revenue` and `net_revenue` columns. **Revenue is defined only here.** |

**Revenue definition:** `quantity × unit_price − discount_amount`, completed orders only.

**Validation rules.** A row is rejected for:

- a missing or duplicate id
- a missing text field
- a date not in `DD-MM-YYYY` format
- a non-numeric or ≤ 0 quantity or price
- a negative discount, or a discount larger than the order value
- an unknown status

Region, channel and status casing is normalised.

**What happens to the supplied data:**

| Row | Issue | Handling |
|---|---|---|
| TXN0034 | Quantity = 0 | Rejected (an order of nothing carries no revenue and is likely an entry error) |
| TXN0051 | Unit_price missing | Rejected (revenue can't be computed; we don't guess a price) |
| TXN0033 | Status = Returned | Loaded but excluded from revenue |

The result is 51 rows: 49 loaded, 2 rejected, and 48 rows that count toward revenue.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/ask` | `{"question": "..."}` → structured answer |
| `GET` | `/api/data-quality` | Load statistics and rejected rows with reasons |
| `GET` | `/api/examples` | Example questions |
| `GET` | `/api/health` | Status and interpreter mode |

Example successful response:

```json
{
  "question": "How did revenue change between March and April?",
  "intent": "month_comparison",
  "parameters": {"dimension": null, "months": ["2026-03", "2026-04"]},
  "interpreted_by": "rules",
  "answer": "Revenue increased from ₹223,900.00 in March 2026 to ₹240,700.00 in April 2026, a change of +₹16,800.00 (+7.5%).",
  "data": {"columns": ["month", "net_revenue", "orders", "units"], "rows": [["2026-03", 223900.0, 8, 73], ["2026-04", 240700.0, 7, 120]]},
  "chart": {"type": "bar", "x": "month", "y": "net_revenue"},
  "sql": "SELECT order_month AS month, ... WHERE order_month IN (:month_a, :month_b) ...",
  "sql_params": {"month_a": "2026-03", "month_b": "2026-04"},
  "assumptions": ["Revenue = quantity x unit_price - discount_amount, counting only orders with status 'Completed' ..."]
}
```

Errors always have the shape `{"error": {"code", "message", ...}}`:

| Status | Code | When |
|---|---|---|
| 400 | `invalid_request` | Empty, missing, too long, or non-JSON body |
| 422 | `unsupported_question` | The question doesn't map to a supported intent (includes `suggestions`) |
| 422 | `cannot_answer` | The question was understood but the data can't answer it (e.g. a month with no data, a comparison naming only one month) |
| 500 | `query_blocked` / `internal_error` | The safety guard tripped, or an unexpected failure (logged server-side, never leaked to the client) |

## What you can ask

**Offline (rules), with no API key needed.** Combine any of the following:

| Part | Options | Example wording |
|---|---|---|
| Metric | net revenue (default), gross revenue, units, orders, average order value, discount, returned orders, returned value | "units", "how many orders", "AOV", "returns" |
| Filters | any product, region, category, channel, month(s) present in the data | "Monitor in the North in March" |
| Grouping | product, region, category, channel, month | "by region", "which month", "Online vs Retail" |
| Ranking | highest, lowest, top-N / bottom-N | "lowest", "fewest", "top 3" |

Examples (all tested):
- "How many units of Monitor were sold in the North?"
- "Top 3 products by units sold in March"
- "Which month had the lowest sales?"
- "Average order value by sales channel"
- "Compare Desk revenue in March and April"
- "How many returned orders?"

The five questions from the brief keep their purpose-built answers: totals with the gross/discount split, month-over-month %, and share of total.

**With an LLM configured.** The model handles paraphrases the keyword rules miss. For questions outside the spec (e.g. "how many different products did each region sell?") it writes SQL, which then goes through the untrusted-SQL checks.

## Preventing unsupported or incorrect answers

1. **Refuse rather than guess.**
   - Forecasts, profit/margin, customers and off-topic questions return `unsupported_question`.
   - So do **unknown names**. "revenue from Tablet" or "sales in Hyderabad" is refused with the list of known values, instead of silently returning the unfiltered total.
2. **Numbers come only from SQL.**
   - The interpreter outputs a plan, never a figure.
   - Answers are templated from the query results.
   - The one exception is the LLM-written summary for free-form SQL. It is checked so that every number it states appears in the result; otherwise it is thrown away.
3. **LLM output is validated.**
   - The structured plan is parsed into a Pydantic model with enum-typed fields.
   - Filter values and months are checked against the data.
   - Generated SQL runs in strict mode: allowed tables and functions only, plus row and step limits.
   - Any refusal, timeout, validation failure or blocked SQL falls back to the rules interpreter.
4. **Parameters are checked against the data.** A month not present in the data produces a clear message listing the months that are available.
5. **Write requests are refused.** Questions containing write verbs (drop, delete, update, …) are rejected before any query is chosen.
6. **The database is read-only by construction.** The checks below would block a destructive query even if one got through the steps above:
   - a static check (one statement, starting with `SELECT` or `WITH`)
   - a `mode=ro` connection
   - an SQLite authorizer that denies every action except read
7. **Transparency.** Every response includes the SQL, its parameters, the full supporting rows and the assumptions it made (e.g. "ranked by net revenue, not units").

## Testing

`pytest -q` runs 84 tests in about 2 seconds, offline. Expected figures are computed **independently in plain Python from the raw CSV** (`tests/conftest.py::oracle`), not by reusing the app's SQL.

- `tests/test_api.py`
  - all five brief questions with exact values
  - invalid payloads
  - unsupported and destructive questions (and a check that the data is unchanged afterwards)
  - edge cases: a comparison naming only one month, a month outside the data range, the data-quality report
- `tests/test_components.py`
  - loader rejection rules
  - the SQL guard, including `WITH … DELETE`, which passes the static check and must be stopped by the authorizer
  - rule-interpreter parsing, including "may" used as a verb
  - the query-spec builder
  - unknown-term refusal
  - untrusted-SQL restrictions (forbidden tables, `load_extension`, runaway joins)
  - the number-grounding check
  - LLM fallback, using fake LLM clients (no network)
- `tests/test_text_to_sql.py`: the full text-to-SQL path with a fake LLM
  - a valid query is answered
  - an ungrounded summary is discarded
  - `DELETE` is blocked and the rules answer instead
  - a forbidden table is rejected
- `tests/test_providers.py`: env-var configuration (every preset, a custom URL, misconfiguration), plus the HTTP protocol against a fake server
  - request URL, auth header and model
  - fenced JSON
  - retry with the error message
  - provider HTTP 500 and repeated invalid JSON fall back to the rules

## Known limitations

- **Rule interpreter coverage.** It is keyword-based.
  - Slang or indirect phrasing ("how did the chairs do?") may be refused or misread. The LLM mode is the fix.
  - Offline, one metric per question. "Revenue and units by region" returns revenue only.
  - Offline, one grouping per question. "Revenue by region and month" isn't supported without the LLM.
- **No relative dates.** "last month", "this quarter" and "Q1" are not understood.
- **The grounding check is number-based.** It catches made-up figures but not a wrong *claim* about the right numbers. The SQL and rows are always shown so a user can verify.
- **Currency is assumed.** The CSV has no currency column; ₹ is assumed.
- **Returned orders are simply excluded.** They are not netted against the original sale. The dataset has only one return, with no link to an original transaction.
- **The LLM path has no live test.** It is exercised with fakes in the tests. A real call needs an API key, and accuracy on paraphrases hasn't been benchmarked.
- **Not production-hardened.** There is no auth, rate limiting, or caching. The database is rebuilt on every start.

## Project layout

```
app/
  db.py           CSV loading, validation, schema, read-only query guard
  analytics.py    named intents, QuerySpec builder, text-to-SQL runner, grounding check
  interpreter.py  rule-based parser + LLM planner (prompts, validation)
  llm.py          generic OpenAI-compatible LLM client, configured by env vars
  main.py         FastAPI app and error handling
  static/index.html  single-page UI (no build step, inline SVG chart)
tests/            pytest suite (83 tests)
data/sales.csv    supplied dataset
AI_USAGE.md       how AI tools were used during development
```

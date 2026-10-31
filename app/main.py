"""FastAPI application: POST /api/ask is the single entry point for questions."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from .analytics import Analytics, AnalyticsError, answer_is_grounded
from .db import ReadOnlyDB, UnsafeQueryError, load_csv
from .interpreter import DataContext, Interpreter

log = logging.getLogger("bi")
ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

EXAMPLE_QUESTIONS = [
    "What is the total revenue?",
    "Which product generated the highest revenue?",
    "What is the monthly revenue trend?",
    "Which region generated the highest sales?",
    "How did revenue change between March and April?",
    "Which product had the lowest revenue?",
    "How many units of Monitor were sold in the North?",
    "Top 3 products by units sold in March",
    "Average order value by sales channel",
]


def load_env_file(path: Path) -> None:
    """Minimal .env support: KEY=VALUE lines; real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class AskRequest(BaseModel):
    question: str = Field(..., max_length=500)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


def error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra}})


def create_app(csv_path: Path | None = None, db_path: Path | None = None,
               use_llm: bool | None = None) -> FastAPI:
    csv_path = csv_path or Path(os.getenv("BI_CSV_PATH", ROOT / "data" / "sales.csv"))
    db_path = db_path or Path(os.getenv("BI_DB_PATH", ROOT / "data" / "sales.db"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        report = load_csv(csv_path, db_path)
        log.info("Loaded %d/%d rows (%d rejected)", report.loaded_rows, report.total_rows, len(report.rejected))
        app.state.load_report = report
        app.state.analytics = Analytics(ReadOnlyDB(db_path))
        app.state.data_context = DataContext(app.state.analytics.available_months(),
                                             app.state.analytics.dimension_values())
        app.state.interpreter = Interpreter(use_llm=use_llm)
        yield

    app = FastAPI(title="Mini AI BI Assistant", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        return error(400, "invalid_request", first.get("msg", "invalid request"))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health(request: Request):
        return {"status": "ok", "interpreter": request.app.state.interpreter.mode}

    @app.get("/api/examples")
    def examples():
        return {"examples": EXAMPLE_QUESTIONS}

    @app.get("/api/data-quality")
    def data_quality(request: Request):
        report = request.app.state.load_report
        analytics: Analytics = request.app.state.analytics
        _, status_rows = analytics.db.query("SELECT status, COUNT(*) AS rows FROM sales GROUP BY status")
        return {
            "total_rows": report.total_rows,
            "loaded_rows": report.loaded_rows,
            "rejected_rows": len(report.rejected),
            "rejected": [{k: r[k] for k in ("row_number", "transaction_id", "reason")} for r in report.rejected],
            "loaded_by_status": dict(status_rows),
        }

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request):
        analytics: Analytics = request.app.state.analytics
        interpreter: Interpreter = request.app.state.interpreter
        ctx: DataContext = request.app.state.data_context

        intent, interpreted_by = interpreter.interpret(body.question, ctx)
        if intent.name == "unsupported":
            return error(422, "unsupported_question",
                         intent.note or
                         "I couldn't map that to the sales data. I can answer questions about revenue, "
                         "units, orders, average order value, discounts and returns - by product, region, "
                         "category, channel or month, including highest/lowest, top-N and month comparisons.",
                         suggestions=EXAMPLE_QUESTIONS, interpreted_by=interpreted_by)
        try:
            try:
                result = analytics.run(intent)
            except (UnsafeQueryError, sqlite3.Error) as exc:
                if intent.name != "sql_query":
                    raise
                # AI-written SQL was rejected or invalid: fall back to the deterministic interpreter.
                rejected_sql = intent.sql
                log.warning("Generated SQL rejected (%s): %s", exc, rejected_sql)
                intent, interpreted_by = interpreter.rules.interpret(body.question, ctx), "rules"
                if intent.name == "unsupported":
                    return error(422, "cannot_answer",
                                 "The AI-generated query was rejected by the safety checks, and the question "
                                 "could not be answered another way.", rejected_sql=rejected_sql)
                result = analytics.run(intent)
        except AnalyticsError as exc:
            return error(422, "cannot_answer", str(exc), intent=intent.name, interpreted_by=interpreted_by)
        except UnsafeQueryError:
            log.exception("Query blocked by read-only guard")
            return error(500, "query_blocked", "The generated query was blocked by the safety guard.")
        except Exception:
            log.exception("Unexpected error answering %r", body.question)
            return error(500, "internal_error", "Something went wrong while answering the question.")

        if intent.name == "sql_query" and interpreter.llm and result.rows:
            try:
                summary = interpreter.llm.summarize(body.question, result.columns, result.rows)
                if answer_is_grounded(summary, result.columns, result.rows):
                    result.answer = summary
                else:
                    result.assumptions.append("The AI summary was discarded because it contained "
                                              "numbers that are not in the query result.")
            except Exception as exc:
                log.warning("Summary failed, keeping template answer: %s", exc)

        return {
            "question": body.question,
            "intent": intent.name,
            "parameters": {"dimension": intent.dimension, "months": intent.months,
                           "spec": asdict(intent.spec) if intent.spec else None},
            "interpreted_by": interpreted_by,
            "answer": result.answer,
            "data": {"columns": result.columns, "rows": result.rows},
            "chart": result.chart,
            "sql": result.sql,
            "sql_params": result.params,
            "assumptions": result.assumptions,
        }

    return app


load_env_file(ROOT / ".env")
app = create_app()

"""Unit tests for the loader, the read-only SQL guard and the interpreters."""

from pathlib import Path

import pytest

from app.analytics import Intent
from app.db import DataLoadError, ReadOnlyDB, UnsafeQueryError, load_csv
from app.interpreter import DataContext, Interpreter, RuleInterpreter, extract_months
from tests.conftest import CSV_PATH

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
CTX = DataContext(MONTHS, {
    "product": ["Desk", "Keyboard", "Laptop Stand", "Monitor", "Office Chair", "USB-C Hub", "Wireless Mouse"],
    "region": ["East", "North", "South", "West"],
    "category": ["Accessories", "Electronics", "Furniture"],
    "sales_channel": ["Online", "Retail"],
})
HEADER = "Transaction_Id,Order_date,Product,Category,Region,Sales_channel,Quantity,Unit_price,Discount_amount,Status\n"


@pytest.fixture
def db(tmp_path) -> ReadOnlyDB:
    path = tmp_path / "t.db"
    load_csv(CSV_PATH, path)
    return ReadOnlyDB(path)


# --- Loader ---------------------------------------------------------------------------


def test_loader_rejects_bad_rows(tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(HEADER + "\n".join([
        "T1,01-01-2026,Desk,Furniture,north,online,2,100,0,completed",   # valid; casing normalised
        "T1,02-01-2026,Desk,Furniture,North,Online,2,100,0,Completed",   # duplicate id
        "T2,2026-01-03,Desk,Furniture,North,Online,2,100,0,Completed",   # wrong date format
        "T3,04-01-2026,Desk,Furniture,North,Online,abc,100,0,Completed", # non-numeric quantity
        "T4,05-01-2026,Desk,Furniture,North,Online,2,100,500,Completed", # discount > order value
        "T5,06-01-2026,Desk,Furniture,North,Online,2,100,0,Shipped",     # unknown status
        "T6,07-01-2026,,Furniture,North,Online,2,100,0,Completed",       # missing product
    ]))
    report = load_csv(csv_file, tmp_path / "bad.db")
    assert report.loaded_rows == 1
    assert [r["transaction_id"] for r in report.rejected] == ["T1", "T2", "T3", "T4", "T5", "T6"]
    _, rows = ReadOnlyDB(tmp_path / "bad.db").query("SELECT region, sales_channel, status FROM sales")
    assert rows == [["North", "Online", "Completed"]]


def test_loader_fails_fast_on_wrong_columns(tmp_path):
    csv_file = tmp_path / "cols.csv"
    csv_file.write_text("id,amount\n1,2\n")
    with pytest.raises(DataLoadError):
        load_csv(csv_file, tmp_path / "x.db")


def test_returned_orders_excluded_from_revenue(db):
    _, rows = db.query("SELECT COUNT(*) FROM revenue_sales WHERE transaction_id = 'TXN0033'")
    assert rows == [[0]]


# --- Read-only guard ------------------------------------------------------------------


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "SELECT 1; DROP TABLE sales",
    "UPDATE sales SET quantity = 0",
    "PRAGMA writable_schema = 1",
    "",
])
def test_guard_rejects_non_select(db, sql):
    with pytest.raises(UnsafeQueryError):
        db.query(sql)


def test_guard_blocks_write_hidden_behind_with(db):
    # Passes the static "starts with WITH" check; the SQLite authorizer must stop it.
    with pytest.raises(UnsafeQueryError):
        db.query("WITH x AS (SELECT 1) DELETE FROM sales")
    assert db.query("SELECT COUNT(*) FROM sales")[1] == [[49]]


# --- Interpreters ---------------------------------------------------------------------


@pytest.mark.parametrize("question, intent, months", [
    ("What is the total revenue?", "total_revenue", []),
    ("total sales in feb", "total_revenue", ["2026-02"]),
    ("Which product generated the highest revenue?", "top_product", []),
    ("best selling product in June 2026", "top_product", ["2026-06"]),
    ("What is the monthly revenue trend?", "monthly_trend", []),
    ("Which region generated the highest sales?", "top_region", []),
    ("May revenue vs June revenue", "month_comparison", ["2026-05", "2026-06"]),
    ("Is revenue in 2026-06 higher than 2026-05?", "month_comparison", ["2026-06", "2026-05"]),
    ("revenue by sales channel", "revenue_breakdown", []),
    ("Which product had the lowest revenue?", "custom_query", []),
    ("Monitor units in March", "custom_query", ["2026-03"]),
    ("tell me a joke", "unsupported", []),
])
def test_rule_interpreter(question, intent, months):
    got = RuleInterpreter().interpret(question, CTX)
    got_months = got.spec.months if got.spec else got.months
    assert (got.name, got_months) == (intent, months)


def test_may_as_verb_is_not_a_month():
    assert extract_months("may I see the total revenue", MONTHS) == []
    assert extract_months("revenue in may", MONTHS) == ["2026-05"]


def test_llm_failure_falls_back_to_rules():
    class BrokenLLM:
        def interpret(self, *_):
            raise TimeoutError("network down")

    interp = Interpreter(use_llm=False)
    interp.llm = BrokenLLM()
    intent, used = interp.interpret("What is the total revenue?", CTX)
    assert (intent.name, used) == ("total_revenue", "rules")


def test_llm_intent_is_used_when_available():
    class FakeLLM:
        def interpret(self, *_):
            return Intent("revenue_breakdown", dimension="category")

    interp = Interpreter(use_llm=False)
    interp.llm = FakeLLM()
    intent, used = interp.interpret("how are the categories doing?", CTX)
    assert (intent.name, intent.dimension, used) == ("revenue_breakdown", "category", "llm")


@pytest.mark.parametrize("question, metric, group_by, filters, order, limit", [
    ("How many units of Monitor were sold in the North?", "units", None,
     {"product": ["Monitor"], "region": ["North"]}, None, None),
    ("Top 3 products by units sold", "units", "product", {}, "desc", 3),
    ("Which region had the fewest orders?", "orders", "region", {}, "asc", None),
    ("Average order value by sales channel", "avg_order_value", "sales_channel", {}, "desc", None),
    ("Online vs Retail revenue", "revenue", "sales_channel", {"sales_channel": ["Online", "Retail"]}, "desc", None),
    ("How many returned orders?", "returned_orders", None, {}, None, None),
])
def test_rule_interpreter_builds_query_spec(question, metric, group_by, filters, order, limit):
    spec = RuleInterpreter().interpret(question, CTX).spec
    assert (spec.metric, spec.group_by, spec.filters, spec.order, spec.limit) == (metric, group_by, filters, order, limit)


@pytest.mark.parametrize("question", ["revenue from Tablet", "sales in Hyderabad"])
def test_unknown_terms_are_refused_not_ignored(question):
    intent = RuleInterpreter().interpret(question, CTX)
    assert intent.name == "unsupported" and "recognise" in intent.note


# --- Untrusted (LLM-written) SQL --------------------------------------------------------


def test_untrusted_sql_allows_normal_analytics(db):
    cols, rows = db.query("SELECT product, SUM(quantity) AS units FROM revenue_sales "
                          "GROUP BY product ORDER BY units DESC LIMIT 1", untrusted=True)
    assert rows == [["Wireless Mouse", 228]]


@pytest.mark.parametrize("sql", [
    "SELECT * FROM rejected_records",                                   # table not allowed
    "SELECT name FROM sqlite_master",                                   # schema snooping
    "SELECT load_extension('evil')",                                    # function not allowed
    "SELECT COUNT(*) FROM sales a, sales b, sales c, sales d, sales e", # runaway query
])
def test_untrusted_sql_restrictions(db, sql):
    with pytest.raises(UnsafeQueryError):
        db.query(sql, untrusted=True)


def test_answer_grounding_check():
    from app.analytics import answer_is_grounded
    cols, rows = ["product", "units"], [["Wireless Mouse", 228], ["Keyboard", 139]]
    assert answer_is_grounded("Wireless Mouse sold the most units (228), ahead of Keyboard (139).", cols, rows)
    assert not answer_is_grounded("Wireless Mouse sold 64% more than Keyboard.", cols, rows)

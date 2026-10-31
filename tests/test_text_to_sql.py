"""Text-to-SQL path, exercised with a fake LLM so the tests run offline."""

from app.analytics import Intent
from tests.conftest import ask


class FakeLLM:
    def __init__(self, sql, summary="unused"):
        self.sql, self.summary = sql, summary

    def interpret(self, question, ctx):
        return Intent("sql_query", sql=self.sql)

    def summarize(self, question, columns, rows):
        return self.summary


def use_llm(client, llm):
    client.app.state.interpreter.llm = llm


def test_generated_sql_is_executed_and_summary_kept_when_grounded(client):
    use_llm(client, FakeLLM(
        "SELECT region, COUNT(DISTINCT product) AS products FROM revenue_sales GROUP BY region ORDER BY region",
        summary="All 4 regions sold 7 different products each.",
    ))
    body = ask(client, "How many different products did each region sell?").json()
    assert body["intent"] == "sql_query" and body["interpreted_by"] == "llm"
    assert body["data"]["rows"] == [["East", 7], ["North", 7], ["South", 7], ["West", 7]]
    assert body["answer"].startswith("All 4 regions")
    assert any("AI-generated SQL" in a for a in body["assumptions"])


def test_ungrounded_summary_is_discarded(client):
    use_llm(client, FakeLLM("SELECT SUM(quantity) AS units FROM revenue_sales",
                            summary="We sold 999 units, up 12% on last year."))
    body = ask(client, "units overall?").json()
    assert "999" not in body["answer"]
    assert body["answer"] == f"Result: units = {body['data']['rows'][0][0]}."


def test_destructive_generated_sql_is_blocked_then_rules_fallback(client):
    use_llm(client, FakeLLM("DELETE FROM sales"))
    r = ask(client, "What is the total revenue?")
    assert r.status_code == 200
    body = r.json()
    assert body["interpreted_by"] == "rules" and body["intent"] == "total_revenue"
    assert body["data"]["rows"][0][0] == 1439200.0          # data still intact


def test_forbidden_table_in_generated_sql_is_rejected(client):
    use_llm(client, FakeLLM("SELECT * FROM rejected_records"))
    r = ask(client, "show me something odd")
    assert r.status_code == 422
    assert r.json()["error"]["rejected_sql"] == "SELECT * FROM rejected_records"

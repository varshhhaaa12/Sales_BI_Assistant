"""End-to-end tests through the HTTP API."""

import pytest

from tests.conftest import ask

# --- Valid questions (the five from the brief + extras) ------------------------------


def test_total_revenue(client, oracle):
    r = ask(client, "What is the total revenue?")
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "total_revenue"
    assert body["data"]["rows"][0][0] == pytest.approx(oracle["total"])
    assert "1,439,200.00" in body["answer"]
    assert body["sql"].startswith("SELECT")


def test_top_product(client, oracle):
    body = ask(client, "Which product generated the highest revenue?").json()
    expected = max(oracle["product"], key=oracle["product"].get)
    assert body["intent"] == "top_product"
    assert body["data"]["rows"][0][0] == expected == "Monitor"
    assert body["data"]["rows"][0][1] == pytest.approx(oracle["product"][expected])
    # Supporting data includes the full ranking, not just the winner.
    assert len(body["data"]["rows"]) == len(oracle["product"])


def test_monthly_trend(client, oracle):
    body = ask(client, "What is the monthly revenue trend?").json()
    assert body["intent"] == "monthly_trend"
    got = {row[0]: row[1] for row in body["data"]["rows"]}
    assert got == pytest.approx(oracle["month"])
    assert body["chart"]["type"] == "line"


def test_top_region(client, oracle):
    body = ask(client, "Which region generated the highest sales?").json()
    expected = max(oracle["region"], key=oracle["region"].get)
    assert body["intent"] == "top_region"
    assert body["data"]["rows"][0][0] == expected == "South"


def test_month_comparison(client, oracle):
    body = ask(client, "How did revenue change between March and April?").json()
    assert body["intent"] == "month_comparison"
    march, april = oracle["month"]["2026-03"], oracle["month"]["2026-04"]
    assert [row[1] for row in body["data"]["rows"]] == pytest.approx([march, april])
    assert "increased" in body["answer"] and f"{(april - march) / march * 100:+.1f}%" in body["answer"]


def test_month_filter_and_breakdown(client, oracle):
    body = ask(client, "What was the revenue in May?").json()
    assert body["data"]["rows"][0][0] == pytest.approx(oracle["month"]["2026-05"])
    body = ask(client, "Show revenue by category").json()
    assert body["intent"] == "revenue_breakdown"
    assert body["data"]["columns"][0] == "category"


# --- Invalid requests ----------------------------------------------------------------


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {}, {"question": "x" * 501}])
def test_invalid_payloads_return_400(client, payload):
    r = client.post("/api/ask", json=payload)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_non_json_body_returns_400(client):
    assert client.post("/api/ask", content="not json").status_code == 400


@pytest.mark.parametrize("question", [
    "What's the weather in Hyderabad?",
    "Predict next month's revenue",
    "What is our profit margin?",
    "tell me a joke",
    "How many different products did each region sell?",   # distinct count: not expressible offline
])
def test_unsupported_questions_are_refused(client, question):
    r = ask(client, question)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "unsupported_question"
    assert err["suggestions"]


@pytest.mark.parametrize("question", ["DROP TABLE sales", "delete all sales records", "update revenue to 0"])
def test_destructive_requests_are_refused_and_data_untouched(client, question, oracle):
    assert ask(client, question).status_code == 422
    total = ask(client, "What is the total revenue?").json()["data"]["rows"][0][0]
    assert total == pytest.approx(oracle["total"])


# --- Edge cases ----------------------------------------------------------------------


def test_comparison_without_two_months_asks_for_them(client):
    r = ask(client, "How did revenue change between two months?")
    assert r.status_code == 422
    assert "two months" in r.json()["error"]["message"]


def test_month_outside_data_range(client):
    r = ask(client, "Compare January and July")
    assert r.status_code == 422
    assert "No revenue data for July 2026" in r.json()["error"]["message"]


def test_data_quality_report(client):
    body = client.get("/api/data-quality").json()
    assert body["total_rows"] == 51
    assert body["loaded_rows"] == 49
    reasons = {r["transaction_id"]: r["reason"] for r in body["rejected"]}
    assert "Quantity" in reasons["TXN0034"]
    assert "Unit_price" in reasons["TXN0051"]
    assert body["loaded_by_status"] == {"Completed": 48, "Returned": 1}

import csv
from collections import defaultdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sales.csv"


@pytest.fixture
def client(tmp_path):
    """App wired to a throwaway database, rules-only so tests are deterministic and offline."""
    app = create_app(csv_path=CSV_PATH, db_path=tmp_path / "test.db", use_llm=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def oracle():
    """Expected figures computed in plain Python, independent of the app's SQL."""
    by_product, by_region, by_month = defaultdict(float), defaultdict(float), defaultdict(float)
    total = 0.0
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if r["Status"] != "Completed" or not r["Unit_price"] or int(r["Quantity"]) <= 0:
                continue
            rev = int(r["Quantity"]) * float(r["Unit_price"]) - float(r["Discount_amount"] or 0)
            d, m, y = r["Order_date"].split("-")
            total += rev
            by_product[r["Product"]] += rev
            by_region[r["Region"]] += rev
            by_month[f"{y}-{m}"] += rev
    return {"total": total, "product": dict(by_product), "region": dict(by_region), "month": dict(by_month)}


def ask(client, question):
    return client.post("/api/ask", json={"question": question})

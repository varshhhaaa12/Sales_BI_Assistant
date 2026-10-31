"""Data layer: load the CSV into SQLite, clean it, and run read-only queries.

Storage model
-------------
* ``sales``            - every row that passed validation (all statuses).
* ``rejected_records`` - rows that failed validation, with the raw values and a reason.
* ``revenue_sales``    - a VIEW over ``sales`` containing only revenue-bearing rows
                         (status = 'Completed') plus computed revenue columns.

Every analytics query reads from ``revenue_sales`` so the business rule
"what counts as revenue" lives in exactly one place.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REQUIRED_COLUMNS = [
    "Transaction_Id", "Order_date", "Product", "Category", "Region",
    "Sales_channel", "Quantity", "Unit_price", "Discount_amount", "Status",
]
KNOWN_STATUSES = {"Completed", "Returned", "Cancelled", "Pending"}

SCHEMA = """
DROP VIEW IF EXISTS revenue_sales;
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS rejected_records;

CREATE TABLE sales (
    transaction_id  TEXT PRIMARY KEY,
    order_date      TEXT NOT NULL,          -- ISO 8601 (YYYY-MM-DD)
    order_month     TEXT NOT NULL,          -- YYYY-MM, pre-computed for grouping
    product         TEXT NOT NULL,
    category        TEXT NOT NULL,
    region          TEXT NOT NULL,
    sales_channel   TEXT NOT NULL,
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    unit_price      REAL    NOT NULL CHECK (unit_price > 0),
    discount_amount REAL    NOT NULL DEFAULT 0 CHECK (discount_amount >= 0),
    status          TEXT NOT NULL
);
CREATE INDEX idx_sales_month ON sales(order_month);

CREATE TABLE rejected_records (
    row_number     INTEGER NOT NULL,        -- line number in the CSV (header = 1)
    transaction_id TEXT,
    reason         TEXT NOT NULL,
    raw            TEXT NOT NULL            -- original row as JSON
);

-- Net revenue = quantity * unit_price - discount, for completed orders only.
CREATE VIEW revenue_sales AS
SELECT *,
       quantity * unit_price                   AS gross_revenue,
       quantity * unit_price - discount_amount AS net_revenue
FROM sales
WHERE status = 'Completed';
"""


class DataLoadError(Exception):
    """The CSV cannot be loaded at all (missing file, wrong columns)."""


class UnsafeQueryError(Exception):
    """A query was rejected by the read-only guard."""


@dataclass
class LoadReport:
    total_rows: int = 0
    loaded_rows: int = 0
    rejected: list[dict] = field(default_factory=list)


def _clean_row(raw: dict[str, str], seen_ids: set[str]) -> tuple[dict | None, str | None]:
    """Validate one CSV row. Returns (clean_row, None) or (None, reason)."""
    row = {k: (v or "").strip() for k, v in raw.items()}

    txn_id = row["Transaction_Id"]
    if not txn_id:
        return None, "missing Transaction_Id"
    if txn_id in seen_ids:
        return None, "duplicate Transaction_Id"

    for col in ("Product", "Category", "Region", "Sales_channel", "Status"):
        if not row[col]:
            return None, f"missing {col}"

    try:
        order_date = datetime.strptime(row["Order_date"], "%d-%m-%Y").date()
    except ValueError:
        return None, f"invalid Order_date '{row['Order_date']}' (expected DD-MM-YYYY)"

    try:
        quantity = int(row["Quantity"])
    except ValueError:
        return None, f"invalid Quantity '{row['Quantity']}'"
    if quantity <= 0:
        return None, f"non-positive Quantity ({quantity})"

    if not row["Unit_price"]:
        return None, "missing Unit_price"
    try:
        unit_price = float(row["Unit_price"])
    except ValueError:
        return None, f"invalid Unit_price '{row['Unit_price']}'"
    if unit_price <= 0:
        return None, f"non-positive Unit_price ({unit_price})"

    try:
        discount = float(row["Discount_amount"] or 0)
    except ValueError:
        return None, f"invalid Discount_amount '{row['Discount_amount']}'"
    if discount < 0:
        return None, "negative Discount_amount"
    if discount > quantity * unit_price:
        return None, "Discount_amount exceeds order value"

    status = row["Status"].title()
    if status not in KNOWN_STATUSES:
        return None, f"unknown Status '{row['Status']}'"

    return {
        "transaction_id": txn_id,
        "order_date": order_date.isoformat(),
        "order_month": order_date.strftime("%Y-%m"),
        "product": row["Product"],
        "category": row["Category"],
        "region": row["Region"].title(),
        "sales_channel": row["Sales_channel"].title(),
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_amount": discount,
        "status": status,
    }, None


def load_csv(csv_path: Path, db_path: Path) -> LoadReport:
    """(Re)build the SQLite database from the CSV. Safe to call repeatedly."""
    if not csv_path.exists():
        raise DataLoadError(f"CSV not found: {csv_path}")

    report = LoadReport()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DataLoadError(f"CSV is missing columns: {missing}")

        clean_rows, seen_ids = [], set()
        for line_no, raw in enumerate(reader, start=2):
            report.total_rows += 1
            clean, reason = _clean_row(raw, seen_ids)
            if clean:
                seen_ids.add(clean["transaction_id"])
                clean_rows.append(clean)
            else:
                report.rejected.append({
                    "row_number": line_no,
                    "transaction_id": (raw.get("Transaction_Id") or "").strip() or None,
                    "reason": reason,
                    "raw": json.dumps(raw),
                })

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO sales VALUES (:transaction_id, :order_date, :order_month, :product, "
            ":category, :region, :sales_channel, :quantity, :unit_price, :discount_amount, :status)",
            clean_rows,
        )
        conn.executemany(
            "INSERT INTO rejected_records VALUES (:row_number, :transaction_id, :reason, :raw)",
            report.rejected,
        )
    report.loaded_rows = len(clean_rows)
    return report


# --- Read-only query execution ------------------------------------------------

_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_LEADING_KEYWORD = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

# Extra limits applied to SQL we did not write ourselves (LLM-generated).
UNTRUSTED_TABLES = {"sales", "revenue_sales"}
UNTRUSTED_FUNCTIONS = {
    "sum", "avg", "count", "min", "max", "total", "round", "abs", "coalesce", "ifnull",
    "nullif", "iif", "strftime", "date", "julianday", "substr", "lower", "upper", "trim",
    "length", "instr", "printf", "cast", "group_concat",
}
MAX_ROWS = 200
MAX_VM_STEPS = 5_000_000      # aborts runaway queries (e.g. huge cross joins)


def _authorizer(action, *_):
    """SQLite calls this for every operation while compiling a statement."""
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def _strict_authorizer(action, arg1, arg2, *_):
    """Read-only AND limited to known tables/functions - used for untrusted SQL."""
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 in UNTRUSTED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in UNTRUSTED_FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def validate_sql(sql: str) -> None:
    """Cheap static checks before SQLite even sees the query."""
    stripped = sql.strip().rstrip(";")
    if not stripped:
        raise UnsafeQueryError("empty query")
    if ";" in stripped:
        raise UnsafeQueryError("multiple statements are not allowed")
    if not _LEADING_KEYWORD.match(stripped):
        raise UnsafeQueryError("only SELECT queries are allowed")


class ReadOnlyDB:
    """Opens the database read-only, with an authorizer as a second safety layer.

    Three independent layers stop destructive operations:
      1. ``validate_sql`` - single statement, must start with SELECT/WITH.
      2. ``mode=ro`` URI - SQLite refuses any write at the file level.
      3. ``set_authorizer`` - denies anything except SELECT/READ/FUNCTION.
    With ``untrusted=True`` (LLM-written SQL) the authorizer additionally allows
    only the sales tables and a fixed set of functions, and results are capped
    in rows and execution steps.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def query(self, sql: str, params: dict | tuple = (), untrusted: bool = False) -> tuple[list[str], list[list]]:
        validate_sql(sql)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            conn.set_authorizer(_strict_authorizer if untrusted else _authorizer)
            if untrusted:
                steps = {"n": 0}

                def watchdog():
                    steps["n"] += 1000
                    return 1 if steps["n"] > MAX_VM_STEPS else 0   # non-zero aborts the query

                conn.set_progress_handler(watchdog, 1000)
            try:
                cur = conn.execute(sql, params)
                columns = [d[0] for d in cur.description]
                rows = cur.fetchmany(MAX_ROWS) if untrusted else cur.fetchall()
            except sqlite3.DatabaseError as exc:
                msg = str(exc)
                if "not authorized" in msg or "prohibited" in msg or "readonly" in msg:
                    raise UnsafeQueryError(msg) from exc
                if "interrupted" in msg:
                    raise UnsafeQueryError("query took too long and was stopped") from exc
                raise
            return columns, [list(r) for r in rows]
        finally:
            conn.close()

    def scalar_list(self, sql: str) -> list:
        _, rows = self.query(sql)
        return [r[0] for r in rows]

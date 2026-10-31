"""Analytics: turns a validated ``Intent`` into SQL, runs it, and phrases the answer.

Three levels, from most to least constrained:

1. Named intents (total revenue, top product, trend, ...) - hand-written SQL.
2. ``custom_query`` - a ``QuerySpec`` (metric x filters x grouping x ordering)
   compiled to SQL from whitelisted fragments only. Covers most ad-hoc questions.
3. ``sql_query`` - LLM-written SQL for the long tail, executed in the strict
   untrusted mode of ``ReadOnlyDB`` (allowed tables/functions, row + time caps).

Numbers are only ever produced by SQLite. The interpreter decides *what* to run;
it never computes or invents figures itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from .db import ReadOnlyDB

IntentName = Literal[
    "total_revenue", "top_product", "monthly_trend", "top_region",
    "month_comparison", "revenue_breakdown", "custom_query", "sql_query", "unsupported",
]
Dimension = Literal["product", "region", "category", "sales_channel"]

# Column names come from this whitelist only; they are never taken from user text.
DIMENSIONS: dict[str, str] = {
    "product": "product",
    "region": "region",
    "category": "category",
    "sales_channel": "sales_channel",
}

# metric name -> (SQL aggregate, human label, is_money)
METRICS: dict[str, tuple[str, str, bool]] = {
    "revenue": ("ROUND(SUM(net_revenue), 2)", "net revenue", True),
    "gross_revenue": ("ROUND(SUM(gross_revenue), 2)", "gross revenue", True),
    "units": ("SUM(quantity)", "units sold", False),
    "orders": ("COUNT(*)", "number of orders", False),
    "avg_order_value": ("ROUND(AVG(net_revenue), 2)", "average order value", True),
    "discount": ("ROUND(SUM(discount_amount), 2)", "total discount", True),
    "returned_orders": ("COUNT(*)", "number of returned orders", False),
    "returned_value": ("ROUND(SUM(quantity * unit_price - discount_amount), 2)", "value of returned orders", True),
}
# Return metrics read returned rows from the base table; everything else reads completed orders.
METRIC_SOURCE = {"returned_orders": "sales WHERE status = 'Returned'",
                 "returned_value": "sales WHERE status = 'Returned'"}
GROUPABLE = {**DIMENSIONS, "month": "order_month"}

REVENUE_DEFINITION = (
    "Revenue = quantity x unit_price - discount_amount, counting only orders with "
    "status 'Completed' (returned orders and invalid records are excluded)."
)


@dataclass
class QuerySpec:
    """A structured description of an aggregate query. Every field is whitelisted."""
    metric: str = "revenue"
    group_by: str | None = None                                  # a DIMENSIONS key or "month"
    filters: dict[str, list[str]] = field(default_factory=dict)  # dimension -> allowed values
    months: list[str] = field(default_factory=list)              # 'YYYY-MM' filter
    order: Literal["desc", "asc"] | None = None                  # None = natural order
    limit: int | None = None


@dataclass
class Intent:
    name: IntentName
    dimension: Dimension | None = None
    months: list[str] = field(default_factory=list)   # 'YYYY-MM'
    spec: QuerySpec | None = None                     # for custom_query
    sql: str | None = None                            # for sql_query (LLM-written)
    note: str | None = None                           # why a question is unsupported


@dataclass
class AnalyticsResult:
    answer: str
    columns: list[str]
    rows: list[list]
    sql: str
    params: dict
    chart: dict | None = None
    assumptions: list[str] = field(default_factory=list)


class AnalyticsError(Exception):
    """The question was understood but cannot be answered from the data."""


def fmt_money(value: float) -> str:
    return f"₹{value:,.2f}"


def month_label(month: str) -> str:
    return datetime.strptime(month, "%Y-%m").strftime("%B %Y")


def fmt_metric(metric: str, value) -> str:
    if value is None:
        return "n/a"
    return fmt_money(value) if METRICS[metric][2] else f"{value:,.0f}"


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in _NUMBER_RE.findall(text)]


def answer_is_grounded(answer: str, columns: list[str], rows: list[list]) -> bool:
    """True if every number in ``answer`` appears in the query result.

    Used to stop an LLM-written summary from stating figures (or computing new
    ones, like percentages) that the database did not return.
    """
    allowed: list[float] = [float(len(rows))]
    for cell in [c for r in rows for c in r] + columns:
        if isinstance(cell, (int, float)):
            allowed.append(float(cell))
        elif isinstance(cell, str):
            allowed.extend(_numbers(cell.replace("-", " ")))
    return all(any(abs(n - a) <= max(0.01, abs(a) * 0.0005) for a in allowed)
               for n in _numbers(answer.replace("₹", "")))


class Analytics:
    def __init__(self, db: ReadOnlyDB):
        self.db = db

    def available_months(self) -> list[str]:
        return self.db.scalar_list("SELECT DISTINCT order_month FROM revenue_sales ORDER BY 1")

    def run(self, intent: Intent) -> AnalyticsResult:
        handler = {
            "total_revenue": self.total_revenue,
            "top_product": lambda i: self.top_by(i, "product"),
            "top_region": lambda i: self.top_by(i, "region"),
            "revenue_breakdown": lambda i: self.top_by(i, i.dimension or "product"),
            "monthly_trend": self.monthly_trend,
            "month_comparison": self.month_comparison,
            "custom_query": self.custom_query,
            "sql_query": self.sql_query,
        }.get(intent.name)
        if handler is None:
            raise AnalyticsError(f"Unsupported intent: {intent.name}")
        result = handler(intent)
        result.assumptions.insert(0, REVENUE_DEFINITION)
        return result

    def dimension_values(self) -> dict[str, list[str]]:
        """Distinct values per dimension, used to recognise entities in questions."""
        return {d: self.db.scalar_list(f"SELECT DISTINCT {col} FROM sales ORDER BY 1")
                for d, col in DIMENSIONS.items()}

    # -- helpers ---------------------------------------------------------------

    def _check_months(self, months: list[str]) -> None:
        available = self.available_months()
        missing = [m for m in months if m not in available]
        if missing:
            have = ", ".join(month_label(m) for m in available)
            raise AnalyticsError(
                f"No revenue data for {', '.join(month_label(m) for m in missing)}. "
                f"Data is available for: {have}."
            )

    def _month_filter(self, intent: Intent) -> tuple[str, dict, str]:
        """Optional single-month filter for total/top-N questions."""
        if len(intent.months) == 1:
            self._check_months(intent.months)
            m = intent.months[0]
            return "WHERE order_month = :month", {"month": m}, f" in {month_label(m)}"
        return "", {}, ""

    # -- intents ---------------------------------------------------------------

    def total_revenue(self, intent: Intent) -> AnalyticsResult:
        where, params, scope = self._month_filter(intent)
        sql = (
            "SELECT ROUND(SUM(net_revenue), 2) AS net_revenue, "
            "ROUND(SUM(gross_revenue), 2) AS gross_revenue, "
            "ROUND(SUM(discount_amount), 2) AS total_discount, "
            "COUNT(*) AS orders, SUM(quantity) AS units "
            f"FROM revenue_sales {where}"
        )
        cols, rows = self.db.query(sql, params)
        net = rows[0][0] or 0
        answer = (
            f"Total net revenue{scope} is {fmt_money(net)} from {rows[0][3]} completed orders "
            f"(gross {fmt_money(rows[0][1] or 0)} minus {fmt_money(rows[0][2] or 0)} in discounts)."
        )
        return AnalyticsResult(answer, cols, rows, sql, params)

    def top_by(self, intent: Intent, dimension: str) -> AnalyticsResult:
        col = DIMENSIONS[dimension]
        where, params, scope = self._month_filter(intent)
        sql = (
            f"SELECT {col}, ROUND(SUM(net_revenue), 2) AS net_revenue, "
            "SUM(quantity) AS units, COUNT(*) AS orders "
            f"FROM revenue_sales {where} GROUP BY {col} ORDER BY net_revenue DESC, {col}"
        )
        cols, rows = self.db.query(sql, params)
        if not rows:
            raise AnalyticsError("No revenue data matched this question.")
        label = dimension.replace("_", " ")
        top_value = rows[0][1]
        leaders = [r[0] for r in rows if r[1] == top_value]
        total = sum(r[1] for r in rows)
        share = top_value / total * 100 if total else 0
        if intent.name == "revenue_breakdown":
            answer = f"Net revenue by {label}{scope}: {rows[0][0]} leads with {fmt_money(top_value)} ({share:.1f}% of total)."
        elif len(leaders) > 1:
            answer = f"Tie for highest revenue {label}{scope}: {', '.join(leaders)} with {fmt_money(top_value)} each."
        else:
            answer = (
                f"{rows[0][0]} is the highest-revenue {label}{scope} with {fmt_money(top_value)} "
                f"({share:.1f}% of total net revenue)."
            )
        assumptions = [f"{label.capitalize()}s are ranked by net revenue, not units sold."]
        return AnalyticsResult(answer, cols, rows, sql, params,
                               chart={"type": "bar", "x": col, "y": "net_revenue"},
                               assumptions=assumptions)

    def monthly_trend(self, intent: Intent) -> AnalyticsResult:
        sql = (
            "SELECT order_month AS month, ROUND(SUM(net_revenue), 2) AS net_revenue, "
            "COUNT(*) AS orders, SUM(quantity) AS units "
            "FROM revenue_sales GROUP BY order_month ORDER BY order_month"
        )
        cols, rows = self.db.query(sql)
        if not rows:
            raise AnalyticsError("No revenue data available.")
        # Month-over-month % change is derived deterministically from the query output.
        cols = cols + ["mom_change_pct"]
        prev = None
        for r in rows:
            r.append(None if prev in (None, 0) else round((r[1] - prev) / prev * 100, 1))
            prev = r[1]
        best = max(rows, key=lambda r: r[1])
        worst = min(rows, key=lambda r: r[1])
        first, last = rows[0], rows[-1]
        direction = "up" if last[1] > first[1] else "down" if last[1] < first[1] else "flat"
        answer = (
            f"Revenue across {len(rows)} months ({month_label(first[0])} – {month_label(last[0])}) "
            f"ended {direction} at {fmt_money(last[1])} vs {fmt_money(first[1])} at the start. "
            f"Peak: {month_label(best[0])} ({fmt_money(best[1])}); "
            f"lowest: {month_label(worst[0])} ({fmt_money(worst[1])})."
        )
        return AnalyticsResult(answer, cols, rows, sql, {},
                               chart={"type": "line", "x": "month", "y": "net_revenue"})

    def month_comparison(self, intent: Intent) -> AnalyticsResult:
        if len(intent.months) != 2:
            raise AnalyticsError("Please name exactly two months to compare, e.g. 'March vs April'.")
        a, b = sorted(intent.months)
        if a == b:
            raise AnalyticsError("Please name two different months to compare.")
        self._check_months([a, b])
        sql = (
            "SELECT order_month AS month, ROUND(SUM(net_revenue), 2) AS net_revenue, "
            "COUNT(*) AS orders, SUM(quantity) AS units "
            "FROM revenue_sales WHERE order_month IN (:month_a, :month_b) "
            "GROUP BY order_month ORDER BY order_month"
        )
        params = {"month_a": a, "month_b": b}
        cols, rows = self.db.query(sql, params)
        va, vb = rows[0][1], rows[1][1]
        diff = vb - va
        pct = diff / va * 100 if va else None
        verb = "increased" if diff > 0 else "decreased" if diff < 0 else "did not change"
        answer = f"Revenue {verb} from {fmt_money(va)} in {month_label(a)} to {fmt_money(vb)} in {month_label(b)}"
        if diff:
            answer += f", a change of {'+' if diff > 0 else '−'}{fmt_money(abs(diff))}"
            if pct is not None:
                answer += f" ({pct:+.1f}%)"
        answer += "."
        return AnalyticsResult(answer, cols, rows, sql, params,
                               chart={"type": "bar", "x": "month", "y": "net_revenue"})

    # -- flexible queries ---------------------------------------------------------

    def _validate_spec(self, spec: QuerySpec) -> QuerySpec:
        if spec.metric not in METRICS:
            raise AnalyticsError(f"Unknown metric '{spec.metric}'.")
        if spec.group_by is not None and spec.group_by not in GROUPABLE:
            raise AnalyticsError(f"Cannot group by '{spec.group_by}'.")
        if spec.limit is not None and not 1 <= spec.limit <= 50:
            raise AnalyticsError("Limit must be between 1 and 50.")
        known = self.dimension_values()
        clean_filters = {}
        for dim, values in spec.filters.items():
            if dim not in DIMENSIONS:
                raise AnalyticsError(f"Cannot filter by '{dim}'.")
            lookup = {v.lower(): v for v in known[dim]}
            unknown = [v for v in values if v.lower() not in lookup]
            if unknown:
                raise AnalyticsError(
                    f"Unknown {dim.replace('_', ' ')} {unknown}. Known values: {', '.join(known[dim])}."
                )
            clean_filters[dim] = [lookup[v.lower()] for v in values]
        if spec.months:
            self._check_months(spec.months)
        return QuerySpec(spec.metric, spec.group_by, clean_filters, sorted(set(spec.months)),
                         spec.order, spec.limit)

    @staticmethod
    def _scope(spec: QuerySpec) -> str:
        parts = [" / ".join(v) for v in spec.filters.values()]
        if spec.months:
            parts.append(" and ".join(month_label(m) for m in spec.months))
        return f" for {', '.join(parts)}" if parts else ""

    def custom_query(self, intent: Intent) -> AnalyticsResult:
        spec = self._validate_spec(intent.spec or QuerySpec())
        agg, label, _ = METRICS[spec.metric]

        where, params = [], {}
        for dim, values in spec.filters.items():
            names = [f"{dim}_{i}" for i in range(len(values))]
            params.update(zip(names, values))
            where.append(f"{DIMENSIONS[dim]} IN ({', '.join(':' + n for n in names)})")
        if spec.months:
            names = [f"month_{i}" for i in range(len(spec.months))]
            params.update(zip(names, spec.months))
            where.append(f"order_month IN ({', '.join(':' + n for n in names)})")
        source = METRIC_SOURCE.get(spec.metric, "revenue_sales")
        if " WHERE " in source:
            source, base_filter = source.split(" WHERE ")
            where.insert(0, base_filter)
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        extra = "" if spec.metric in ("orders", "returned_orders") else ", COUNT(*) AS orders"

        if spec.group_by:
            col = GROUPABLE[spec.group_by]
            alias = "month" if spec.group_by == "month" else col
            order = spec.order or ("asc" if spec.group_by == "month" else "desc")
            order_sql = (f"{alias} ASC" if spec.order is None and spec.group_by == "month"
                         else f"{spec.metric} {order.upper()}, {alias}")
            sql = (f"SELECT {col} AS {alias}, {agg} AS {spec.metric}{extra} FROM {source}"
                   f"{where_sql} GROUP BY {col} ORDER BY {order_sql}")
            if spec.limit:
                sql += f" LIMIT {int(spec.limit)}"
        else:
            sql = f"SELECT {agg} AS {spec.metric}{extra} FROM {source}{where_sql}"

        cols, rows = self.db.query(sql, params)
        scope = self._scope(QuerySpec(filters={d: v for d, v in spec.filters.items() if d != spec.group_by},
                                      months=spec.months if spec.group_by != "month" else []))
        assumptions = [f"Metric: {label}."]

        if not spec.group_by:
            value = rows[0][0]
            if value is None or (extra and rows[0][-1] == 0):
                answer = f"There are no matching orders{scope}, so the {label} is 0."
            else:
                answer = f"The {label}{scope} is {fmt_metric(spec.metric, value)}"
                n = rows[0][-1]
                answer += f" across {n} order{'s' if n != 1 else ''}." if extra else "."
            return AnalyticsResult(answer, cols, rows, sql, params, assumptions=assumptions)

        if not rows:
            raise AnalyticsError(f"No completed orders match this question{scope}.")
        group_label = spec.group_by.replace("_", " ")
        chart = {"type": "line" if spec.group_by == "month" and spec.order is None else "bar",
                 "x": cols[0], "y": spec.metric}
        if spec.order is None and spec.group_by == "month":
            listing = "; ".join(f"{month_label(r[0])}: {fmt_metric(spec.metric, r[1])}" for r in rows)
            answer = f"{label.capitalize()} by month{scope}: {listing}."
        else:
            best = "lowest" if spec.order == "asc" else "highest"
            first = rows[0]
            name = month_label(first[0]) if spec.group_by == "month" else first[0]
            answer = f"{name} has the {best} {label}{scope}: {fmt_metric(spec.metric, first[1])}."
            if 1 < len(rows) <= 8:
                answer += " Ranking: " + ", ".join(
                    f"{r[0]} ({fmt_metric(spec.metric, r[1])})" for r in rows) + "."
            ties = [r[0] for r in rows[1:] if r[1] == first[1]]
            if ties:
                answer += f" (Tied with {', '.join(map(str, ties))}.)"
        return AnalyticsResult(answer, cols, rows, sql, params, chart=chart, assumptions=assumptions)

    def sql_query(self, intent: Intent) -> AnalyticsResult:
        """Run LLM-written SQL in strict untrusted mode. The answer text is filled in by the caller."""
        if not intent.sql:
            raise AnalyticsError("No query was generated for this question.")
        cols, rows = self.db.query(intent.sql, untrusted=True)
        if not rows:
            answer = "The query ran successfully but returned no rows."
        elif len(rows) == 1 and len(cols) == 1:
            answer = f"Result: {cols[0]} = {rows[0][0]}."
        else:
            answer = f"The query returned {len(rows)} row(s); see the supporting data."
        return AnalyticsResult(answer, cols, rows, intent.sql, {}, assumptions=[
            "This question was answered with AI-generated SQL. It was validated (read-only, "
            "allowed tables and functions only, row and time limits), but please check the SQL below.",
        ])

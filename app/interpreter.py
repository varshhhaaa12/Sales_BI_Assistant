"""Turn a natural-language question into a validated ``Intent``.

Two interpreters share one contract:

* ``RuleInterpreter`` - deterministic parsing. It recognises the metric (revenue,
  units, orders, ...), entities that exist in the data (products, regions,
  categories, channels, months), grouping and highest/lowest/top-N. Always
  available, used by the tests, and used as the fallback.
* ``LLMInterpreter``  - asks the configured LLM (any provider, see ``app/llm.py``) to
  plan the query as JSON matching ``LLMPlan``.
  It prefers the named intents or a ``QuerySpec``; only for questions those
  cannot express does it write SQL, which then runs in strict untrusted mode.
  Every plan is re-validated before use.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .analytics import Intent, QuerySpec
from .llm import LLMClient, LLMConfig, LLMError

log = logging.getLogger(__name__)

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_NAME_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b(?:\s*,?\s*(\d{4}))?")
_ISO_MONTH_RE = re.compile(r"\b(\d{4})-(\d{1,2})\b")
_MONTH_FORMAT_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_LIMIT_RE = re.compile(r"\b(?:top|bottom|best|worst|first)\s+(\d{1,2})\b")

COMPARE_WORDS = ("compare", "comparison", "change", "changed", " vs", "versus", "between",
                 "difference", "differ", "growth", "grow", "increase", "decrease",
                 "higher than", "lower than", "more than", "less than")
TREND_WORDS = ("trend", "monthly", "month over month", "month-over-month", "month by month",
               "each month", "per month", "by month", "over time", "every month", "month wise", "month-wise")
TOP_WORDS = ("highest", "top", "best", "most", "largest", "biggest", "maximum", "leading", "max ", "greatest")
BOTTOM_WORDS = ("lowest", "least", "worst", "smallest", "minimum", "bottom", "fewest", "weakest")
WRITE_WORDS = re.compile(r"\b(delete|drop|update|insert|alter|truncate|remove|modify|replace|create)\b")
# Things this dataset cannot answer - refuse instead of returning a nearby figure.
OUT_OF_SCOPE = re.compile(r"\b(predict|forecast|next (month|year|quarter)|future|profit|margin|cost|"
                          r"customer|employee|salary|weather|stock price)s?\b"
                          # distinct counts and ratios are not in the QuerySpec - the LLM path can do them
                          r"|\b(different|distinct|unique)\b|\b(ratio|percentage|percent|share|median)\b")
REVENUE_WORDS = ("revenue", "sales", "sale", "earn", "income", "turnover", "money", "perform", "value")
MEASURE_HINTS = ("how much", "how many", "total", "overall", "sum", "count", "number of")
DIMENSION_WORDS = {
    "product": ("product", "item"),
    "region": ("region", "area", "zone", "territory"),
    "category": ("category", "categories"),
    "sales_channel": ("channel",),
}


# Words that may follow "for/of/from/in/..." without naming a data value.
KNOWN_WORDS = set(MONTHS) | {
    "the", "a", "an", "all", "each", "every", "our", "my", "this", "that", "total", "overall", "terms",
    "revenue", "sales", "sale", "units", "unit", "orders", "order", "month", "months", "period", "data",
    "dataset", "business", "company", "products", "product", "regions", "region", "categories",
    "category", "channels", "channel", "sales_channel", "quantity", "discount", "discounts", "value",
    "it", "them", "which", "what", "2026", "first", "last", "half", "year", "comparison", "average",
    "gross", "net", "returned", "returns", "units", "items", "transactions", "money", "general",
}
_PREP_TERM_RE = re.compile(r"\b(?:for|of|from|in|on|at|about)\s+(?:the\s+|our\s+)?([a-z][a-z\-]+)")


@dataclass
class DataContext:
    """What exists in the data, so questions can be matched against real values."""
    months: list[str]
    values: dict[str, list[str]]   # dimension -> distinct values


def extract_months(question: str, available_months: list[str]) -> list[str]:
    """Find months in the text and normalise them to 'YYYY-MM'.

    When no year is given, use the year in which that month exists in the data
    (falling back to the latest data year).
    """
    q = question.lower()
    years_in_data = sorted({m[:4] for m in available_months}) or ["2026"]
    found: list[str] = []

    for y, mo in _ISO_MONTH_RE.findall(q):
        if 1 <= int(mo) <= 12:
            found.append(f"{y}-{int(mo):02d}")

    for name, year in _MONTH_NAME_RE.findall(q):
        # "may" is also a verb - only treat it as a month when it looks like one.
        if name == "may" and re.search(r"\bmay\s+(i|we|you|be|have|it|need|want)\b", q):
            continue
        num = MONTHS[name]
        if year:
            found.append(f"{year}-{num:02d}")
        else:
            match = [m for m in available_months if int(m[5:]) == num]
            found.append(match[0] if match else f"{years_in_data[-1]}-{num:02d}")

    # De-duplicate while keeping order of appearance.
    return list(dict.fromkeys(found))


def extract_entities(question: str, values: dict[str, list[str]]) -> dict[str, list[str]]:
    """Find known data values (e.g. 'Monitor', 'North', 'Online') mentioned in the text."""
    q = question.lower()
    found: dict[str, list[str]] = {}
    for dim, options in values.items():
        hits = [v for v in options if re.search(rf"\b{re.escape(v.lower())}(e?s)?\b", q)]
        if hits:
            found[dim] = hits
    return found


def detect_metric(q: str) -> str | None:
    if re.search(r"\breturn(s|ed)?\b", q):
        return "returned_value" if re.search(r"\b(value|revenue|amount|worth|how much)\b", q) else "returned_orders"
    if re.search(r"\b(average|avg|mean)\b.*\b(order|sale|revenue|transaction)|\baov\b", q):
        return "avg_order_value"
    if "discount" in q:
        return "discount"
    if "gross" in q:
        return "gross_revenue"
    if re.search(r"\b(units?|quantity|quantities|pieces|volume)\b|\bhow many\b.*\b(sold|items)\b"
                 r"|\bitems sold\b|\bsold the (most|least)\b|\bsells? the (most|least)\b", q):
        return "units"
    if re.search(r"\b(orders?|transactions?)\b", q) and re.search(
            r"\b(how many|number of|count|most|least|fewest|highest|lowest)\b|\btransactions\b", q):
        return "orders"
    if any(w in q for w in REVENUE_WORDS):
        return "revenue"
    return None


def unknown_terms(q: str, ctx: "DataContext") -> list[str]:
    """Words after a preposition that are neither keywords nor values in the data.

    Catches "revenue from Tablet" or "sales in Hyderabad", which would otherwise
    silently fall back to an unfiltered total - a wrong answer that looks right.
    """
    known = KNOWN_WORDS | {w for vals in ctx.values.values() for v in vals for w in v.lower().split()}
    known |= {w for words in DIMENSION_WORDS.values() for w in words}
    out = []
    for term in _PREP_TERM_RE.findall(q):
        if term not in known and term.rstrip("s") not in known and not term.rstrip("es") in known:
            out.append(term)
    return out


def _has(q: str, words) -> bool:
    return any(w in q for w in words)


class RuleInterpreter:
    name = "rules"

    def interpret(self, question: str, ctx: DataContext) -> Intent:
        q = f" {question.lower().strip()} "
        if WRITE_WORDS.search(q) or OUT_OF_SCOPE.search(q):
            # Read-only assistant; and forecasts/profit etc. are not in the data.
            return Intent("unsupported")

        unknown = unknown_terms(q, ctx)
        if unknown:
            known = "; ".join(f"{d.replace('_', ' ')}s: {', '.join(v)}" for d, v in ctx.values.items())
            return Intent("unsupported", note=f"I don't recognise '{unknown[0]}' in the sales data. "
                                              f"Known values - {known}.")

        months = extract_months(q, ctx.months)
        entities = extract_entities(q, ctx.values)
        metric = detect_metric(q)
        top, bottom = _has(q, TOP_WORDS), _has(q, BOTTOM_WORDS)
        trend, compare = _has(q, TREND_WORDS), _has(q, COMPARE_WORDS)
        limit_match = _LIMIT_RE.search(q)
        limit = int(limit_match.group(1)) if limit_match else None

        # A dimension is a grouping candidate if named ("which region") but not pinned to
        # one value, or if several of its values are named ("online vs retail").
        named_dims = [d for d, words in DIMENSION_WORDS.items() if _has(q, words)]
        group_dims = [d for d in named_dims if len(entities.get(d, [])) != 1]
        group_dims += [d for d, v in entities.items() if len(v) > 1 and d not in group_dims]
        if re.search(r"\b(which|what) month\b", q) or re.search(r"\bmonth\b", q) and (top or bottom):
            group_dims.append("month")

        has_measure = metric is not None or top or bottom or trend or compare or _has(q, MEASURE_HINTS)
        about_data = metric is not None or entities or named_dims or months or "month" in group_dims
        if not (has_measure and about_data):
            return Intent("unsupported")

        # 1. Plain revenue questions keep their purpose-built answers.
        if metric in (None, "revenue") and not entities and not bottom and limit is None:
            if trend and len(months) < 2 and not group_dims:
                return Intent("monthly_trend")
            if compare and not group_dims:
                # With fewer than two months the analytics layer asks the user to name them.
                return Intent("month_comparison", months=months)
            if len(months) <= 1 and len(group_dims) == 1 and group_dims[0] != "month":
                dim = group_dims[0]
                if top and dim in ("product", "region"):
                    return Intent(f"top_{dim}", months=months)
                return Intent("revenue_breakdown", dimension=dim, months=months)
            if len(months) <= 1 and not group_dims and not top:
                return Intent("total_revenue", months=months)

        # 2. Everything else becomes a flexible, whitelisted query spec.
        group_by = group_dims[0] if group_dims else None
        if group_by is None and (trend or (compare and len(months) >= 2)):
            group_by = "month"
        order = None
        if bottom:
            order = "asc"
        elif top or (group_by and group_by != "month"):
            order = "desc"
        filters = {d: v for d, v in entities.items() if d != group_by or len(v) > 1}
        return Intent("custom_query", spec=QuerySpec(
            metric=metric or "revenue", group_by=group_by, filters=filters,
            months=months, order=order, limit=limit,
        ))


# --- LLM interpreter ------------------------------------------------------------

DimensionName = Literal["product", "region", "category", "sales_channel"]


class LLMFilter(BaseModel):
    dimension: DimensionName
    value: str


class LLMPlan(BaseModel):
    intent: Literal["total_revenue", "top_product", "monthly_trend", "top_region",
                    "month_comparison", "revenue_breakdown", "custom_query", "sql_query", "unsupported"]
    dimension: Optional[DimensionName] = None
    months: list[str] = Field(default_factory=list, description="Months as YYYY-MM")
    metric: Optional[Literal["revenue", "gross_revenue", "units", "orders", "avg_order_value", "discount",
                             "returned_orders", "returned_value"]] = None
    group_by: Optional[Literal["product", "region", "category", "sales_channel", "month"]] = None
    filters: list[LLMFilter] = Field(default_factory=list)
    order: Optional[Literal["desc", "asc"]] = None
    limit: Optional[int] = None
    sql: Optional[str] = None


SYSTEM_PROMPT = """You plan queries that answer business questions about a sales dataset.
You never answer the question or produce numbers yourself; you only choose how to query.

Pick the FIRST option that can answer the question exactly:
1. Named intents (no SQL needed):
   - total_revenue: total net revenue; optional single month in `months`.
   - top_product / top_region: highest revenue; optional single month in `months`.
   - revenue_breakdown: revenue grouped by `dimension`, highest first; optional single month.
   - monthly_trend: revenue per month over the whole period.
   - month_comparison: revenue in exactly two months given in `months`.
2. custom_query: one aggregate `metric` (revenue, gross_revenue, units, orders, avg_order_value,
   discount, returned_orders, returned_value), optional `group_by` (product, region, category, sales_channel, month), optional
   `filters` (dimension = value, values must come from the known values below), optional `months`
   filter, `order` (desc = highest first, asc = lowest first, null = chronological), optional `limit`.
3. sql_query: only if 1 and 2 cannot express the question. Write ONE SQLite SELECT in `sql`.
   Use table revenue_sales for revenue questions (completed orders only) and table sales only when
   the question is about statuses/returns. Allowed functions: SUM, AVG, COUNT, MIN, MAX, TOTAL,
   ROUND, ABS, COALESCE, IFNULL, NULLIF, IIF, STRFTIME, DATE, JULIANDAY, SUBSTR, LOWER, UPPER,
   TRIM, LENGTH, INSTR, PRINTF, CAST, GROUP_CONCAT. Return at most 50 rows with clear column aliases.
4. unsupported: forecasts, profit/margin/cost, customers, anything not in the data, requests to
   modify data, or if you are unsure what is being asked.

Schema:
  sales(transaction_id TEXT, order_date TEXT 'YYYY-MM-DD', order_month TEXT 'YYYY-MM', product TEXT,
        category TEXT, region TEXT, sales_channel TEXT, quantity INTEGER, unit_price REAL,
        discount_amount REAL, status TEXT 'Completed'|'Returned')
  revenue_sales = sales rows with status 'Completed', plus gross_revenue (quantity*unit_price)
        and net_revenue (gross_revenue - discount_amount). "Revenue" means SUM(net_revenue)."""

SUMMARY_PROMPT = """Write a one or two sentence answer to the user's question using ONLY the query
result provided. Do not calculate new numbers (no percentages, differences or totals that are not in
the result) and do not add information that is not in the result. Amounts are in Indian rupees (₹)."""


class LLMInterpreter:
    """Plans queries with whichever LLM the environment configures (see ``app/llm.py``)."""
    name = "llm"

    def __init__(self, client: LLMClient):
        self.client = client

    @property
    def label(self) -> str:
        return self.client.label

    def interpret(self, question: str, ctx: DataContext) -> Intent:
        known = "\n".join(f"  {d}: {', '.join(v)}" for d, v in ctx.values.items())
        plan = self.client.complete_json(
            SYSTEM_PROMPT,
            f"Available months: {', '.join(ctx.months)}\nKnown values:\n{known}\n\nQuestion: {question}",
            LLMPlan,
        )
        return self._to_intent(plan)

    def summarize(self, question: str, columns: list[str], rows: list[list]) -> str:
        return self.client.complete_text(
            SUMMARY_PROMPT,
            f"Question: {question}\n\nQuery result (JSON):\n{json.dumps({'columns': columns, 'rows': rows[:50]})}",
        )

    @staticmethod
    def _to_intent(plan: LLMPlan) -> Intent:
        months = [m for m in plan.months if _MONTH_FORMAT_RE.match(m)]
        if len(months) != len(plan.months):
            raise ValueError(f"LLM returned malformed months: {plan.months}")
        if plan.intent == "revenue_breakdown" and plan.dimension is None:
            raise ValueError("LLM chose revenue_breakdown without a dimension")
        if plan.intent == "sql_query":
            if not plan.sql:
                raise ValueError("LLM chose sql_query without SQL")
            return Intent("sql_query", sql=plan.sql)
        if plan.intent == "custom_query":
            filters: dict[str, list[str]] = {}
            for f in plan.filters:
                filters.setdefault(f.dimension, []).append(f.value)
            return Intent("custom_query", spec=QuerySpec(
                metric=plan.metric or "revenue", group_by=plan.group_by, filters=filters,
                months=months, order=plan.order, limit=plan.limit,
            ))
        return Intent(plan.intent, dimension=plan.dimension, months=months)


class Interpreter:
    """LLM first when configured; deterministic rules otherwise or on any LLM failure."""

    def __init__(self, use_llm: bool | None = None):
        self.rules = RuleInterpreter()
        self.llm: LLMInterpreter | None = None
        if use_llm is False:
            return
        try:
            config = LLMConfig.from_env()
            if config:
                self.llm = LLMInterpreter(LLMClient(config))
                log.info("LLM planner enabled: %s", self.llm.label)
        except LLMError as exc:  # misconfiguration should not stop the app
            log.warning("LLM planner disabled: %s", exc)

    @property
    def mode(self) -> str:
        return f"{self.llm.label}+rules" if self.llm else "rules"

    def interpret(self, question: str, ctx: DataContext) -> tuple[Intent, str]:
        if self.llm:
            try:
                return self.llm.interpret(question, ctx), "llm"
            except Exception as exc:
                log.warning("LLM interpretation failed, using rules: %s", exc)
        return self.rules.interpret(question, ctx), "rules"

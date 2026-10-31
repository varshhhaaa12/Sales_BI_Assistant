# AI Usage Notes

> Draft: the factual parts below describe what actually happened while building this. Edit the wording so it is in your own voice, and add anything you did yourself (manual testing, changes you made, what you'd defend in the interview).

## AI tools used

- **Claude Code (Claude Opus)** in VS Code: analysis of the brief, design discussion, code generation, test writing, debugging.
- **Any LLM at runtime** (optional): Gemini, OpenAI, Claude, Groq, OpenRouter or local Ollama, chosen by env vars. A single generic client speaks the OpenAI-compatible Chat Completions protocol. It is provider-agnostic on purpose: the business logic, prompts and safety checks never depend on one vendor's SDK.

## How AI was used during development

1. **Requirement analysis.** Extracted the brief from the PDF, mapped each functional requirement to a component, and profiled the CSV before writing code. Profiling found the zero-quantity row, the missing unit price and the returned order.
2. **Architecture decision.** Weighed free-form text-to-SQL against intent routing over a fixed query catalog. I chose the catalog: the brief stresses correctness and no destructive operations, and with a catalog the LLM can't make up numbers or SQL.
3. **Code generation.** Generated each module, then read and checked it against the requirements. I checked the revenue numbers against an independent calculation instead of trusting the generated SQL.
4. **Tests.** Generated the test cases, with an "oracle" that recomputes expected values in plain Python from the CSV, so the tests don't share logic with the code under test.

## Problems encountered and how they were debugged

I ran a batch of realistic questions through the API before writing tests. That surfaced these bugs in the rule interpreter:

| Problem | How it showed up | Fix |
|---|---|---|
| `"drop table sales"` was treated as a month comparison | "drop" was in the comparison keyword list | Removed it; added an explicit write-verb guard that returns `unsupported` |
| `"delete all sales"` returned **total revenue** | "sales" plus "all" matched the total-revenue rule | Same write-verb guard, checked before any other rule |
| `"May revenue vs June revenue"` was not recognised | Heuristic to tell the verb "may" from the month only accepted "in May", "and May", etc. | Inverted it: treat "may" as a month unless it is followed by a pronoun or verb ("may I…") |
| `"Is June revenue higher than May?"` was unsupported | No comparison keyword matched | Added "higher than", "lower than", "more than", "less than" |
| `"How did revenue change between two months?"` was refused generically | Comparison with no months fell through | Route it to `month_comparison` so the user gets "please name two months" |
| "Which product sold the most?" was ambiguous: revenue or units? | Found during review | Kept revenue ranking; every ranking now states "ranked by net revenue, not units sold" in the assumptions |
| The first version only answered the 5 example questions | User testing: real users ask about units, specific products and regions, lowest, etc. | Added the `QuerySpec` builder (metric × filters × grouping × ranking) and an LLM text-to-SQL fallback under strict validation |
| `"revenue from Tablet"` silently returned the **overall total** | Batch testing after widening the parser; an unknown product was simply ignored | Words after "for/of/from/in" must be known keywords or data values, otherwise the question is refused with the list of known values |
| `"How many returned orders?"` answered 48 (all completed orders) | Same batch | Added `returned_orders` / `returned_value` metrics that read returned rows |
| An LLM summary could state numbers the query never returned | Design review of the text-to-SQL path | Number-grounding check; the summary is discarded if any number isn't in the result |

Safety guard: I wanted to prove the SQLite authorizer is a real second layer, not just duplicate the static check. So I added a test with `WITH x AS (SELECT 1) DELETE FROM sales`. It passes the "starts with WITH" check, and the authorizer has to block it.

## What I would improve with another day

- Benchmark the LLM planner on 50+ paraphrased questions and compare its accuracy with the rules.
- Multiple metrics and two-level grouping offline ("revenue and units by region and month").
- Relative dates ("last month", "Q1", "first half").
- Follow-up questions with conversation memory ("and in the North?").
- Dockerfile and CI (GitHub Actions running pytest).

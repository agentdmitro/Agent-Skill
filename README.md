# Wikipedia Interest Agent Skill

A deterministic Agent Skill for analyzing topic interest across Wikipedia language editions using Wikimedia pageview data.

The skill is designed for B2C product teams that want an additional data signal when deciding:

- which topics to explore;
- which content areas may be gaining or losing interest;
- which language audiences may be worth validating next.

Wikipedia pageviews are treated as an **attention signal**, not as market size, purchase intent, or revenue potential.

## What it does

The skill can:

- resolve topics through Wikidata;
- verify Wikipedia articles across requested languages;
- fetch monthly Wikimedia pageview data;
- analyze long-term and recent trends;
- evaluate data coverage, volatility and outliers;
- estimate how reliable a trend conclusion is;
- compare the same topic across multiple Wikipedia editions;
- generate PNG charts;
- generate concise one-page PDF reports;
- preserve canonical topic identity across follow-up requests;
- explicitly handle ambiguity, missing articles and partial API failures.

Most analytical work is deterministic Python code rather than LLM reasoning.

## Architecture

```text
User request
    ↓
Agent
    ↓
Structured request
    ↓
Wikidata topic resolution
    ↓
Verified Wikipedia articles
    ↓
Wikimedia Pageviews API
    ↓
Deterministic statistical analysis
    ↓
Cross-language comparison
    ↓
Compact structured result
    ↓
Chart / one-page PDF
    ↓
Agent explanation
```

The language model is not responsible for calculating statistics, guessing article titles or assigning trend reliability.

## Requirements

- Python 3.12+
- `uv`

Runtime dependencies are intentionally small:

- Pydantic
- httpx
- matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/agentdmitro/Agent-Skill.git
cd Agent-Skill
```

Install dependencies:

```bash
uv sync --group dev
```

## Wikimedia User-Agent

Wikimedia requires an identifiable User-Agent for API access.

Set one before running live analyses:

```bash
export WIKIPEDIA_INTEREST_USER_AGENT='WikipediaInterestSkill/1.0 (https://github.com/agentdmitro/Agent-Skill)'
```

No Wikimedia API key is required.

## Usage

The main entry point is:

```bash
uv run python scripts/run_analysis.py
```

### Single-language trend analysis

Analyze interest in astronomy in Ukrainian Wikipedia:

```bash
uv run python scripts/run_analysis.py \
  --topic Q333 \
  --query-language en \
  --languages uk \
  --start 2024-01 \
  --end 2026-08 \
  --intent trend
```

The result includes:

- canonical Wikidata entity;
- verified Wikipedia article;
- data coverage;
- total and median pageviews;
- robust trend direction;
- period and year-over-year change;
- volatility;
- detected outliers;
- recent trend confirmation;
- evidence reliability.

## Cross-language comparison

Compare astronomy across Ukrainian, Polish and Czech Wikipedia:

```bash
uv run python scripts/run_analysis.py \
  --topic Q333 \
  --query-language en \
  --languages uk pl cs \
  --start 2024-01 \
  --end 2026-08 \
  --intent compare
```

The skill keeps separate:

- absolute Wikipedia attention;
- trend strength;
- reliability of the trend.

It does not convert higher pageviews into claims about market size.

## Generate a chart and PDF report

```bash
uv run python scripts/run_analysis.py \
  --topic Q333 \
  --query-language en \
  --languages uk pl cs \
  --start 2024-01 \
  --end 2026-08 \
  --intent compare \
  --chart \
  --pdf \
  --output-dir ./output
```

Example outputs:

```text
output/
├── wi-q333-v1-astronomy-chart.png
└── wi-q333-v1-astronomy-report.pdf
```

The PDF is intentionally limited to a short one-page research brief.

## Natural-language topic resolution

Topics can also be supplied as text:

```bash
uv run python scripts/run_analysis.py \
  --topic "astronomy" \
  --query-language en \
  --languages uk pl \
  --start 2024-01 \
  --end 2026-08 \
  --intent compare
```

Topic resolution is deliberately conservative.

If Wikidata returns multiple plausible entities, the skill returns:

```text
clarification_required
```

instead of silently choosing the first search result.

For example, a query such as:

```text
Mercury
```

may require clarification between the planet, chemical element, mythological figure or another entity.

Direct Wikidata QIDs avoid this ambiguity when the intended entity is already known.

## Interpreting results

A trend may be classified as:

```text
growing
declining
stable
unclear
insufficient_data
```

Reliability is an evidence grade:

```text
high
medium
low
insufficient
```

It is **not a probability**.

For example:

```text
direction: declining
reliability: high
```

means that the available Wikipedia time series provides strong evidence for a descriptive decline under the skill's deterministic rules.

It does not mean there is a specific percentage probability that demand is declining.

## Candidate classification

For multi-language research, the skill can classify an observed Wikipedia signal as:

```text
strong_candidate
possible_candidate
inconclusive
weak_current_signal
```

These labels mean:

> worth considering for further validation based on Wikipedia attention data.

They do **not** mean:

> launch in this market.

Market decisions should be validated with additional signals.

## Missing and partial data

The skill does not silently convert missing months into zero pageviews.

It explicitly tracks:

- requested months;
- observed months;
- missing months;
- coverage ratio.

Partial language failures are also preserved.

Example:

```text
Polish      → analyzed
Czech       → analyzed
Slovak      → API timeout
```

The Polish/Czech comparison can still proceed while the Slovak failure remains visible.

## Follow-up analysis

Resolved topic identity is stored in canonical analysis state.

A follow-up such as:

```text
Add Ukrainian.
```

can reuse the existing Wikidata QID rather than searching for the topic again.

This reduces:

- ambiguity;
- unnecessary API requests;
- context size;
- risk of resolving a different entity.

Changing the actual topic triggers new entity resolution.

## Development checks

Run the complete deterministic test suite:

```bash
uv run pytest -q
```

Lint:

```bash
uv run ruff check .
```

Formatting:

```bash
uv run ruff format --check .
```

Type checking:

```bash
uv run mypy src tests
uv run mypy evals
```

Offline evaluation:

```bash
uv run python evals/runner.py --mode offline
```

Current validated baseline:

```text
186 tests passed
39 / 39 offline evaluation scenarios passed
Ruff clean
mypy clean
```

## Evaluation

The project includes a dedicated evaluation harness under:

```text
evals/
```

It tests:

- skill activation;
- request extraction;
- ambiguity;
- direct QID resolution;
- single-language analysis;
- multi-language comparison;
- follow-up behavior;
- partial failures;
- insufficient data;
- adversarial prompts;
- numeric hallucinations;
- unsupported market claims;
- prompt injection;
- report requests.

Offline evaluation uses mocked Wikimedia/Wikidata HTTP boundaries while executing the real production orchestration pipeline.

### Real-model evaluation

Optional real-model evaluation can be run through OpenRouter.

Configure:

```bash
export OPENROUTER_API_KEY="..."
export EVAL_MODEL="..."
```

Then run:

```bash
uv run python evals/runner.py \
  --mode model \
  --model "$EVAL_MODEL"
```

The purpose is to verify that a small, inexpensive tool-capable model can use the skill correctly without performing the statistical reasoning itself.

## Key design principles

The skill follows several strict rules:

- Never invent a Wikipedia article.
- Never guess a Wikidata entity when resolution is ambiguous.
- Never fabricate missing pageviews.
- Never treat missing data as zero.
- Never let the LLM recalculate deterministic metrics.
- Never present reliability as a probability.
- Never infer purchase intent from Wikipedia traffic.
- Never treat raw cross-language pageviews as market size.
- Never invent causes for spikes.
- Never forecast future demand from the current V1 analysis.

## Limitations

Wikipedia pageviews are only one behavioral signal.

Important limitations include:

- pageviews do not measure willingness to pay;
- Wikipedia usage differs between language communities;
- language edition size differs substantially;
- users may browse Wikipedia in languages other than their native language;
- article structure and coverage can differ between editions;
- external events may create temporary spikes;
- V1 does not perform causal attribution;
- V1 does not forecast future demand.

Results should therefore be used to identify **directions for further validation**, not as standalone product-launch decisions.

## Future development

Possible V2 improvements should be introduced only after the V1 workflow is validated on real usage.

Potential directions:

- configurable user-defined prospectiveness criteria;
- topic clusters instead of a single Wikipedia article;
- additional research signals such as Google Trends;
- App Store / Google Play demand signals;
- larger batch studies;
- persistent caching for repeated research;
- deeper seasonality analysis;
- event attribution using external evidence;
- richer comparative research reports.

These signals should remain separate and explainable rather than being collapsed into one opaque opportunity score.

## Project status

V1 deterministic implementation is complete and has passed:

- unit/integration testing;
- offline evaluation;
- live Wikidata resolution;
- live Wikimedia pageview retrieval;
- live single-language analysis;
- live multi-language comparison;
- chart generation;
- one-page PDF generation.

Final acceptance additionally includes Agent Skills specification validation and real cheap-model evaluation.

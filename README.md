# Code Sample — Tongrui (Neil) Zhang
## Application: Data Analyst, Research and Analytics Department
## Office of the New York State Attorney General (RAD_NYC_DAT_6444)

---

### Overview

This code sample is drawn from an ongoing research project at Georgetown University's McCourt School of Public Policy, where I work as a Research Assistant under the supervision of faculty in the international investment policy group.

The project analyzes how investment-environment signals in official Chinese government guidance documents have changed over time, using a corpus of policy texts spanning 110+ countries across two time periods (2015/16 and 2022/23). The pipeline transforms raw, OCR-extracted government documents into structured, quantitative datasets suitable for econometric modeling.

I am sharing three modules from this pipeline. The underlying data cannot be shared due to the ongoing nature of the research, but each module is self-contained and documented to make the logic fully traceable without running the code.

---

### Files Included

#### 1. [`policy_text_extraction_pipeline.py`](./policy_text_extraction_pipeline.py) — Stage 1: Text Extraction & Structuring

**What it does:**
Ingests plain-text files (one per country-year) that were produced by OCR from official government policy documents. Each document contains 10–15 policy topics (e.g. anti-corruption regulations, labor law, environmental compliance). The pipeline:

- Parses document structure using numbered heading patterns (e.g. `3.5.1`)
- Filters out table-of-contents entries using heuristic rules
- Removes embedded table blocks and OCR noise
- Applies topic-specific regex patterns (OCR-tolerant, allowing character spacing errors) to match each section to a predefined topic
- Handles special-case sub-section extraction (e.g. extracting the "media attitude toward China" sub-section from within a broader "major media" chapter)
- Exports a structured CSV: one row per country-year, one column per topic

**Key techniques demonstrated:** document segmentation, regex pattern design for noisy OCR text, configurable ETL pipeline, reproducible batch processing.

---

#### 2. [`policy_sentiment_scoring_pipeline.py`](./policy_sentiment_scoring_pipeline.py) —  Stage 2: LLM-Based Sentiment Scoring

**What it does:**
Takes the structured topic-level text segments from Stage 1 and assigns a sentiment score (1–5) to each sentence using a large language model (Qwen-Max via OpenAI-compatible API). The scoring rubric is:

- **5** — Strongly positive signal for foreign business / investment environment
- **4** — Moderately positive
- **3** — Neutral / purely descriptive
- **2** — Moderately negative / restrictive
- **1** — Strongly negative / high-risk

Each topic has a tailored expert-persona prompt that reflects the domain-specific interpretation of "positive" vs "negative" — for example, the anti-corruption topic scores a country higher if enforcement is rigorous, while the contracting topic scores higher if foreign firms face fewer barriers.

The pipeline:
- Sends sentences to the LLM in batches of 20 to manage token cost
- Implements prompt caching strategies that reduced token usage by ~50% vs baseline
- Parses structured JSON output with a fallback extraction strategy
- Flags sentences over 200 characters as too long for reliable single-sentence scoring
- Exports both a wide table (score columns appended to original rows) and a long table (one row per sentence, with metadata)
- Aggregates sentence-level scores into country × topic summary statistics (mean, variance, per-score ratios)

**Key techniques demonstrated:** LLM prompt engineering, structured output parsing, batched API calls, reproducible scoring with temperature=0, data validation before model training.

---

#### 3. [`policy_period_comparison_analysis.py`](./policy_period_comparison_analysis.py)  —  Stage 3: Cross-Period Comparison & Visualisation

**What it does:**
Loads the country-level topic score summaries for both periods and produces a set of comparative analyses and visualisations:

**Tables (CSV + Excel workbook):**
- Country-level overall score change ranking
- Per-topic aggregate statistics (mean score change, sentence-count change, variance change)
- Top-10 improvers and top-10 decliners per topic

**Visualisations:**
- Choropleth world map: overall score change by country (interactive HTML)
- Choropleth world map: per-topic score change with dropdown selector (interactive HTML)
- Bubble scatter plot: old vs new scores per topic, bubble size = sentence count (interactive HTML)
- Heatmap: all countries × all topics, score change (PNG)
- Bar charts: mean score/sentence/variance change by topic (PNG)

**Key techniques demonstrated:** cross-period data alignment, aggregation and ranking, Plotly interactive visualisation, Seaborn static visualisation, structured output to multiple formats.

---

### How the Three Stages Fit Together

```
Stage 1: policy_text_extraction_pipeline.py
    Input : {country}_{year}.txt  (OCR-extracted policy documents)
    Output: policy_topics_extracted.csv  (structured topic-level text)
            ↓
Stage 2: policy_sentiment_scoring_pipeline.py
    Input : policy_topics_extracted.csv
    Output: {topic}-scored.csv (wide)  +  {topic}-long-scored.csv
            {year_range}_topic_score_summary.csv
            ↓
Stage 3: policy_period_comparison_analysis.py
    Input : topic_score_summary files for two periods
    Output: comparison tables (CSV + Excel)
            interactive maps and charts (HTML + PNG)
```


*For questions about this sample, please contact: tz280@georgetown.edu*

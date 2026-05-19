"""
Policy Sentiment Scoring Pipeline  —  Stage 2: LLM-Based Classification
=========================================================================
Author : Tongrui (Neil) Zhang
Project: Cross-Country Investment Policy Analysis
         Georgetown University McCourt School of Public Policy

Overview
--------
This module implements Stage 2 of the policy analytics pipeline. It takes the
structured topic-level text segments produced by Stage 1 (text extraction) and
assigns a sentiment score (1–5) to each sentence using a large language model.

Scoring rubric (applied per topic):
    5  Strongly positive signal for foreign investors / business environment
    4  Moderately positive
    3  Neutral / descriptive — no directional signal
    2  Moderately negative / restrictive
    1  Strongly negative / high-risk signal

Sentences exceeding 200 characters are flagged as too long for reliable
single-sentence scoring and are excluded from LLM calls (score = -1).

Pipeline steps
--------------
1. Load per-topic sentence CSV produced by Stage 1.
2. Separate sentences into scoreable (<= 200 chars) and long (> 200 chars).
3. Call the LLM API in batches of BATCH_SIZE sentences per request.
4. Parse the JSON array returned by the model; fall back to neutral (3) on
   parse failure to avoid silent data loss.
5. Write results in two formats:
   - Wide table  : original columns + score_1 … score_N columns (one col per
                   sentence slot), one row per country-year.
   - Long table  : one row per sentence, with country, year, topic, sentence
                   text, score, and a note field.

Input
-----
Directory of CSVs named  {topic}-{year_range}-sentences.csv
Each file has columns: country, year, sentence_count, total_chinese_chars,
sent_1, sent_2, …, sent_N

Output
------
{topic}-{year_range}-sentences-scored.csv   (wide format)
{topic}-{year_range}-long-scored.csv        (long format)

Dependencies
------------
    pip install openai pandas

API key
-------
Export your DashScope (Alibaba Cloud) key before running:
    export DASHSCOPE_API_KEY="sk-..."
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd
from openai import OpenAI


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

# --- Topics to process -------------------------------------------------------
TOPICS: list[str] = [
    "commercial_bribery",
    "media_relations",
    "labor_employment_regulations",
    "law_enforcement_relations",
    "foreign_contracting",
    "trade_union_relations",
    "union_relationship_management",
    "overseas_contracting",
    "overseas_labor_cooperation",
    "environmental_regulations",
    "ecological_protection",
    "labor_supply_and_wages",
    "media_attitude_toward_china",
]

# --- I/O paths ---------------------------------------------------------------
INPUT_DIR  = "outputs/sentences"       # Stage 1 sentence CSVs
OUTPUT_DIR = "outputs/scored"          # Scored CSVs written here
YEAR_RANGE = "2022_23"                 # Suffix used in filenames

# --- Model settings ----------------------------------------------------------
MODEL_NAME   = "qwen-max"
TEMPERATURE  = 0.0     # Zero temperature for reproducibility
BATCH_SIZE   = 20      # Sentences per API call
MAX_SENTENCE_CHARS = 200  # Sentences longer than this are excluded from scoring


# =============================================================================
# SECTION 2: PROMPT ENGINEERING
# =============================================================================

# Each topic has a tailored scoring prompt that instructs the model to adopt a
# domain-expert persona and apply directional scoring criteria.
# The rubric is standardised across all topics:
#   1-2 = negative/restrictive environment
#   3   = neutral / purely descriptive
#   4-5 = positive/open environment

_TOPIC_PROMPTS: dict[str, str] = {

    "foreign_contracting": (
        "You are an expert in international construction markets and market-access policy, "
        "advising Chinese enterprises operating overseas. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  The local environment is open to foreign contractors: low barriers, fair "
        "competition, Chinese qualifications recognised, minimal local-content requirements.\n"
        "  1–2  The environment is restrictive: protectionist rules, opaque procedures, "
        "relationship-dependent access, long approval cycles, or only Western qualifications "
        "accepted.\n"
        "  3    Neutral description of laws or procedures with no clear directional signal.\n\n"
    ),

    "overseas_contracting": (
        "You are an expert in international construction markets and market-access policy, "
        "advising Chinese enterprises operating overseas. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  The local environment is open to foreign contractors: low barriers, fair "
        "competition, Chinese qualifications recognised, minimal local-content requirements.\n"
        "  1–2  The environment is restrictive: protectionist rules, opaque procedures, "
        "relationship-dependent access, long approval cycles, or only Western qualifications "
        "accepted.\n"
        "  3    Neutral description of laws or procedures with no clear directional signal.\n\n"
    ),

    "commercial_bribery": (
        "You are an expert in anti-corruption policy and governance. "
        "Score each sentence on a 1–5 scale, focusing on evaluative statements rather than "
        "mere enumeration of rules:\n\n"
        "  4–5  Anti-corruption laws are comprehensive and enforced; independent oversight "
        "bodies exist; penalties are meaningful; public accountability is strong.\n"
        "  1–2  Laws are weak, unenforced, or absent; bribery is prevalent; monitoring is "
        "inadequate; impunity is common.\n"
        "  3    Neutral listing of rules, institutions, or penalties without an evaluative "
        "stance. Note: listing strict penalties does not itself indicate good governance — "
        "score 3 unless effectiveness is explicitly assessed.\n\n"
    ),

    "labor_supply_and_wages": (
        "You are a labour-market and employment-law specialist. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  Labour law is comprehensive and well enforced; worker rights are protected; "
        "labour–management relations are stable; workforce quality is high; "
        "unemployment is low or supply–demand is balanced.\n"
        "  1–2  Labour law is weak or poorly enforced; worker protections are inadequate; "
        "workforce quality is low; employer power dominates unchecked.\n"
        "  3    Objective description of regulations or institutions without a clear "
        "evaluative stance.\n\n"
    ),

    "labor_employment_regulations": (
        "You are a labour-market and employment-law specialist. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  Labour law is comprehensive and well enforced; worker rights are protected; "
        "labour–management relations are stable; workforce quality is high; "
        "unemployment is low or supply–demand is balanced.\n"
        "  1–2  Labour law is weak or poorly enforced; worker protections are inadequate; "
        "workforce quality is low; employer power dominates unchecked.\n"
        "  3    Objective description of regulations or institutions without a clear "
        "evaluative stance.\n\n"
    ),

    "overseas_labor_cooperation": (
        "You are a labour-policy specialist advising foreign enterprises on overseas "
        "staffing and visa strategy. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  Foreign workers are welcomed or encouraged; work-permit procedures are "
        "straightforward and stable; policy is flexible toward overseas labour.\n"
        "  1–2  Foreign workers face significant restrictions; local-hire mandates are "
        "strict; visa or permit processes are burdensome, unpredictable, or risky.\n"
        "  3    Neutral description of applicable rules without a clear directional signal.\n\n"
    ),

    "trade_union_relations": (
        "You are a labour-relations and social-organisation policy specialist advising "
        "foreign investors. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  Union power is limited; strikes are rare; government effectively manages "
        "labour disputes; operational disruption risk is low.\n"
        "  1–2  Unions are powerful, politically influential, or prone to frequent strikes "
        "that directly threaten foreign investment or business operations.\n"
        "  3    Neutral or purely descriptive content with no clear directional signal.\n\n"
    ),

    "union_relationship_management": (
        "You are a labour-relations and social-organisation policy specialist advising "
        "foreign investors. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  Union power is limited; strikes are rare; government effectively manages "
        "labour disputes; operational disruption risk is low.\n"
        "  1–2  Unions are powerful, politically influential, or prone to frequent strikes "
        "that directly threaten foreign investment or business operations.\n"
        "  3    Neutral or purely descriptive content with no clear directional signal.\n\n"
    ),

    "media_relations": (
        "You are a public-diplomacy and media-environment specialist advising Chinese "
        "enterprises on overseas reputation management. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  The local media environment is neutral or favourable toward China; proactive "
        "engagement with media is encouraged.\n"
        "  1–2  Media coverage of China is predominantly negative, biased, or hostile; "
        "caution or passive media engagement is advised.\n"
        "  3    Neutral description of the media landscape without a clear directional "
        "signal.\n\n"
    ),

    "media_attitude_toward_china": (
        "You are a public-diplomacy and media-environment specialist advising Chinese "
        "enterprises on overseas reputation management. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  The local media environment is neutral or favourable toward China; proactive "
        "engagement with media is encouraged.\n"
        "  1–2  Media coverage of China is predominantly negative, biased, or hostile; "
        "caution or passive media engagement is advised.\n"
        "  3    Neutral description of the media landscape without a clear directional "
        "signal.\n\n"
    ),

    "environmental_regulations": (
        "You are an environmental-compliance and regulatory-risk specialist advising "
        "foreign investors. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  Environmental regulations are strict and rigorously enforced; public "
        "environmental awareness is high; there is precedent for projects being halted on "
        "environmental grounds; compliance obligations must be taken seriously.\n"
        "  1–2  Environmental regulations are weak or rarely enforced; pollution is "
        "tolerated; ecological harm may go unpunished.\n"
        "  3    Neutral description of applicable rules or penalties without an evaluative "
        "stance. Note: stating the content of penalties (without comparing them to other "
        "countries) defaults to 3.\n\n"
    ),

    "ecological_protection": (
        "You are an environmental-compliance and regulatory-risk specialist advising "
        "foreign investors. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  Environmental regulations are strict and rigorously enforced; public "
        "environmental awareness is high; there is precedent for projects being halted on "
        "environmental grounds; compliance obligations must be taken seriously.\n"
        "  1–2  Environmental regulations are weak or rarely enforced; pollution is "
        "tolerated; ecological harm may go unpunished.\n"
        "  3    Neutral description of applicable rules or penalties without an evaluative "
        "stance. Note: stating the content of penalties (without comparing them to other "
        "countries) defaults to 3.\n\n"
    ),

    "law_enforcement_relations": (
        "You are a governance and law-enforcement specialist. "
        "Score each sentence on a 1–5 scale:\n\n"
        "  4–5  Law enforcement is systematic, transparent, and consistent; rules are "
        "applied impartially.\n"
        "  1–2  Law enforcement is arbitrary, corrupt, abusive, or relationship-dependent; "
        "informal payments or personal connections are required.\n"
        "  3    Neutral or descriptive content with no clear directional signal.\n\n"
    ),
}

_DEFAULT_PROMPT = (
    "You are a policy analyst. Score each sentence on a 1–5 scale:\n\n"
    "  4–5  Positive signal for foreign business environment.\n"
    "  1–2  Negative / restrictive signal.\n"
    "  3    Neutral or purely descriptive.\n\n"
)


# =============================================================================
# SECTION 3: API CLIENT
# =============================================================================

def _build_client() -> OpenAI:
    """
    Initialise and return an OpenAI-compatible client for the DashScope API.

    Raises:
        RuntimeError: If DASHSCOPE_API_KEY is not set in the environment.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Environment variable DASHSCOPE_API_KEY is not set. "
            "Export it before running: export DASHSCOPE_API_KEY='sk-...'"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


CLIENT: OpenAI = _build_client()


# =============================================================================
# SECTION 4: PROMPT CONSTRUCTION & RESPONSE PARSING
# =============================================================================

def build_batch_prompt(topic: str, sentences: list[str]) -> str:
    """
    Construct a numbered, topic-specific scoring prompt for a batch of sentences.

    The prompt instructs the model to return a JSON integer array only —
    no prose, no markdown fences — e.g.  [4, 3, 2, 5, 3].

    Args:
        topic:     Topic key (must match a key in _TOPIC_PROMPTS or falls back
                   to the default rubric).
        sentences: List of sentence strings to score in this batch.

    Returns:
        Formatted prompt string ready to send to the model.
    """
    header = _TOPIC_PROMPTS.get(topic, _DEFAULT_PROMPT)

    numbered_sentences = "\n".join(
        f"{i + 1}. {s}" for i, s in enumerate(sentences)
    )

    tail = (
        "\n\nScore every sentence above in order. "
        "Return ONLY a JSON integer array, e.g. [4,3,2]. "
        "Do not include any explanation or markdown formatting."
    )

    return header + numbered_sentences + tail


def parse_scores(response_text: str, expected_count: int) -> list[int]:
    """
    Extract an integer score list from the model's response text.

    Parsing strategy:
        1. Attempt to locate and JSON-decode the first [...] block.
        2. Fall back to extracting all single digits 1–5 from the text.
        3. If neither yields enough scores, pad with neutral score (3).

    Args:
        response_text:  Raw string returned by the model.
        expected_count: Number of scores expected (= batch size).

    Returns:
        List of integers of length exactly ``expected_count``,
        each in range [1, 5].
    """
    # Strategy 1: JSON array extraction
    match = re.search(r"\[.*?\]", response_text, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                scores = [
                    int(x) if isinstance(x, (int, float)) and 1 <= int(x) <= 5 else 3
                    for x in parsed
                ]
                if len(scores) >= expected_count:
                    return scores[:expected_count]
                return scores + [3] * (expected_count - len(scores))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 2: regex digit extraction
    digits = [int(d) for d in re.findall(r"[1-5]", response_text)]
    if len(digits) >= expected_count:
        return digits[:expected_count]
    return digits + [3] * (expected_count - len(digits))


# =============================================================================
# SECTION 5: LLM SCORING
# =============================================================================

def score_batch(topic: str, sentences: list[str]) -> list[int]:
    """
    Send one batch of sentences to the LLM and return integer scores.

    Args:
        topic:     Topic key for prompt selection.
        sentences: Batch of sentence strings (length <= BATCH_SIZE).

    Returns:
        List of integer scores, one per sentence.
    """
    prompt = build_batch_prompt(topic, sentences)
    response = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.choices[0].message.content.strip()
    return parse_scores(raw_text, len(sentences))


def score_all_sentences(
    topic: str,
    sentence_items: list[tuple[int, int, str]],
) -> list[int]:
    """
    Score all sentences for a topic using batched API calls.

    Args:
        topic:           Topic key.
        sentence_items:  List of (row_index, col_index, sentence_text) tuples.

    Returns:
        Flat list of scores in the same order as ``sentence_items``.
    """
    all_texts = [text for _, _, text in sentence_items]
    all_scores: list[int] = []

    total_batches = (len(all_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[batch_idx: batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        print(f"    Batch {batch_num}/{total_batches}: {len(batch)} sentences")
        scores = score_batch(topic, batch)
        all_scores.extend(scores)

    return all_scores


# =============================================================================
# SECTION 6: PER-TOPIC PROCESSING
# =============================================================================

def process_topic(topic: str) -> None:
    """
    Load the sentence CSV for one topic, score all sentences, and write output.

    Output files:
        Wide table  — original rows with appended score_1 … score_N columns.
        Long table  — one row per sentence with metadata and score.

    Args:
        topic: Topic key. Must match a filename in INPUT_DIR.
    """
    print(f"\n{'=' * 65}")
    print(f"  Topic: {topic}")
    print(f"{'=' * 65}")

    input_path = os.path.join(INPUT_DIR, f"{topic}-{YEAR_RANGE}-sentences.csv")
    if not os.path.exists(input_path):
        print(f"  [SKIP] Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Identify sentence columns (sent_1, sent_2, …) in sorted order
    sentence_cols = sorted(
        [c for c in df.columns if c.startswith("sent_")],
        key=lambda x: int(x.split("_")[1]),
    )
    n_sent_cols = len(sentence_cols)
    n_rows = len(df)

    # ------------------------------------------------------------------
    # Partition sentences into scoreable and too-long
    # ------------------------------------------------------------------
    scoreable: list[tuple[int, int, str]] = []   # (row_idx, col_idx, text)
    too_long:  list[tuple[int, int, str]] = []

    for row_idx in range(n_rows):
        row = df.iloc[row_idx]
        sentence_count = int(row.get("sentence_count", 0) or 0)

        for col_idx in range(min(sentence_count, n_sent_cols)):
            text = str(row[sentence_cols[col_idx]]).strip()
            if not text:
                continue
            if len(text) > MAX_SENTENCE_CHARS:
                too_long.append((row_idx, col_idx, text))
            else:
                scoreable.append((row_idx, col_idx, text))

    print(f"  Sentences to score : {len(scoreable)}")
    print(f"  Sentences too long : {len(too_long)}  (flagged as -1)")

    # ------------------------------------------------------------------
    # Score in batches
    # ------------------------------------------------------------------
    scores = score_all_sentences(topic, scoreable)

    # ------------------------------------------------------------------
    # Build result matrices
    # ------------------------------------------------------------------
    scores_matrix: list[list[Any]] = [[0] * n_sent_cols for _ in range(n_rows)]
    long_records:  list[dict[str, Any]] = []

    # Sentences excluded from scoring — score = -1
    for row_idx, col_idx, text in too_long:
        scores_matrix[row_idx][col_idx] = -1
        long_records.append({
            "country":        df.iloc[row_idx]["country"],
            "year":           df.iloc[row_idx]["year"],
            "topic":          topic,
            "sentence_index": col_idx + 1,
            "sentence":       text,
            "score":          -1,
            "note":           f"excluded: length > {MAX_SENTENCE_CHARS} chars",
        })

    # Scored sentences
    for (row_idx, col_idx, text), score in zip(scoreable, scores):
        scores_matrix[row_idx][col_idx] = score
        long_records.append({
            "country":        df.iloc[row_idx]["country"],
            "year":           df.iloc[row_idx]["year"],
            "topic":          topic,
            "sentence_index": col_idx + 1,
            "sentence":       text,
            "score":          score,
            "note":           "model_scored",
        })

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Wide table
    score_col_names = [f"score_{i + 1}" for i in range(n_sent_cols)]
    score_df = pd.DataFrame(scores_matrix, columns=score_col_names)
    wide_df  = pd.concat([df, score_df], axis=1)
    wide_path = os.path.join(OUTPUT_DIR, f"{topic}-{YEAR_RANGE}-sentences-scored.csv")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] Wide table  → {wide_path}")

    # Long table
    long_df   = pd.DataFrame(long_records)
    long_path = os.path.join(OUTPUT_DIR, f"{topic}-{YEAR_RANGE}-long-scored.csv")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] Long table  → {long_path}")


# =============================================================================
# SECTION 7: SCORE AGGREGATION
# =============================================================================

def aggregate_scores(topics: list[str]) -> pd.DataFrame:
    """
    Aggregate sentence-level scores from all long-scored CSVs into a
    country × topic summary table.

    For each (country, year, topic) group the function computes:
        sentence_count   — number of valid sentences (score 1–5)
        average_score    — weighted mean score
        score_variance   — variance of scores (ddof=1)
        ratio_1 … ratio_5 — proportion of sentences at each score level

    Args:
        topics: List of topic keys whose long-scored CSVs exist in OUTPUT_DIR.

    Returns:
        DataFrame with one row per (country, year, topic) combination,
        sorted by topic then country then year.
    """
    frames: list[pd.DataFrame] = []

    for topic in topics:
        path = os.path.join(OUTPUT_DIR, f"{topic}-{YEAR_RANGE}-long-scored.csv")
        if not os.path.exists(path):
            print(f"  [SKIP] Long-scored file not found for topic: {topic}")
            continue
        df = pd.read_csv(path)
        df["topic"] = topic
        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No long-scored files found. Run process_topic() for each topic first."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Keep only valid scores (1–5)
    valid = combined[combined["score"].between(1, 5)].copy()

    # Count Chinese characters per sentence (proxy for content density)
    valid["char_count"] = valid["sentence"].apply(
        lambda s: len(re.findall(r"[\u4e00-\u9fff]", s)) if isinstance(s, str) else 0
    )

    grp = valid.groupby(["country", "year", "topic"])

    # Core aggregations
    agg = grp.agg(
        sentence_count  = ("score", "count"),
        average_score   = ("score", "mean"),
        score_variance  = ("score", lambda x: x.var(ddof=1)),
        total_chars     = ("char_count", "sum"),
    ).reset_index()

    # Per-score ratios
    for s in range(1, 6):
        valid[f"_is_{s}"] = (valid["score"] == s).astype(int)

    ratio_agg = grp.agg({f"_is_{s}": "sum" for s in range(1, 6)}).reset_index()

    agg = agg.merge(ratio_agg, on=["country", "year", "topic"])
    for s in range(1, 6):
        agg[f"ratio_{s}"] = (agg[f"_is_{s}"] / agg["sentence_count"]).round(4)
        agg.drop(columns=[f"_is_{s}"], inplace=True)

    # Round numeric columns
    agg["average_score"]  = agg["average_score"].round(4)
    agg["score_variance"] = agg["score_variance"].fillna(0.0).round(4)

    return agg.sort_values(["topic", "country", "year"]).reset_index(drop=True)


# =============================================================================
# SECTION 8: MAIN ENTRY POINT
# =============================================================================

def run_scoring_pipeline(topics: list[str] = TOPICS) -> None:
    """
    Run the full scoring pipeline: score all topics, then aggregate.

    Args:
        topics: List of topic keys to process (default: all defined topics).
    """
    print("=" * 65)
    print("  Policy Sentiment Scoring Pipeline  —  Stage 2")
    print("=" * 65)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Year range : {YEAR_RANGE}")
    print(f"  Topics     : {len(topics)}")
    print(f"  Input dir  : {INPUT_DIR}")
    print(f"  Output dir : {OUTPUT_DIR}")

    # Score each topic
    for topic in topics:
        process_topic(topic)

    # Aggregate into summary table
    print(f"\n{'=' * 65}")
    print("  Aggregating scores …")
    print(f"{'=' * 65}")

    summary = aggregate_scores(topics)
    summary_path = os.path.join(OUTPUT_DIR, f"{YEAR_RANGE}_topic_score_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n  Summary saved → {summary_path}")
    print(f"  Rows : {len(summary)}")
    print(f"  Cols : {len(summary.columns)}")
    print(f"\n{'=' * 65}")
    print("  Pipeline complete.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    run_scoring_pipeline()

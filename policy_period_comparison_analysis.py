#!/usr/bin/env python3
"""
Cross-Period Comparison Analysis  —  Stage 3: Temporal Change & Visualisation
==============================================================================
Author : Tongrui (Neil) Zhang
Project: Cross-Country Investment Policy Analysis
         Georgetown University McCourt School of Public Policy

Overview
--------
This module implements Stage 3 of the policy analytics pipeline. It takes the
country-level topic score summaries produced by Stage 2 (LLM scoring) for two
separate time periods — 2015/16 and 2022/23 — and produces:

    1. Structured comparison tables (CSV + Excel workbook)
       - Country-level overall score changes
       - Topic-level aggregate change statistics
       - Top-10 improvers and decliners per topic

    2. Interactive visualisations (HTML, viewable in any browser)
       - Choropleth world map: overall score change by country
       - Choropleth world map: per-topic score change with dropdown selector
       - Bubble scatter plot: old vs new scores, per topic
       - Bubble scatter plot: overall score comparison across countries

    3. Static figures (PNG)
       - Heatmap: all countries × all topics, score change
       - Bar charts: mean score change, sentence-count change, variance change

Analytical value
----------------
Comparing the two periods reveals how investment-environment signals in
official Chinese government guidance documents shifted between the mid-2010s
and early 2020s — covering dimensions such as labour regulation, anti-
corruption enforcement, media relations, and environmental compliance across
110+ countries.

Input
-----
Two pairs of CSV files produced by Stage 2:
    {period}_country_bigtopic_score_summary_with_code.csv   (long format)
    {period}_country_wide_summary.csv                       (wide format)

Output
------
outputs/tables/   — CSV files and an Excel workbook
outputs/figures/  — PNG heatmaps/bar charts and interactive HTML maps

Dependencies
------------
    pip install pandas numpy matplotlib seaborn plotly
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
OUTPUTS = ROOT / "directONtest" / "OUTPUTS"

OLD_BIG = OUTPUTS / "15&16_country_bigtopic_score_summary_with_code.csv"
NEW_BIG = OUTPUTS / "22&23_country_bigtopic_score_summary_with_code.csv"
OLD_WIDE = OUTPUTS / "15&16_country_wide_summary.csv"
NEW_WIDE = OUTPUTS / "22&23_country_wide_summary.csv"

TOPIC_ORDER = ["劳动相关", "和执法行政人员打交道", "商业贿赂", "媒体", "工会", "承包工程", "环保"]
TOPIC_EN = {
    "劳动相关": "Labor",
    "和执法行政人员打交道": "Dealing with Officials",
    "商业贿赂": "Commercial Bribery",
    "媒体": "Media",
    "工会": "Labor Unions",
    "承包工程": "Contracting Projects",
    "环保": "Environmental Protection",
}

NAME_MAP = {
    "USA": "United States",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "BH": "Bosnia and Herzegovina",
    "DRC": "Democratic Republic of the Congo",
    "SA": "South Africa",
    "HK": "Hong Kong",
    "Rwandan": "Rwanda",
    "Macedonian": "Macedonia",
    "Antigua B": "Antigua and Barbuda",
    "Congo": "Republic of the Congo",
    "Ivory": "Cote d'Ivoire",
    "Saudi": "Saudi Arabia",
    "Saotome": "Sao Tome and Principe",
    "Solomon": "Solomon Islands",
    "Chadian": "Chad",
}
EXCLUDED = {"EU", "ASEAN", ".", ""}
MAP_LOCATION_ALIASES = {
    "Republic of the Congo": "Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "Macedonia": "North Macedonia",
}


# =============================================================================
# SECTION 2: DATA LOADING & NORMALISATION
# =============================================================================

def normalize_country(value: object) -> str:
    name = str(value).strip()
    name = NAME_MAP.get(name, name)
    return name


def map_location_name(country_name: str) -> str:
    return MAP_LOCATION_ALIASES.get(country_name, country_name)


def change_to_color(change: float, vmax: float) -> str:
    if vmax <= 0 or pd.isna(change):
        return "rgba(160,160,160,0.70)"
    ratio = min(abs(change) / vmax, 1.0)
    alpha = 0.35 + 0.55 * ratio
    return f"rgba(47,111,221,{alpha:.3f})" if change >= 0 else f"rgba(196,60,57,{alpha:.3f})"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and normalise the four input CSVs for both time periods.

    Applies country-name normalisation, numeric type coercion, and filters
    out non-country entities (e.g. ASEAN, EU).

    Returns:
        Tuple of (old_big, new_big, old_wide, new_wide) DataFrames.
    """
    old_big = pd.read_csv(OLD_BIG, encoding="utf-8-sig")
    new_big = pd.read_csv(NEW_BIG, encoding="utf-8-sig")
    old_wide = pd.read_csv(OLD_WIDE, encoding="utf-8-sig")
    new_wide = pd.read_csv(NEW_WIDE, encoding="utf-8-sig")

    for df in [old_big, new_big]:
        df["country"] = df["country"].map(normalize_country)
        df["topic_en"] = df["topic"].map(TOPIC_EN).fillna(df["topic"])
        df["period"] = "old" if df is old_big else "new"
        df["code"] = pd.to_numeric(df["code"], errors="coerce")
        df["average_score"] = pd.to_numeric(df["average_score"], errors="coerce")
        df["score_variance"] = pd.to_numeric(df["score_variance"], errors="coerce")

    for df in [old_wide, new_wide]:
        df["国家"] = df["国家"].map(normalize_country)
        df["总平均分"] = pd.to_numeric(df["总平均分"], errors="coerce")
        df["总句子数"] = pd.to_numeric(df["总句子数"], errors="coerce")
        df["总字数"] = pd.to_numeric(df["总字数"], errors="coerce")

    old_big = old_big[~old_big["country"].isin(EXCLUDED)].copy()
    new_big = new_big[~new_big["country"].isin(EXCLUDED)].copy()
    old_wide = old_wide[~old_wide["国家"].isin(EXCLUDED)].copy()
    new_wide = new_wide[~new_wide["国家"].isin(EXCLUDED)].copy()
    return old_big, new_big, old_wide, new_wide


# =============================================================================
# SECTION 3: COMPARISON TABLE CONSTRUCTION
# =============================================================================

def create_aligned_wide(old_wide: pd.DataFrame, new_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join the two wide-format country summary tables and compute
    overall score, sentence-count, and character-count changes.

    Args:
        old_wide: Wide-format summary for the earlier period (2015/16).
        new_wide: Wide-format summary for the later period (2022/23).

    Returns:
        Merged DataFrame with change columns appended, sorted by country.
    """
    old = old_wide.copy().rename(columns={"国家": "country"})
    new = new_wide.copy().rename(columns={"国家": "country"})
    comp = old.merge(new, on="country", how="outer", suffixes=("_old", "_new"))

    comp["old_year"] = np.nan
    comp["new_year"] = np.nan
    comp["overall_score_change"] = comp["总平均分_new"] - comp["总平均分_old"]
    comp["total_sentences_change"] = comp["总句子数_new"] - comp["总句子数_old"]
    comp["total_chars_change"] = comp["总字数_new"] - comp["总字数_old"]
    return comp.sort_values("country")


def create_bigtopic_change_table(old_big: pd.DataFrame, new_big: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a country × topic change table by outer-joining the two
    long-format summary files and computing score, sentence-count, and
    variance differences between periods.

    Args:
        old_big: Long-format topic summary for the earlier period.
        new_big: Long-format topic summary for the later period.

    Returns:
        DataFrame with one row per (country, topic) pair and delta columns.
    """
    old = old_big.rename(
        columns={
            "year": "old_year",
            "sentence_count": "sentence_count_old",
            "total_chinese_chars": "total_chars_old",
            "average_score": "average_score_old",
            "score_variance": "score_variance_old",
            "ratio_1": "ratio_1_old",
            "ratio_2": "ratio_2_old",
            "ratio_3": "ratio_3_old",
            "ratio_4": "ratio_4_old",
            "ratio_5": "ratio_5_old",
            "code": "code_old",
        }
    )
    new = new_big.rename(
        columns={
            "year": "new_year",
            "sentence_count": "sentence_count_new",
            "total_chinese_chars": "total_chars_new",
            "average_score": "average_score_new",
            "score_variance": "score_variance_new",
            "ratio_1": "ratio_1_new",
            "ratio_2": "ratio_2_new",
            "ratio_3": "ratio_3_new",
            "ratio_4": "ratio_4_new",
            "ratio_5": "ratio_5_new",
            "code": "code_new",
        }
    )
    keep_old = [
        "country", "topic", "topic_en", "old_year", "sentence_count_old", "total_chars_old",
        "average_score_old", "score_variance_old", "ratio_1_old", "ratio_2_old", "ratio_3_old",
        "ratio_4_old", "ratio_5_old", "code_old",
    ]
    keep_new = [
        "country", "topic", "topic_en", "new_year", "sentence_count_new", "total_chars_new",
        "average_score_new", "score_variance_new", "ratio_1_new", "ratio_2_new", "ratio_3_new",
        "ratio_4_new", "ratio_5_new", "code_new",
    ]
    comp = old[keep_old].merge(new[keep_new], on=["country", "topic", "topic_en"], how="outer")
    comp["score_change"] = comp["average_score_new"] - comp["average_score_old"]
    comp["sentence_count_change"] = comp["sentence_count_new"] - comp["sentence_count_old"]
    comp["total_chars_change"] = comp["total_chars_new"] - comp["total_chars_old"]
    comp["variance_change"] = comp["score_variance_new"] - comp["score_variance_old"]
    comp["code"] = comp["code_new"].fillna(comp["code_old"])
    return comp.sort_values(["topic", "country"])


def enrich_years(aligned_wide: pd.DataFrame, old_big: pd.DataFrame, new_big: pd.DataFrame) -> pd.DataFrame:
    old_years = old_big.groupby("country")["year"].max()
    new_years = new_big.groupby("country")["year"].max()
    aligned_wide["old_year"] = aligned_wide["country"].map(old_years)
    aligned_wide["new_year"] = aligned_wide["country"].map(new_years)
    return aligned_wide


# =============================================================================
# SECTION 4: TABLE EXPORT
# =============================================================================

def save_tables(aligned_wide: pd.DataFrame, bigtopic_changes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Derive and export all comparison tables to CSV files and an Excel workbook.

    Tables produced:
        aligned_country_overview       — full country-level merge
        overall_score_change_ranking   — countries ranked by overall score delta
        topic_change_summary           — per-topic aggregate statistics
        topic_top_increases            — top-10 improvers per topic
        topic_top_declines             — top-10 decliners per topic
        bigtopic_country_changes_full  — complete country × topic change matrix

    Args:
        aligned_wide:      Output of create_aligned_wide().
        bigtopic_changes:  Output of create_bigtopic_change_table().

    Returns:
        Dict mapping table name to DataFrame.
    """
    tables = {}
    tables["aligned_country_overview"] = aligned_wide
    common = aligned_wide.dropna(subset=["总平均分_old", "总平均分_new"]).copy()
    tables["overall_score_change_ranking"] = common[
        ["country", "old_year", "new_year", "总平均分_old", "总平均分_new", "overall_score_change",
         "总句子数_old", "总句子数_new", "total_sentences_change", "总字数_old", "总字数_new", "total_chars_change"]
    ].sort_values("overall_score_change", ascending=False)
    topic_means = (
        bigtopic_changes.groupby(["topic", "topic_en"], dropna=False)
        .agg(
            countries_compared=("country", lambda s: s.nunique()),
            mean_old_score=("average_score_old", "mean"),
            mean_new_score=("average_score_new", "mean"),
            mean_score_change=("score_change", "mean"),
            mean_old_sentences=("sentence_count_old", "mean"),
            mean_new_sentences=("sentence_count_new", "mean"),
            mean_sentence_change=("sentence_count_change", "mean"),
            mean_old_variance=("score_variance_old", "mean"),
            mean_new_variance=("score_variance_new", "mean"),
            mean_variance_change=("variance_change", "mean"),
        )
        .reset_index()
    )
    tables["topic_change_summary"] = topic_means

    top_up = (
        bigtopic_changes.dropna(subset=["score_change"])
        .sort_values(["topic", "score_change"], ascending=[True, False])
        .groupby("topic")
        .head(10)
        .reset_index(drop=True)
    )
    top_down = (
        bigtopic_changes.dropna(subset=["score_change"])
        .sort_values(["topic", "score_change"], ascending=[True, True])
        .groupby("topic")
        .head(10)
        .reset_index(drop=True)
    )
    tables["topic_top_increases"] = top_up
    tables["topic_top_declines"] = top_down

    tables["bigtopic_country_changes_full"] = bigtopic_changes

    with pd.ExcelWriter(OUT / "period_comparison_tables.xlsx") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            df.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")
    return tables


# =============================================================================
# SECTION 5: VISUALISATION
# =============================================================================

def plot_heatmap(bigtopic_changes: pd.DataFrame) -> None:
    common = bigtopic_changes.dropna(subset=["score_change"]).copy()
    pivot = common.pivot(index="country", columns="topic_en", values="score_change")
    topic_cols = [TOPIC_EN[t] for t in TOPIC_ORDER if TOPIC_EN[t] in pivot.columns]
    pivot = pivot[topic_cols]
    plt.figure(figsize=(12, max(12, len(pivot) * 0.18)))
    vmax = np.nanmax(np.abs(pivot.to_numpy(dtype=float)))
    sns.heatmap(pivot, cmap="RdBu", center=0, vmin=-vmax, vmax=vmax, linewidths=0.2)
    plt.title("LLM Big-Topic Score Change Heatmap (2022/23 minus 2015/16)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / "llm_bigtopic_change_heatmap.png", dpi=240)
    plt.close()


def plot_topic_bar_summary(topic_summary: pd.DataFrame) -> None:
    ordered = topic_summary.copy()
    ordered["topic_en"] = pd.Categorical(ordered["topic_en"], categories=[TOPIC_EN[t] for t in TOPIC_ORDER], ordered=True)
    ordered = ordered.sort_values("topic_en")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.barplot(data=ordered, x="mean_score_change", y="topic_en", color="#4e79a7", ax=axes[0])
    axes[0].set_title("Average Score Change by Topic")
    axes[0].set_xlabel("New minus old")
    axes[0].set_ylabel("")

    sns.barplot(data=ordered, x="mean_sentence_change", y="topic_en", color="#59a14f", ax=axes[1])
    axes[1].set_title("Average Sentence Count Change by Topic")
    axes[1].set_xlabel("New minus old")
    axes[1].set_ylabel("")

    sns.barplot(data=ordered, x="mean_variance_change", y="topic_en", color="#e15759", ax=axes[2])
    axes[2].set_title("Average Variance Change by Topic")
    axes[2].set_xlabel("New minus old")
    axes[2].set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / "topic_change_summary_bars.png", dpi=220)
    plt.close()


def plot_overall_map(aligned_wide: pd.DataFrame) -> None:
    common = aligned_wide.dropna(subset=["总平均分_old", "总平均分_new"]).copy()
    common["location"] = common["country"].map(map_location_name)
    common["change"] = common["overall_score_change"]
    fig = go.Figure(
        go.Choropleth(
            locations=common["location"],
            locationmode="country names",
            z=common["change"],
            text=common["country"],
            colorscale="RdBu",
            zmid=0,
            marker_line_color="white",
            marker_line_width=0.4,
            customdata=np.column_stack([
                common["old_year"],
                common["new_year"],
                common["总平均分_old"],
                common["总平均分_new"],
                common["总句子数_old"],
                common["总句子数_new"],
                common["总字数_old"],
                common["总字数_new"],
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Old year: %{customdata[0]}<br>"
                "New year: %{customdata[1]}<br>"
                "Old overall score: %{customdata[2]:.3f}<br>"
                "New overall score: %{customdata[3]:.3f}<br>"
                "Score change: %{z:+.3f}<br>"
                "Old sentences: %{customdata[4]:.0f}<br>"
                "New sentences: %{customdata[5]:.0f}<br>"
                "Old chars: %{customdata[6]:.0f}<br>"
                "New chars: %{customdata[7]:.0f}<extra></extra>"
            ),
            colorbar_title="Score change",
        )
    )
    fig.update_layout(
        title="Overall LLM Score Change Map (2022/23 minus 2015/16)",
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        width=1200,
        height=650,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    fig.write_html(FIGURES / "overall_score_change_map.html")


def plot_topic_map_dropdown(bigtopic_changes: pd.DataFrame) -> None:
    common = bigtopic_changes.dropna(subset=["score_change", "code"]).copy()
    topics = [t for t in TOPIC_ORDER if t in common["topic"].unique()]
    fig = go.Figure()
    buttons = []
    for i, topic in enumerate(topics):
        sub = common[common["topic"] == topic].copy()
        sub["location"] = sub["country"].map(map_location_name)
        z = sub["score_change"].to_numpy()
        visible = i == 0
        fig.add_trace(go.Choropleth(
            locations=sub["location"],
            locationmode="country names",
            z=z,
            text=sub["country"],
            colorscale="RdBu",
            zmid=0,
            visible=visible,
            marker_line_color="white",
            marker_line_width=0.4,
            customdata=np.column_stack([
                sub["old_year"],
                sub["new_year"],
                sub["average_score_old"],
                sub["average_score_new"],
                sub["sentence_count_old"],
                sub["sentence_count_new"],
                sub["score_variance_old"],
                sub["score_variance_new"],
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Old year: %{customdata[0]}<br>"
                "New year: %{customdata[1]}<br>"
                "Old score: %{customdata[2]:.3f}<br>"
                "New score: %{customdata[3]:.3f}<br>"
                "Score change: %{z:+.3f}<br>"
                "Old sentences: %{customdata[4]:.0f}<br>"
                "New sentences: %{customdata[5]:.0f}<br>"
                "Old variance: %{customdata[6]:.3f}<br>"
                "New variance: %{customdata[7]:.3f}<extra></extra>"
            ),
            colorbar_title="Score change",
        ))
        visibility = [False] * len(topics)
        visibility[i] = True
        buttons.append(dict(
            label=TOPIC_EN[topic],
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"{TOPIC_EN[topic]} Score Change Map (2022/23 minus 2015/16)"},
            ],
        ))
    fig.update_layout(
        title=f"{TOPIC_EN[topics[0]]} Score Change Map (2022/23 minus 2015/16)",
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.02, y=1.08)],
        width=1200,
        height=650,
        margin=dict(l=0, r=0, t=85, b=0),
    )
    fig.write_html(FIGURES / "topic_score_change_map.html")


def plot_topic_scatter_dropdown(bigtopic_changes: pd.DataFrame) -> None:
    common = bigtopic_changes.dropna(subset=["average_score_old", "average_score_new"]).copy()
    topics = [t for t in TOPIC_ORDER if t in common["topic"].unique()]
    fig = go.Figure()
    buttons = []
    for i, topic in enumerate(topics):
        sub = common[common["topic"] == topic].sort_values("country").copy()
        all_vals = list(sub["average_score_old"]) + list(sub["average_score_new"])
        axis_min = max(0, min(all_vals) * 0.95)
        axis_max = max(all_vals) * 1.08 if max(all_vals) > 0 else 1
        bubble_size = np.sqrt(sub["sentence_count_old"].fillna(0) + sub["sentence_count_new"].fillna(0)).clip(lower=4) * 2.4
        vmax = sub["score_change"].abs().max()
        colors = [change_to_color(val, vmax) for val in sub["score_change"]]
        hover_text = [
            f"<b>{row.country}</b><br>"
            f"Old year: {row.old_year}<br>"
            f"New year: {row.new_year}<br>"
            f"Old score: {row.average_score_old:.3f}<br>"
            f"New score: {row.average_score_new:.3f}<br>"
            f"Score change: {row.score_change:+.3f}<br>"
            f"Old sentences: {row.sentence_count_old:.0f}<br>"
            f"New sentences: {row.sentence_count_new:.0f}<br>"
            f"Old variance: {row.score_variance_old:.3f}<br>"
            f"New variance: {row.score_variance_new:.3f}<extra></extra>"
            for row in sub.itertuples(index=False)
        ]
        visible = i == 0
        fig.add_trace(go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            line=dict(color="gray", width=1, dash="dash"),
            hoverinfo="skip",
            name="No change",
            visible=visible,
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=sub["average_score_old"],
            y=sub["average_score_new"],
            mode="markers",
            marker=dict(
                size=bubble_size,
                color=colors,
                line=dict(width=0.8, color="rgba(0,0,0,0.35)")
            ),
            text=hover_text,
            hovertemplate="%{text}",
            name="Countries",
            visible=visible,
            showlegend=False,
        ))
        visibility = [False] * (len(topics) * 2)
        visibility[i * 2] = True
        visibility[i * 2 + 1] = True
        buttons.append(dict(
            label=TOPIC_EN[topic],
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title.text": f"<b>{TOPIC_EN[topic]}</b> Old vs New Scores",
                    "xaxis.range": [axis_min, axis_max],
                    "yaxis.range": [axis_min, axis_max],
                },
            ],
        ))
    first_topic = topics[0]
    first = common[common["topic"] == first_topic]
    first_vals = list(first["average_score_old"]) + list(first["average_score_new"])
    first_min = max(0, min(first_vals) * 0.95)
    first_max = max(first_vals) * 1.08 if max(first_vals) > 0 else 1
    fig.update_layout(
        title=dict(text=f"<b>{TOPIC_EN[first_topic]}</b> Old vs New Scores", font=dict(size=16)),
        xaxis=dict(title="2015/16 Score", range=[first_min, first_max], showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(title="2022/23 Score", range=[first_min, first_max], showgrid=True, gridcolor="#eeeeee", scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1040,
        height=760,
        margin=dict(l=80, r=240, t=95, b=70),
        updatemenus=[dict(buttons=buttons, direction="down", x=1.04, y=0.98, xanchor="left", yanchor="top")],
        annotations=[
            dict(x=1.04, y=1.04, xref="paper", yref="paper", text="Topic", showarrow=False, xanchor="left"),
            dict(x=0.98, y=0.08, xref="paper", yref="paper", text="Above line = increase<br>Below line = decrease", showarrow=False, align="right", font=dict(size=11, color="#555555"), xanchor="right"),
        ],
    )
    fig.write_html(FIGURES / "topic_score_scatter_all.html")


def plot_overall_bubbles(aligned_wide: pd.DataFrame) -> None:
    common = aligned_wide.dropna(subset=["总平均分_old", "总平均分_new"]).copy()
    all_vals = list(common["总平均分_old"]) + list(common["总平均分_new"])
    lo = max(0, min(all_vals) * 0.95)
    hi = max(all_vals) * 1.08
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="gray", width=1, dash="dash"), hoverinfo="skip", showlegend=False
    ))
    size = np.sqrt(common["总句子数_old"].fillna(0) + common["总句子数_new"].fillna(0)).clip(lower=4) * 2.5
    vmax = common["overall_score_change"].abs().max()
    colors = [change_to_color(v, vmax) for v in common["overall_score_change"]]
    hover = [
        f"<b>{row.country}</b><br>"
        f"Old year: {row.old_year}<br>New year: {row.new_year}<br>"
        f"Old overall score: {row.总平均分_old:.3f}<br>"
        f"New overall score: {row.总平均分_new:.3f}<br>"
        f"Score change: {row.overall_score_change:+.3f}<br>"
        f"Old sentences: {row.总句子数_old:.0f}<br>"
        f"New sentences: {row.总句子数_new:.0f}<br>"
        f"Old chars: {row.总字数_old:.0f}<br>"
        f"New chars: {row.总字数_new:.0f}<extra></extra>"
        for row in common.itertuples(index=False)
    ]
    fig.add_trace(go.Scatter(
        x=common["总平均分_old"], y=common["总平均分_new"], mode="markers",
        marker=dict(size=size, color=colors, line=dict(width=0.8, color="rgba(0,0,0,0.35)")),
        text=hover, hovertemplate="%{text}", showlegend=False
    ))
    fig.update_layout(
        title="Overall Average Score: 2015/16 vs 2022/23",
        xaxis=dict(title="2015/16 Score", range=[lo, hi], showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(title="2022/23 Score", range=[lo, hi], showgrid=True, gridcolor="#eeeeee", scaleanchor="x", scaleratio=1),
        plot_bgcolor="white", paper_bgcolor="white", width=900, height=780,
        annotations=[dict(x=0.98, y=0.08, xref="paper", yref="paper", text="Bubble size = total sentences in both periods", showarrow=False, xanchor="right")],
    )
    fig.write_html(FIGURES / "overall_score_scatter.html")


# =============================================================================
# SECTION 6: MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Orchestrate the full cross-period comparison workflow:
        1. Load and normalise data for both periods.
        2. Build aligned wide and long comparison tables.
        3. Export tables to CSV and Excel.
        4. Generate all static and interactive visualisations.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    old_big, new_big, old_wide, new_wide = load_data()
    aligned_wide = create_aligned_wide(old_wide, new_wide)
    aligned_wide = enrich_years(aligned_wide, old_big, new_big)
    bigtopic_changes = create_bigtopic_change_table(old_big, new_big)
    tables = save_tables(aligned_wide, bigtopic_changes)
    plot_heatmap(bigtopic_changes)
    plot_topic_bar_summary(tables["topic_change_summary"])
    plot_overall_map(aligned_wide)
    plot_topic_map_dropdown(bigtopic_changes)
    plot_topic_scatter_dropdown(bigtopic_changes)
    plot_overall_bubbles(aligned_wide)
    print("Tables:", TABLES)
    print("Figures:", FIGURES)
    print("Workbook:", OUT / "period_comparison_tables.xlsx")


if __name__ == "__main__":
    main()

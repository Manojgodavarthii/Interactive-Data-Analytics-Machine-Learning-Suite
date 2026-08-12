import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from modules.utils import *
from modules.ai_engine import (
    analyze_dataset, column_insight, cleaning_recommendations,
    important_columns, correlation_insight
)

def render():
    st.markdown('<div class="section-title">🤖 AI-Powered Insights</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state.df
    col_types = detect_column_types(df)
    stats, _ = get_summary_stats(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    text_cols = [c for c, t in col_types.items() if t == "text"]
    ai_stats = analyze_dataset(df, col_types)
    st.markdown(f'<div style="background:white;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);border:1px solid #e4e8f0;"><strong>📊 Dataset:</strong> {st.session_state.get("filename", "N/A")} &middot; <strong>📏 Size:</strong> {stats["rows"]} rows × {stats["columns"]} cols &middot; <strong>⭐ Quality:</strong> <span style="color:{"#22c55e" if ai_stats["quality"] == "excellent" else "#f59e0b" if ai_stats["quality"] == "fair" else "#ef4444"};font-weight:700;">{ai_stats["quality"].upper()}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Overall AI Summary</div>', unsafe_allow_html=True)
    quality_desc = {
        "excellent": "Your dataset is in excellent shape with minimal missing values and duplicates.",
        "fair": "Your dataset has some quality issues but can be cleaned for reliable analysis.",
        "poor": "Your dataset requires significant cleaning before reliable analysis can be performed."
    }
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1a2e0d,#0f346008);border-left:4px solid #0f3460;border-radius:0 12px 12px 0;padding:1rem 1.2rem;font-size:0.9rem;color:#111827;">'
        f'<strong>📋 Dataset Overview:</strong> This dataset contains <strong>{stats["rows"]} rows</strong> and <strong>{stats["columns"]} columns</strong>. '
        f'It includes <strong>{stats["numeric_cols"]} numeric</strong>, <strong>{stats["categorical_cols"]} categorical</strong>, '
        f'<strong>{stats["date_cols"]} date</strong>, and <strong>{len(text_cols)} text</strong> columns.<br><br>'
        f'<strong>⭐ Quality Assessment:</strong> {quality_desc[ai_stats["quality"]]} '
        f'({ai_stats["missing_pct"]}% missing, {ai_stats["duplicate_pct"]}% duplicates).'
        f'</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">⭐ AI-Identified Important Columns</div>', unsafe_allow_html=True)
    imp = important_columns(df, col_types)
    for col, score, dtype in imp:
        st.markdown(f'<div class="insight-box"><strong>{col}</strong> (importance score: {score}) — type: {dtype}. {column_insight(col, df[col], dtype)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔗 AI Correlation Insights</div>', unsafe_allow_html=True)
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.unstack().dropna().sort_values(key=abs, ascending=False)
        if len(pairs) > 0:
            for i in range(min(3, len(pairs))):
                a, b = pairs.index[i]
                st.markdown(f'<div class="insight-box">{correlation_insight(a, b, pairs.iloc[i], "pearson")}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-box">Not enough numeric columns for correlation analysis.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Distribution Insights</div>', unsafe_allow_html=True)
    for col in num_cols[:4]:
        s = df[col].dropna()
        if len(s) > 0:
            ins = column_insight(col, s, "numeric")
            st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)
    for col in cat_cols[:3]:
        s = df[col].dropna()
        ins = column_insight(col, s, "categorical")
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧹 AI Cleaning Recommendations</div>', unsafe_allow_html=True)
    for rec in cleaning_recommendations(df, col_types):
        st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💼 Business Insights</div>', unsafe_allow_html=True)
    bus_insights = []
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.unstack().dropna()
        if len(pairs) > 0:
            best = pairs.abs().idxmax()
            cv = corr.loc[best[0], best[1]]
            bus_insights.append(f"<strong>{best[0]}</strong> and <strong>{best[1]}</strong> are {'strongly' if abs(cv) > 0.7 else 'moderately'} correlated ({cv:.2f}) — {'consider this relationship in decision-making.' if abs(cv) > 0.5 else 'but the relationship is not strong enough for direct prediction.'}")
    for col in cat_cols[:2]:
        top_val = df[col].value_counts().index[0]
        top_pct = df[col].value_counts().iloc[0] / len(df) * 100
        bus_insights.append(f"<strong>'{top_val}'</strong> dominates <strong>{col}</strong> ({top_pct:.0f}% of records) — this is the most common category.")
    for col in num_cols[:2]:
        s = df[col].dropna()
        if len(s) > 0:
            bus_insights.append(f"The average <strong>{col}</strong> is <strong>{s.mean():.2f}</strong> (range: {s.min():.2f}–{s.max():.2f}). Typical values cluster around {s.median():.2f}.")
    for ins in bus_insights:
        st.markdown(f'<div class="insight-box">💡 {ins}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">➡️ AI Recommended Next Steps</div>', unsafe_allow_html=True)
    next_steps = []
    if ai_stats["missing_pct"] > 5 or ai_stats["duplicate_pct"] > 0:
        next_steps.append("🧹 <strong>Clean your data</strong> first — visit Data Cleaning page")
    if date_cols and num_cols:
        next_steps.append("📈 Perform <strong>Time-Series Analysis</strong> in Advanced Analytics")
    if len(num_cols) >= 2:
        next_steps.append("🔗 Run <strong>Correlation Analysis</strong> to quantify relationships")
    if cat_cols and num_cols:
        next_steps.append("⚙️ Use <strong>Custom Analysis</strong> to group by categories")
    next_steps.append("📝 <strong>Generate a Report</strong> with all findings — click Report Generation")
    next_steps.append("🎨 Create <strong>Visualizations</strong> to explore patterns")
    for n in next_steps:
        st.markdown(f'<div class="insight-box" style="border-left-color:#7c3aed;">➡️ {n}</div>', unsafe_allow_html=True)

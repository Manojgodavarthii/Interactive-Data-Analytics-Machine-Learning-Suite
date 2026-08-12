import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types


def _score_analysis(analysis, df, col_types):
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    score = 0
    details = {}
    if analysis == "Correlation":
        score = min(100, len(num_cols) * 20)
        details = {"why": "Find relationships between numeric variables", "columns": num_cols[:6], "output": "Correlation matrix + scatter plots"}
    elif analysis == "Trend Analysis":
        score = min(100, len(date_cols) * 30)
        details = {"why": "Analyze patterns over time", "columns": date_cols[:3] + num_cols[:3], "output": "Line charts with trends"}
    elif analysis == "Dashboard":
        score = min(100, (len(num_cols) + len(cat_cols)) * 10)
        details = {"why": "Get a complete overview of your data", "columns": df.columns.tolist()[:8], "output": "KPI cards + charts + insights"}
    elif analysis == "Outlier Detection":
        score = min(100, len(num_cols) * 20)
        details = {"why": "Identify unusual values in numeric data", "columns": num_cols[:6], "output": "Box plots + anomaly scores"}
    elif analysis == "Forecast":
        score = min(90, len(date_cols) * 25 + len(num_cols) * 5)
        details = {"why": "Predict future values based on historical data", "columns": (date_cols[:1] if date_cols else []) + num_cols[:2], "output": "Forecast chart + confidence intervals"}
    elif analysis == "PCA":
        score = min(80, len(num_cols) * 15)
        details = {"why": "Reduce dimensionality and find patterns", "columns": num_cols[:8], "output": "PCA scatter + variance plot"}
    elif analysis == "Clustering":
        score = min(80, len(num_cols) * 15)
        details = {"why": "Group similar data points together", "columns": num_cols[:6], "output": "Cluster scatter + silhouette score"}
    elif analysis == "Distribution":
        score = min(95, len(num_cols) * 15 + len(cat_cols) * 10)
        details = {"why": "Understand data distribution and spread", "columns": (num_cols + cat_cols)[:6], "output": "Histograms + box plots"}
    elif analysis == "Category Analysis":
        score = min(90, len(cat_cols) * 25)
        details = {"why": "Analyze categorical variable patterns", "columns": cat_cols[:6], "output": "Bar charts + frequency tables"}
    elif analysis == "Feature Importance":
        score = min(85, len(num_cols) * 15)
        details = {"why": "Identify most influential variables", "columns": num_cols[:8], "output": "Feature importance chart + ranking"}
    return score, details


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F4A1;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">Smart Analysis Recommendations</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#6b7280;font-size:0.85rem;margin-bottom:1rem;">{len(num_cols)} numeric · {len(cat_cols)} categorical columns</div>', unsafe_allow_html=True)

    analyses = ["Correlation", "Trend Analysis", "Dashboard", "Outlier Detection", "Forecast", "Distribution", "Category Analysis", "Feature Importance", "PCA", "Clustering"]
    scored = []
    for a in analyses:
        s, d = _score_analysis(a, df, col_types)
        scored.append((s, a, d))
    scored.sort(key=lambda x: -x[0])

    for score, name, details in scored:
        stars = "\u2605" * round(score / 20) + "\u2606" * (5 - round(score / 20))
        color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#9ca3af"
        with st.container():
            st.markdown(f'<div style="background:white;border:1px solid #e4e8f0;border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.6rem;"><div style="display:flex;align-items:center;gap:0.5rem;"><span style="font-weight:700;font-size:0.95rem;color:#1a1a2e;">{name}</span><span style="color:{color};font-size:0.85rem;">{stars}</span><span style="margin-left:auto;font-size:0.7rem;background:{color}20;color:{color};padding:0.1rem 0.4rem;border-radius:4px;font-weight:600;">{score}%</span></div><div style="font-size:0.78rem;color:#6b7280;margin-top:0.3rem;">{details.get("why", "")}</div><div style="font-size:0.72rem;color:#9ca3af;margin-top:0.15rem;">Columns: {", ".join(details.get("columns", [])[:4])} &middot; Output: {details.get("output", "")}</div></div>', unsafe_allow_html=True)

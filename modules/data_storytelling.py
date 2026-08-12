import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types, get_summary_stats


def _build_executive_narrative(df, col_types):
    """Synthesize KPI bullets, risks, and action items for reporting."""
    stats, _ = get_summary_stats(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]

    kpis = []
    for col in num_cols[:5]:
        s = df[col].dropna()
        if len(s) > 0:
            kpis.append(f"<strong>{col}</strong>: avg {s.mean():,.2f} · total {s.sum():,.2f} · range [{s.min():,.2f}, {s.max():,.2f}]")
    for col in cat_cols[:2]:
        top = df[col].value_counts().index[0]
        pct = round(df[col].value_counts().iloc[0] / len(df) * 100, 1)
        kpis.append(f"<strong>{col}</strong>: most common '{top}' ({pct}% of records)")

    risks = []
    if stats["missing"] > 0:
        risks.append(f"<strong>{stats['missing']:,}</strong> missing values — may bias downstream analysis")
    if stats["duplicates"] > 0:
        risks.append(f"<strong>{stats['duplicates']:,}</strong> duplicate rows — can inflate metrics")
    for col in num_cols[:3]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out > len(df) * 0.01:
            risks.append(f"<strong>{n_out}</strong> potential outliers in <strong>{col}</strong> ({n_out/len(df)*100:.1f}% of records)")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        mx = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).unstack().dropna()
        if not mx.empty and mx.max() > 0.85:
            a, b = mx.idxmax()
            risks.append(f"High collinearity between <strong>{a}</strong> and <strong>{b}</strong> (r={mx.max():.2f})")

    actions = []
    if stats["missing"] > 0:
        actions.append("Impute or drop records with missing values before modelling")
    if stats["duplicates"] > 0:
        actions.append("De-duplicate the dataset to avoid inflated statistics")
    if num_cols:
        for col in num_cols[:1]:
            actions.append(f"Investigate drivers of {col} — start with the strongest correlated features")
    if date_cols:
        actions.append(f"Use '{date_cols[0]}' for time-based trend and seasonality review")
    if cat_cols:
        actions.append(f"Segment performance by '{cat_cols[0]}' to surface best/worst groups")
    if not actions:
        actions.append("Dataset is healthy — focus on deriving business KPIs and tracking them over time")

    return {
        "kpis": kpis[:6],
        "risks": risks[:5] or ["No material risks detected — data quality looks sound"],
        "actions": actions[:5],
    }


def _generate_story(df, col_types):
    stats, _ = get_summary_stats(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    story = {}

    exec_parts = []
    if num_cols:
        for col in num_cols[:3]:
            trend = "increased" if df[col].iloc[-1] > df[col].iloc[0] else "decreased"
            exec_parts.append(f"{col.capitalize()} has {trend} over the dataset period.")
    cat_col = cat_cols[0] if cat_cols else None
    if cat_col:
        top_cat = df[cat_col].value_counts().index[0]
        exec_parts.append(f"{cat_col.capitalize()} '{top_cat}' has the highest frequency.")
    if len(num_cols) >= 2:
        max_col = df[num_cols].sum().idxmax()
        exec_parts.append(f"{max_col.capitalize()} contributes the most to the total.")
    story["executive_summary"] = exec_parts[:4]

    findings = []
    if date_cols and num_cols:
        finding = f"Sales peak in certain periods based on {date_cols[0]} trends."
        findings.append(finding)
    for col in num_cols[:3]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        findings.append(f"{col.capitalize()} ranges from {df[col].min():.1f} to {df[col].max():.1f} (IQR: {q1:.1f} - {q3:.1f}).")
    for col in cat_cols[:2]:
        top = df[col].value_counts().index[0]
        pct = round(df[col].value_counts().iloc[0] / len(df) * 100, 1)
        findings.append(f"'{top}' is the most common {col} ({pct}% of records).")
    story["key_findings"] = findings[:6]

    recs = []
    if stats["missing"] > 0:
        recs.append("Address missing values to improve data quality")
    if stats["duplicates"] > 0:
        recs.append("Remove duplicate records for accurate analysis")
    if num_cols:
        for col in num_cols[:2]:
            low = df[col].quantile(0.05)
            high = df[col].quantile(0.95)
            recs.append(f"Investigate outliers in {col} (outside {low:.1f} - {high:.1f} range)")
    avg_cols = [c for c in num_cols if "revenue" in c.lower() or "sales" in c.lower() or "profit" in c.lower() or "income" in c.lower()]
    if avg_cols:
        recs.append(f"Focus on improving {avg_cols[0]} performance")
    story["recommendations"] = recs[:4]

    conclusion = f"The dataset contains {stats['rows']:,} records with {stats['columns']} columns. "
    if num_cols:
        conclusion += f"Key numeric variables include {', '.join(num_cols[:4])}. "
    if cat_cols:
        conclusion += f"Categorical insights from {', '.join(cat_cols[:3])}. "
    conclusion += f"Data quality is {'good' if stats['missing'] < stats['rows'] * 0.1 else 'needs improvement'} with {stats['missing']} missing values and {stats['duplicates']} duplicates."
    story["conclusion"] = conclusion
    return story


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)
    story = _generate_story(df, col_types)

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F4D6;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">Data Storytelling</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.85rem;margin-bottom:1rem;">Converting numbers into actionable business insights</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-weight:700;font-size:1.05rem;color:#1a1a2e;margin-bottom:0.5rem;">Executive Summary</div>', unsafe_allow_html=True)
    for part in story["executive_summary"]:
        st.markdown(f'<div style="background:linear-gradient(135deg,#eef2ff,#e0e7ff);border-radius:10px;padding:0.6rem 0.9rem;font-size:0.88rem;color:#3730a3;margin-bottom:0.3rem;">&#x1F4C8; {part}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-weight:700;font-size:1.05rem;color:#1a1a2e;margin:1.2rem 0 0.5rem 0;">Key Findings</div>', unsafe_allow_html=True)
    for finding in story["key_findings"]:
        st.markdown(f'<div style="background:#f8faff;border-left:3px solid #6366f1;border-radius:0 8px 8px 0;padding:0.5rem 0.8rem;font-size:0.84rem;color:#111827;margin-bottom:0.3rem;">&#x1F50D; {finding}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-weight:700;font-size:1.05rem;color:#1a1a2e;margin:1.2rem 0 0.5rem 0;">Business Recommendations</div>', unsafe_allow_html=True)
    for rec in story["recommendations"]:
        st.markdown(f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:0.5rem 0.8rem;font-size:0.84rem;color:#166534;margin-bottom:0.3rem;">&#x2705; {rec}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-weight:700;font-size:1.05rem;color:#1a1a2e;margin:1.2rem 0 0.5rem 0;">Conclusion</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:linear-gradient(135deg,#f0f4ff,#eef2ff);border-radius:12px;padding:1rem;font-size:0.88rem;color:#1e1b4b;line-height:1.6;">{story["conclusion"]}</div>', unsafe_allow_html=True)

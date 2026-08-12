import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types, get_summary_stats


def _detect_issues(df, col_types):
    issues = []
    stats, _ = get_summary_stats(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]

    if stats["missing"] > 0:
        max_miss = df.isnull().sum().idxmax()
        issues.append({"issue": f"High missing values in '{max_miss}'", "detail": f"{df[max_miss].isnull().sum():,} missing values ({df[max_miss].isnull().mean()*100:.1f}%)", "priority": "High", "impact": "High", "recs": ["Fill missing values with mean/median", "Consider dropping column if >50% missing", "Use imputation techniques"]})

    if stats["duplicates"] > 0:
        issues.append({"issue": f"Duplicate records found", "detail": f"{stats['duplicates']:,} duplicate rows ({stats['duplicates']/stats['rows']*100:.1f}%)", "priority": "Medium", "impact": "Medium", "recs": ["Remove duplicate rows", "Investigate source of duplicates", "Add unique constraints"]})

    for col in num_cols[:3]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((df[col] < low) | (df[col] > high)).sum()
        if outliers > len(df) * 0.02:
            issues.append({"issue": f"Outliers detected in '{col}'", "detail": f"{outliers:,} outliers ({outliers/len(df)*100:.1f}% of values)", "priority": "Medium", "impact": "Medium", "recs": ["Cap extreme values at percentiles", "Use robust scaling methods", "Investigate outlier causes"]})

    for col in df.select_dtypes(include="object").columns[:3]:
        if df[col].nunique() > 50:
            issues.append({"issue": f"High cardinality in '{col}'", "detail": f"{df[col].nunique():,} unique values", "priority": "Low", "impact": "Low", "recs": ["Group rare categories", "Use feature hashing", "Consider target encoding"]})

    return issues[:5]


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)
    issues = _detect_issues(df, col_types)

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F3AF;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">AI Action Recommendations</span></div>', unsafe_allow_html=True)

    if not issues:
        st.markdown('<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:1.5rem;text-align:center;font-size:0.9rem;color:#166534;">&#x2705; No issues detected. Your dataset looks great!</div>', unsafe_allow_html=True)
        return

    for item in issues:
        priority_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#6b7280"}.get(item["priority"], "#6b7280")
        impact_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(item["impact"], "#6b7280")
        with st.container():
            st.markdown(f'<div style="background:white;border:1px solid #e4e8f0;border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem;"><div style="display:flex;align-items:center;gap:0.5rem;"><span style="font-weight:700;font-size:0.9rem;color:#1a1a2e;">&#x26A0;&#xFE0F; {item["issue"]}</span></div><div style="font-size:0.8rem;color:#6b7280;margin-top:0.2rem;">{item["detail"]}</div><div style="display:flex;gap:1rem;margin-top:0.4rem;font-size:0.72rem;"><span>Priority: <span style="color:{priority_color};font-weight:600;">{item["priority"]}</span></span><span>Impact: <span style="color:{impact_color};font-weight:600;">{item["impact"]}</span></span></div><div style="margin-top:0.4rem;"><span style="font-size:0.75rem;font-weight:600;color:#059669;">Recommendations:</span><div style="font-size:0.78rem;color:#374151;margin-top:0.15rem;">{" | ".join(item["recs"])}</div></div></div>', unsafe_allow_html=True)

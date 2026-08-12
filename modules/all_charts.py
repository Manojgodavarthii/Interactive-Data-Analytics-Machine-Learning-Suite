import streamlit as st
import pandas as pd
from modules.utils import detect_column_types, get_summary_stats, auto_insights, auto_charts, render_charts_grid


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)
    stats, _ = get_summary_stats(df)
    insights = auto_insights(df, col_types)
    charts = auto_charts(df, col_types)

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F4CA;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">All Charts Gallery</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#6b7280;font-size:0.85rem;margin-bottom:1rem;">{stats["size"]} · {stats["missing"]} missing · {stats["duplicates"]} duplicates</div>', unsafe_allow_html=True)

    if insights:
        st.markdown('<div style="background:linear-gradient(135deg,#eef2ff,#e0e7ff);border-radius:12px;padding:0.8rem 1rem;margin-bottom:1.2rem;font-size:0.85rem;color:#3730a3;">&#x1F916; <strong>AI Dataset Insights</strong><br>' + '<br>'.join(insights) + '</div>', unsafe_allow_html=True)

    if charts:
        items = []
        for title, fig in charts:
            insight_text = title.replace("\u2014", "-").replace("\u2013", "-")
            items.append((fig, f"all_{len(items)}", insight_text))
        render_charts_grid(items, section_key="all_charts")
    else:
        st.info("Not enough columns to generate charts.")

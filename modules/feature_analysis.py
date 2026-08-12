import streamlit as st
import pandas as pd
import plotly.express as px
from modules.utils import *
from modules.ai_engine import column_insight, important_columns

def render():
    st.markdown('<div class="section-title">🔍 Feature Analysis</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state.df
    col_types = detect_column_types(df)
    st.markdown('<div class="section-title">⭐ AI-Identified Important Columns</div>', unsafe_allow_html=True)
    imp = important_columns(df, col_types)
    cols = st.columns(min(5, len(imp)))
    for i, (col, score, dtype) in enumerate(imp):
        with cols[i % len(cols)]:
            st.markdown(f'<div style="background:linear-gradient(135deg,#1a1a2e,#0f3460);color:white;border-radius:12px;padding:0.8rem;text-align:center;"><div style="font-size:1.3rem;font-weight:700;">{score}</div><div style="font-size:0.7rem;opacity:0.7;">{col}</div><div style="font-size:0.65rem;opacity:0.5;">{dtype}</div></div>', unsafe_allow_html=True)
    col = st.selectbox("Select a column to analyze in depth", df.columns)
    if not col:
        return
    dtype = col_types.get(col, "text")
    st.markdown(f'<div style="background:white;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);border:1px solid #e4e8f0;"><h4 style="margin:0 0 0.3rem 0;">{col}</h4><span class="status-badge active" style="font-size:0.75rem;">{dtype.upper()}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:linear-gradient(135deg,#1a1a2e0d,#0f346008);border-left:4px solid #0f3460;border-radius:0 12px 12px 0;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.88rem;color:#111827;"><span style="font-weight:700;color:#0f3460;">🤖 AI Analysis:</span> {column_insight(col, df[col], dtype)}</div>', unsafe_allow_html=True)
    if dtype == "numeric":
        s = df[col].dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#4CAF50;"><div class="metric-value">{s.mean():.2f}</div><div class="metric-label">Mean</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#2196F3;"><div class="metric-value">{s.median():.2f}</div><div class="metric-label">Median</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left-color:#ff9800;"><div class="metric-value">{s.std():.2f}</div><div class="metric-label">Std Dev</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card" style="border-left-color:#7c3aed;"><div class="metric-value">{s.var():.2f}</div><div class="metric-label">Variance</div></div>', unsafe_allow_html=True)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#f44336;"><div class="metric-value">{s.min():.2f}</div><div class="metric-label">Min</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#0891b2;"><div class="metric-value">{s.max():.2f}</div><div class="metric-label">Max</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left-color:#059669;"><div class="metric-value">{int((abs((s - s.mean()) / s.std()) > 3).sum())}</div><div class="metric-label">Outliers</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card" style="border-left-color:#6b7280;"><div class="metric-value">{s.nunique()}</div><div class="metric-label">Unique</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(df, x=col, marginal="box", title=f"Histogram of {col}", color_discrete_sequence=["#0f3460"])
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
            fig1.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
            fig1.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
            render_chart(fig1, f"feat_hist_{col}")
        with c2:
            fig2 = px.box(df, y=col, title=f"Box Plot of {col}", color_discrete_sequence=["#0f3460"])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
            fig2.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
            fig2.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
            render_chart(fig2, f"feat_box_{col}")
        skew = s.skew()
        if abs(skew) < 0.5:
            interp = "approximately normally distributed"
        elif skew > 0:
            interp = "right-skewed (positive skew)"
        else:
            interp = "left-skewed (negative skew)"
        st.markdown(f'<div class="insight-box"><strong>{col}:</strong> The distribution is {interp}.</div>', unsafe_allow_html=True)
    elif dtype in ["categorical", "boolean"]:
        freq = df[col].value_counts().reset_index()
        freq.columns = [col, "Count"]
        freq["Percentage"] = round(freq["Count"] / freq["Count"].sum() * 100, 2)
        st.dataframe(freq, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(freq, x=col, y="Count", title=f"Frequency of {col}", color_discrete_sequence=["#0f3460"])
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
            fig1.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
            fig1.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
            render_chart(fig1, f"feat_bar_{col}")
        with c2:
            fig2 = px.pie(freq, names=col, values="Count", title=f"Distribution of {col}")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
            render_chart(fig2, f"feat_pie_{col}")
        most = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
        least = df[col].value_counts().index[-1] if len(df[col].value_counts()) > 0 else "N/A"
        st.markdown(f'<div class="insight-box"><strong>Most frequent:</strong> {most} &nbsp;|&nbsp; <strong>Least frequent:</strong> {least}</div>', unsafe_allow_html=True)
    elif dtype == "date":
        dates = pd.to_datetime(df[col], errors="coerce").dropna()
        if len(dates) > 0:
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="metric-card" style="border-left-color:#4CAF50;"><div class="metric-value">{dates.min().date()}</div><div class="metric-label">Earliest</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card" style="border-left-color:#2196F3;"><div class="metric-value">{dates.max().date()}</div><div class="metric-label">Latest</div></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="metric-card" style="border-left-color:#ff9800;"><div class="metric-value">{dates.dt.year.nunique()}</div><div class="metric-label">Years Covered</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card" style="border-left-color:#7c3aed;"><div class="metric-value">{(dates.max() - dates.min()).days}</div><div class="metric-label">Total Days</div></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.histogram(dates, x=dates, title=f"Distribution of {col}", color_discrete_sequence=["#0f3460"])
                fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
                fig1.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
                fig1.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
                render_chart(fig1, f"feat_date_hist_{col}")
            with c2:
                monthly = dates.dt.to_period("M").value_counts().sort_index()
                fig2 = px.line(monthly.reset_index(), x=0, y=monthly.values,
                               title=f"Monthly Trend of {col}", labels={"0": "Month", "y": "Count"},
                               color_discrete_sequence=["#0f3460"])
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
                fig2.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
                fig2.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
                render_chart(fig2, f"feat_date_trend_{col}")
    elif dtype == "text":
        texts = df[col].dropna().astype(str)
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#4CAF50;"><div class="metric-value">{len(texts)}</div><div class="metric-label">Total Texts</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#2196F3;"><div class="metric-value">{texts.str.len().mean():.1f}</div><div class="metric-label">Avg Length</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#ff9800;"><div class="metric-value">{texts.str.len().min()}</div><div class="metric-label">Shortest</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#7c3aed;"><div class="metric-value">{texts.str.len().max()}</div><div class="metric-label">Longest</div></div>', unsafe_allow_html=True)
        st.write("**Sample entries:**")
        st.dataframe(texts.head(20).reset_index(drop=True), use_container_width=True)

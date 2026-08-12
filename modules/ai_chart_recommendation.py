import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types
import plotly.express as px


def _recommend_chart(selected_cols, col_types):
    num = [c for c in selected_cols if col_types.get(c) == "numeric"]
    cat = [c for c in selected_cols if col_types.get(c) == "categorical"]
    date = [c for c in selected_cols if col_types.get(c) == "date"]
    total = len(selected_cols)

    if total == 0:
        return None, "Select at least one column to get a recommendation."
    if total == 1:
        col = selected_cols[0]
        if col in num:
            return "Histogram", f"'{col}' is a numeric column. A histogram shows its distribution, revealing skewness, peaks, and outliers."
        elif col in cat:
            return "Bar Chart (Value Counts)", f"'{col}' is a categorical column. A bar chart of value counts shows category frequency."
        elif col in date:
            return "Time Series Line", f"'{col}' is a date column. A line chart over time shows trends and seasonality."
        else:
            return "Bar Chart", f"'{col}' is a text column. A bar chart of value counts shows the distribution."
    if total == 2:
        a, b = selected_cols
        if a in num and b in num:
            return "Scatter Plot", f"'{a}' vs '{b}' — both numeric. A scatter plot reveals correlation, clusters, and outliers."
        if a in cat and b in num:
            return "Bar Chart", f"'{a}' (categorical) vs '{b}' (numeric). Bars show average {b} per category."
        if a in num and b in cat:
            return "Bar Chart", f"'{b}' (categorical) vs '{a}' (numeric). Bars show average {a} per category."
        if a in date and b in num:
            return "Line Chart", f"'{a}' (date) vs '{b}' (numeric). A line chart shows {b} trend over time."
        if a in num and b in date:
            return "Line Chart", f"'{b}' (date) vs '{a}' (numeric). A line chart shows {a} trend over time."
        if a in cat and b in cat:
            return "Stacked Bar / Heatmap", f"'{a}' vs '{b}' — both categorical. A stacked bar or heatmap shows the relationship between categories."
        if a in date and b in cat:
            return "Stacked Area / Line", f"'{a}' (date) vs '{b}' (category). A stacked area chart shows category proportion over time."
        if a in cat and b in date:
            return "Stacked Area / Line", f"'{b}' (date) vs '{a}' (category). A stacked area chart shows category proportion over time."
        return "Scatter Plot", f"A scatter plot between '{a}' and '{b}' reveals their relationship."
    if total >= 3:
        if len(num) >= 3 and len(cat) == 0:
            return "Correlation Heatmap", f"{len(num)} numeric columns selected. A correlation heatmap shows pairwise relationships."
        if len(num) >= 2 and len(cat) >= 1:
            return "Grouped Bar / Scatter Matrix", f"{len(num)} numeric + {len(cat)} categorical columns. A scatter matrix reveals multi-dimensional patterns."
        if len(cat) >= 2:
            return "Heatmap / Parallel Categories", f"Multiple categorical columns. A parallel categories diagram shows flow between categories."
        if len(date) >= 1 and len(num) >= 1:
            return "Multi-Line Chart", f"Date + {len(num)} numeric columns. A multi-line chart compares trends over time."
        return "Multi-Chart Grid", "Multiple columns selected. Consider a dashboard of individual charts rather than one combined view."
    return "Bar Chart", "A bar chart is a safe default for your selected columns."


def _build_chart(recommendation, selected_cols, df, col_types):
    num = [c for c in selected_cols if col_types.get(c) == "numeric"]
    cat = [c for c in selected_cols if col_types.get(c) == "categorical"]
    date = [c for c in selected_cols if col_types.get(c) == "date"]

    try:
        if recommendation == "Histogram" and num:
            fig = px.histogram(df, x=num[0], title=f"Distribution of {num[0]}", marginal="box")
        elif recommendation == "Bar Chart (Value Counts)" and cat:
            counts = df[cat[0]].value_counts().head(20).reset_index()
            counts.columns = [cat[0], "Count"]
            fig = px.bar(counts, x=cat[0], y="Count", title=f"Value Counts for {cat[0]}", text_auto=True)
        elif recommendation == "Scatter Plot" and len(num) >= 2:
            fig = px.scatter(df, x=num[0], y=num[1], title=f"{num[0]} vs {num[1]}", trendline="ols" if len(df) > 10 else None)
        elif recommendation == "Bar Chart" and len(cat) >= 1 and len(num) >= 1:
            fig = px.bar(df.groupby(cat[0])[num[0]].mean().sort_values(ascending=False).head(15).reset_index(), x=cat[0], y=num[0], title=f"Average {num[0]} by {cat[0]}", text_auto=".1f", color=num[0], color_continuous_scale="Blues")
        elif recommendation == "Line Chart" and len(date) >= 1 and len(num) >= 1:
            sorted_df = df.sort_values(date[0])
            fig = px.line(sorted_df, x=date[0], y=num[0], title=f"{num[0]} over Time")
        elif recommendation == "Stacked Bar / Heatmap" and len(cat) >= 2:
            ctab = pd.crosstab(df[cat[0]], df[cat[1]])
            fig = px.imshow(ctab, text_auto=True, title=f"{cat[0]} vs {cat[1]}", aspect="auto", color_continuous_scale="Viridis")
        elif recommendation == "Stacked Area / Line" and len(date) >= 1 and len(cat) >= 1:
            grouped = df.groupby([date[0], cat[0]]).size().reset_index(name="count")
            fig = px.area(grouped, x=date[0], y="count", color=cat[0], title=f"Category Distribution over Time")
        elif recommendation == "Correlation Heatmap" and len(num) >= 3:
            corr = df[num].corr()
            fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", aspect="auto", color_continuous_scale="RdYlGn", zmin=-1, zmax=1)
        elif recommendation == "Grouped Bar / Scatter Matrix" and len(num) >= 2 and len(cat) >= 1:
            fig = px.scatter_matrix(df, dimensions=num[:4], color=cat[0], title=f"Scatter Matrix colored by {cat[0]}")
        elif recommendation == "Multi-Line Chart" and len(date) >= 1 and len(num) >= 1:
            melted = df.melt(id_vars=[date[0]], value_vars=num[:5], var_name="Metric", value_name="Value")
            sorted_melt = melted.sort_values(date[0])
            fig = px.line(sorted_melt, x=date[0], y="Value", color="Metric", title="Metrics over Time")
        elif recommendation == "Time Series Line" and len(date) >= 1:
            fig = px.line(df.sort_values(date[0]), x=date[0], y=df.index if len(selected_cols) == 1 else selected_cols[0], title=f"{date[0]} over Time")
        else:
            fig = px.bar(df[selected_cols].head(30), title="Data Preview")
        fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=30))
        return fig
    except Exception:
        fig = px.bar(df[selected_cols].head(30), title="Preview (fallback)")
        fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=30))
        return fig


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F4CA;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">AI Chart Recommendation</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.85rem;margin-bottom:0.5rem;">Select columns and let AI recommend the best chart type</div>', unsafe_allow_html=True)

    all_cols = df.columns.tolist()
    selected_cols = st.multiselect("Select columns for visualization", all_cols, default=all_cols[:min(3, len(all_cols))])

    if not selected_cols:
        st.info("Select at least one column to get a chart recommendation.")
        return

    rec, reason = _recommend_chart(selected_cols, col_types)

    st.markdown(f'<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:0.8rem 1rem;margin:0.5rem 0;"><div style="font-weight:700;font-size:1rem;color:#0369a1;">&#x1F4A1; Recommended: {rec}</div><div style="font-size:0.82rem;color:#475569;margin-top:0.3rem;">{reason}</div></div>', unsafe_allow_html=True)

    fig = _build_chart(rec, selected_cols, df, col_types)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

    with st.expander("Chart Recommendation Rules", expanded=False):
        rules = [
            ("1 numeric column", "Histogram with box plot marginal"),
            ("1 categorical column", "Bar chart of value counts"),
            ("1 date column", "Time series line chart"),
            ("2 numeric columns", "Scatter plot with trendline"),
            ("1 categorical + 1 numeric", "Bar chart (avg numeric by category)"),
            ("1 date + 1 numeric", "Line chart over time"),
            ("2 categorical columns", "Heatmap (cross-tabulation)"),
            ("3+ numeric columns", "Correlation heatmap"),
            ("Numeric + categorical", "Scatter matrix colored by category"),
            ("Date + multiple numeric", "Multi-line chart"),
        ]
        for inp, out in rules:
            st.markdown(f'<div style="display:flex;gap:0.5rem;font-size:0.8rem;padding:0.15rem 0;"><span style="color:#0369a1;font-weight:600;">{inp}</span><span style="color:#6b7280;">→ {out}</span></div>', unsafe_allow_html=True)

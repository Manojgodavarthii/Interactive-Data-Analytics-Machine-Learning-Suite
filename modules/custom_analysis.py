import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from modules.utils import detect_column_types, render_chart, render_charts_grid
from modules.ai_engine import row_insight, row_comparison, analyze_crosstab


def _style(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#374151", size=13),
        margin=dict(l=30, r=30, t=50, b=30),
    )


def render():
    st.markdown('<div class="section-title">🛠️ Custom Analysis Builder</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df
    col_types = detect_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t in ["categorical", "boolean"]]

    tab_agg, tab_chart, tab_rows, tab_ops = st.tabs([
        "📊 Aggregation", "📈 Custom Chart", "🔍 Row Analysis", "⚡ Operations"
    ])

    # ─── TAB 1: Aggregation ────────────────────────────────────────────────
    with tab_agg:
        agg_funcs = ["Sum", "Average", "Count", "Maximum", "Minimum", "Median", "Std Dev", "Variance"]
        c1, c2, c3 = st.columns(3)
        with c1:
            group_col = st.selectbox("Group By", [""] + cat_cols)
        with c2:
            agg_col = st.selectbox("Value column", [""] + num_cols)
        with c3:
            agg_func = st.selectbox("Function", agg_funcs)
        if group_col and agg_col:
            func_map = {
                "Sum": "sum", "Average": "mean", "Count": "count", "Maximum": "max",
                "Minimum": "min", "Median": "median", "Std Dev": "std", "Variance": "var"
            }
            result = df.groupby(group_col)[agg_col].agg(func_map[agg_func]).reset_index()
            result.columns = [group_col, f"{agg_func} of {agg_col}"]
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(result, use_container_width=True)
            with col2:
                fig = px.bar(result, x=group_col, y=f"{agg_func} of {agg_col}",
                            title=f"{agg_func} of {agg_col} by {group_col}")
                _style(fig)
                render_chart(fig, f"agg_{group_col}")
            st.download_button("📥 Download Result", result.to_csv(index=False).encode("utf-8"), "custom_analysis.csv", "text/csv")

    # ─── TAB 2: Custom Chart ──────────────────────────────────────────────
    with tab_chart:
        st.markdown("Choose chart type, columns, and generate your own graph.")
        chart_type = st.selectbox("Chart Type", [
            "Bar Chart", "Line Chart", "Pie Chart", "Histogram", "Scatter Plot",
            "Box Plot", "Area Chart", "Bubble Chart", "Violin Plot",
        ], key="cust_analysis_chart")
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X-axis", [""] + df.columns.tolist(), key="cust_ana_x")
        with c2:
            y_col = st.selectbox("Y-axis", [""] + num_cols, key="cust_ana_y")
        with c3:
            color_col = st.selectbox("Color by", [None] + df.columns.tolist(), key="cust_ana_c")
        title = st.text_input("Title", value=f"{chart_type} of {x_col or 'data'}", key="cust_ana_title")
        if st.button("📊 Generate Chart", type="primary", key="cust_ana_btn"):
            fig = None
            try:
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col if y_col else None, color=color_col, title=title)
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_col, y=y_col if y_col else (num_cols[0] if num_cols else None), color=color_col, title=title)
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=x_col, values=y_col if y_col else None, title=title)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_col, color=color_col, title=title)
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_col if y_col else (num_cols[0] if num_cols else None), color=color_col, title=title, trendline="ols")
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_col if x_col in cat_cols else None,
                                 y=x_col if x_col in num_cols else (y_col if y_col else (num_cols[0] if num_cols else None)),
                                 color=color_col, title=title)
                elif chart_type == "Area Chart":
                    fig = px.area(df, x=x_col, y=y_col if y_col else (num_cols[0] if num_cols else None), color=color_col, title=title)
                elif chart_type == "Bubble Chart":
                    fig = px.scatter(df, x=x_col, y=y_col if y_col else (num_cols[0] if num_cols else None),
                                     size=num_cols[1] if len(num_cols) > 1 else None, color=color_col, title=title)
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, x=x_col if x_col in cat_cols else None,
                                    y=x_col if x_col in num_cols else (y_col if y_col else (num_cols[0] if num_cols else None)),
                                    color=color_col, box=True, title=title)
            except Exception as e:
                st.error(f"Error: {e}")
            if fig:
                _style(fig)
                render_chart(fig, f"cust_{chart_type}")

    # ─── TAB 3: Row Analysis ──────────────────────────────────────────────
    with tab_rows:
        c1, c2 = st.columns([1, 2])
        with c1:
            n_rows = st.number_input("Number of rows", 1, min(10, max(1, len(df))), min(2, max(1, len(df))), key="ra_n")
        with c2:
            filter_col = st.selectbox("Select column to identify rows", [""] + df.columns.tolist(), key="ra_col")

        row_values = []
        if filter_col:
            unique_vals = sorted(df[filter_col].dropna().unique().tolist())
            st.markdown(f'<div style="font-size:0.78rem;color:#6b7280;margin-bottom:0.3rem;">Select {n_rows} value{"s" if n_rows > 1 else ""} from <strong>{filter_col}</strong>:</div>', unsafe_allow_html=True)
            val_cols = st.columns(min(n_rows, 5))
            for i in range(n_rows):
                with val_cols[i % len(val_cols)]:
                    val = st.selectbox(f"Row {i+1}", [""] + unique_vals, key=f"ra_val_{i}")
                    row_values.append(val)

        if filter_col and st.button("🔍 Analyze Rows", type="primary", key="ra_btn"):
            selected = [v for v in row_values if v]
            if not selected:
                st.warning("Please select at least one row value.")
            else:
                matched = df[df[filter_col].isin(selected)].copy()
                if matched.empty:
                    st.warning("No rows match the selected values.")
                else:
                    st.markdown("---")
                    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:0.5rem;">📋 Selected Row Data</div>', unsafe_allow_html=True)
                    st.dataframe(matched, use_container_width=True)

                    st.markdown("---")
                    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:0.5rem;">🤖 AI Row Insights</div>', unsafe_allow_html=True)
                    for idx in matched.index:
                        ins = row_insight(df, idx, col_types)
                        st.markdown(f'<div style="background:#f8faff;border-left:3px solid #6366f1;border-radius:0 10px 10px 0;padding:0.7rem 1rem;font-size:0.85rem;color:#111827;line-height:1.6;white-space:pre-wrap;margin-bottom:0.5rem;">{ins}</div>', unsafe_allow_html=True)

                    if len(matched) >= 2:
                        st.markdown("---")
                        st.markdown(f'<div style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:0.5rem;">⚖️ Row Comparison</div>', unsafe_allow_html=True)
                        comp = row_comparison(df, matched.index.tolist(), col_types)
                        st.markdown(f'<div style="background:#f8faff;border-left:3px solid #7c3aed;border-radius:0 10px 10px 0;padding:0.7rem 1rem;font-size:0.85rem;color:#111827;line-height:1.6;white-space:pre-wrap;">{comp}</div>', unsafe_allow_html=True)

                    if num_cols or cat_cols:
                        st.markdown("---")
                        st.markdown(f'<div style="font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:0.5rem;">📊 Multi-Chart Analysis</div>', unsafe_allow_html=True)

                        labels = matched[filter_col].astype(str) if filter_col in matched.columns else matched.index.astype(str)
                        num_chart_cols = [c for c in num_cols if c != filter_col]
                        cat_chart_cols = [c for c in cat_cols if c != filter_col]
                        date_chart_cols = [c for c, t in col_types.items() if t == "date" and c != filter_col]
                        chart_types_numeric = ["bar", "area", "bar", "area"]
                        chart_types_cat = ["pie", "bar", "pie", "bar"]
                        all_charts = []

                        if len(matched) >= 2:
                            for i, nc in enumerate(num_chart_cols):
                                ct = chart_types_numeric[i % len(chart_types_numeric)]
                                if ct == "bar":
                                    fig = px.bar(matched, x=labels, y=nc, title=f"📊 {nc}", color=labels,
                                                 color_discrete_sequence=px.colors.qualitative.Bold, text_auto=".2s")
                                elif ct == "area":
                                    fig = px.area(matched, x=labels, y=nc, title=f"📈 {nc}", color=labels,
                                                  color_discrete_sequence=px.colors.qualitative.Bold)
                                fig.update_layout(showlegend=False, height=260)
                                _style(fig)
                                all_charts.append(fig)

                            for i, cc in enumerate(cat_chart_cols):
                                ct = chart_types_cat[i % len(chart_types_cat)]
                                val_counts = matched[cc].value_counts().reset_index()
                                val_counts.columns = [cc, "Count"]
                                if ct == "pie":
                                    fig = px.pie(val_counts, names=cc, values="Count", title=f"🥧 {cc}",
                                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                                    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                                elif ct == "bar":
                                    fig = px.bar(val_counts, x="Count", y=cc, title=f"📊 {cc}", orientation="h",
                                                 color=cc, color_discrete_sequence=px.colors.qualitative.Pastel,
                                                 text_auto=True)
                                    fig.update_layout(showlegend=False, height=260)
                                _style(fig)
                                all_charts.append(fig)

                            for dc in date_chart_cols:
                                if matched[dc].notna().any():
                                    sorted_d = matched.sort_values(dc)
                                    fig = px.line(sorted_d, x=dc,
                                                  y=num_chart_cols[0] if num_chart_cols else matched.select_dtypes(include="number").columns[0] if not matched.select_dtypes(include="number").empty else None,
                                                  title=f"📅 {dc} Timeline", markers=True)
                                    fig.update_layout(height=260)
                                    _style(fig)
                                    all_charts.append(fig)

                            if len(num_chart_cols) >= 2 and len(matched) >= 2:
                                radar_df = matched.reset_index(drop=True)
                                radar_df["label"] = labels.values if hasattr(labels, "values") else labels
                                fig = px.line_polar(radar_df, r=num_chart_cols[0], theta="label",
                                                     line_close=True, title=f"🕸️ {num_chart_cols[0]} Polar",
                                                     range_r=[0, radar_df[num_chart_cols[0]].max() * 1.1])
                                fig.update_layout(height=280)
                                _style(fig)
                                all_charts.append(fig)
                        else:
                            for nc in num_chart_cols:
                                fig = px.pie(values=[matched[nc].values[0], 0], names=[nc, ""],
                                             title=f"📊 {nc}: {matched[nc].values[0]:.2f}",
                                             color_discrete_sequence=["#6366f1", "#e4e8f0"])
                                fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                                _style(fig)
                                all_charts.append(fig)

                        if all_charts:
                            items = [(all_charts[idx], f"ra_{idx}") for idx in range(len(all_charts))]
                            render_charts_grid(items, section_key="ra")
                        else:
                            st.info("No plottable columns found for charting.")

    # ─── TAB 4: Operations ────────────────────────────────────────────────
    with tab_ops:
        op = st.selectbox("Operation", [
            "Top N Records", "Bottom N Records", "Ranking",
            "Cross-tabulation (with AI)", "Running Total",
            "Pivot Table", "Percentage Contribution"
        ])
        max_rows = max(1, len(df))
        init_n = min(10, max_rows)
        if op == "Top N Records":
            n = st.number_input("N", min_value=1, max_value=max_rows, value=init_n)
            sort_col = st.selectbox("Sort by", num_cols if num_cols else df.columns)
            asc = st.checkbox("Ascending", value=False)
            top = df.sort_values(by=sort_col, ascending=asc).head(n)
            st.dataframe(top, use_container_width=True)
            fig = px.bar(top, x=top.index.astype(str), y=sort_col, title=f"Top {n} by {sort_col}")
            _style(fig)
            render_chart(fig, "top_n")
        elif op == "Bottom N Records":
            n = st.number_input("N", min_value=1, max_value=max_rows, value=init_n)
            sort_col = st.selectbox("Sort by", num_cols if num_cols else df.columns)
            asc = st.checkbox("Ascending", value=True)
            bottom = df.sort_values(by=sort_col, ascending=asc).head(n)
            st.dataframe(bottom, use_container_width=True)
            fig = px.bar(bottom, x=bottom.index.astype(str), y=sort_col, title=f"Bottom {n} by {sort_col}")
            _style(fig)
            render_chart(fig, "bottom_n")
        elif op == "Ranking":
            rank_col = st.selectbox("Rank by", num_cols if num_cols else df.columns)
            df_rank = df.copy()
            df_rank["Rank"] = df_rank[rank_col].rank(ascending=False).astype(int)
            df_rank = df_rank.sort_values("Rank")
            st.dataframe(df_rank, use_container_width=True)
            fig = px.bar(df_rank.head(20), x="Rank", y=rank_col, title=f"Ranking by {rank_col}")
            _style(fig)
            render_chart(fig, "ranking")
        elif op == "Cross-tabulation (with AI)":
            if len(cat_cols) >= 2:
                rc = st.selectbox("Row", cat_cols, key="ct_row2")
                cc = st.selectbox("Column", cat_cols, key="ct_col2")
                ct = pd.crosstab(df[rc], df[cc])
                st.dataframe(ct, use_container_width=True)
                fig = px.imshow(ct, text_auto=True, aspect="auto", title=f"Cross-tabulation: {rc} vs {cc}",
                               color_continuous_scale="Blues")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
                render_chart(fig, "cross_tab")
                ai_result = analyze_crosstab(rc, cc, df)
                st.markdown(f'<div style="background:#f0f4ff;border-left:3px solid #6366f1;border-radius:0 10px 10px 0;padding:0.7rem 1rem;font-size:0.85rem;color:#111827;"><span style="font-weight:700;color:#6366f1;">🤖 AI:</span> {ai_result["insight"]}</div>', unsafe_allow_html=True)
            else:
                st.warning("Need at least 2 categorical columns.")
        elif op == "Running Total":
            rt_col = st.selectbox("Select numeric column", num_cols if num_cols else df.columns)
            df_rt = df.copy()
            df_rt["Running Total"] = df_rt[rt_col].cumsum()
            st.dataframe(df_rt, use_container_width=True)
            fig = px.line(df_rt, y="Running Total", title=f"Running Total of {rt_col}")
            _style(fig)
            render_chart(fig, "running_total")
        elif op == "Pivot Table":
            if len(cat_cols) >= 1 and num_cols:
                idx = st.selectbox("Index", cat_cols)
                vals = st.selectbox("Values", num_cols)
                agg = st.selectbox("Aggregation", ["mean", "sum", "count", "median"])
                pt = pd.pivot_table(df, values=vals, index=idx, aggfunc=agg)
                st.dataframe(pt, use_container_width=True)
                fig = px.bar(pt.reset_index(), x=idx, y=vals, title=f"Pivot: {agg} of {vals} by {idx}")
                _style(fig)
                render_chart(fig, "pivot")
        elif op == "Percentage Contribution":
            if num_cols:
                pcol = st.selectbox("Select column", num_cols)
                total = df[pcol].sum()
                contrib = (df[pcol] / total * 100).reset_index()
                contrib.columns = ["Index", "Contribution (%)"]
                contrib = contrib.sort_values("Contribution (%)", ascending=False)
                st.dataframe(contrib.head(20), use_container_width=True)
                fig = px.pie(contrib.head(10), values="Contribution (%)",
                            names=contrib.head(10).index.astype(str),
                            title=f"Top 10 Contributors to {pcol}")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
                render_chart(fig, "pct_contrib")

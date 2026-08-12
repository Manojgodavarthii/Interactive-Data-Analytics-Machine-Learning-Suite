import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.utils import detect_column_types, render_chart, view_all_button

# ── Colour palettes ──────────────────────────────────────────────────────────
PALETTE = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6",
           "#8b5cf6", "#f97316", "#14b8a6", "#ec4899", "#84cc16"]
MONO = "#0f3460"

CHART_COLORS = {
    "scatter":   ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6"],
    "bar":       px.colors.qualitative.Bold,
    "heatmap":   "RdBu_r",
    "histogram": ["#6366f1"],
    "pie":       px.colors.qualitative.Set3,
    "box":       px.colors.qualitative.Bold,
    "violin":    px.colors.qualitative.Bold,
    "line":      ["#0f3460"],
    "area":      ["#6366f1"],
    "treemap":   px.colors.qualitative.Pastel,
}


# ── Helper: style any plotly figure ──────────────────────────────────────────
def _style(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=55, b=40),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
    return fig


# ── Helper: AI insight text for any chart ────────────────────────────────────
def _ai_insight(chart_type, x, y, df_sample):
    """Generate a dataset-aware AI insight sentence for a given chart."""
    try:
        if chart_type == "Scatter Plot" and x and y:
            vals_x = df_sample[x].dropna()
            vals_y = df_sample[y].dropna()
            common = vals_x.index.intersection(vals_y.index)
            if len(common) > 2:
                r = np.corrcoef(vals_x[common].values, vals_y[common].values)[0, 1]
                direction = "positive" if r > 0 else "negative"
                strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                return (f"Correlation r = {r:.2f} — {strength} {direction} relationship between "
                        f"<strong>{x}</strong> and <strong>{y}</strong>. "
                        f"{'Consider this pair for predictive modelling.' if abs(r) > 0.6 else 'Explore further for non-linear patterns.'}")
        elif chart_type == "Histogram" and x:
            s = df_sample[x].dropna()
            skew = s.skew() if len(s) > 1 else 0
            direction = "right-skewed (long right tail)" if skew > 0.5 else ("left-skewed (long left tail)" if skew < -0.5 else "symmetric")
            outliers = int((abs((s - s.mean()) / s.std()) > 3).sum()) if s.std() > 0 else 0
            return (f"<strong>{x}</strong> is {direction} (skewness = {skew:.2f}). "
                    f"Mean = {s.mean():.2f}, Median = {s.median():.2f}. "
                    f"{f'{outliers} outlier(s) detected.' if outliers > 0 else 'No significant outliers.'}")
        elif chart_type == "Bar Chart" and x:
            top = df_sample[x].value_counts()
            if len(top) > 0:
                top_val = top.index[0]
                top_pct = top.iloc[0] / len(df_sample[x].dropna()) * 100
                return (f"<strong>{top_val}</strong> is the most frequent value in <strong>{x}</strong> "
                        f"({top_pct:.1f}% of records). "
                        f"There are {df_sample[x].nunique()} unique categories total.")
        elif chart_type == "Pie Chart" and x:
            top = df_sample[x].value_counts()
            if len(top) > 0:
                top_val = top.index[0]
                top_pct = top.iloc[0] / len(df_sample[x].dropna()) * 100
                return (f"<strong>{top_val}</strong> holds {top_pct:.1f}% of the share. "
                        f"{'Highly concentrated distribution.' if top_pct > 50 else 'Relatively balanced distribution across categories.'}")
        elif chart_type == "Box Plot" and y and x:
            groups = df_sample.groupby(x)[y]
            means = groups.mean()
            if len(means) > 1:
                best = means.idxmax()
                worst = means.idxmin()
                return (f"<strong>{best}</strong> has the highest average <strong>{y}</strong> ({means[best]:.2f}), "
                        f"while <strong>{worst}</strong> has the lowest ({means[worst]:.2f}). "
                        f"Points outside the whiskers are potential outliers.")
        elif chart_type == "Violin Plot" and y and x:
            return (f"Violin plot reveals the full distribution shape of <strong>{y}</strong> "
                    f"within each <strong>{x}</strong> group. Wider sections indicate where data concentrates.")
        elif chart_type == "Line Chart" and y:
            s = df_sample[y].dropna()
            if len(s) > 2:
                slope = np.polyfit(range(len(s)), s.values, 1)[0]
                trend = "increasing" if slope > 0 else "decreasing"
                return (f"<strong>{y}</strong> shows an overall {trend} trend "
                        f"(slope = {slope:.4f} per step). "
                        f"Range: {s.min():.2f} → {s.max():.2f}.")
        elif chart_type == "Correlation Heatmap":
            num_cols = [c for c in df_sample.columns if pd.api.types.is_numeric_dtype(df_sample[c])]
            if len(num_cols) >= 2:
                corr = df_sample[num_cols].corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                pairs = upper.unstack().dropna().sort_values(ascending=False)
                if len(pairs) > 0:
                    (ca, cb), best_r = pairs.index[0], pairs.iloc[0]
                    return (f"Strongest correlation: <strong>{ca}</strong> ↔ <strong>{cb}</strong> "
                            f"(r = {best_r:.2f}). "
                            f"Dark red = strong positive, dark blue = strong negative correlation.")
        elif chart_type == "Treemap" and x:
            return (f"Treemap shows proportional area for each <strong>{x}</strong> category. "
                    f"Larger tiles represent categories with higher values — good for part-of-whole analysis.")
        elif chart_type == "Area Chart" and y:
            s = df_sample[y].dropna()
            return (f"Area chart for <strong>{y}</strong>: total area = {s.sum():.2f}, "
                    f"filled region emphasises cumulative magnitude and trends over time.")
        elif chart_type == "Grouped Bar" and x and y:
            agg = df_sample.groupby(x)[y].mean()
            if len(agg) > 0:
                top_val = agg.idxmax()
                return (f"<strong>{top_val}</strong> has the highest average <strong>{y}</strong> "
                        f"({agg[top_val]:.2f}) among all {x} categories.")
    except Exception:
        pass
    return f"This <strong>{chart_type}</strong> visualises <strong>{x or y or 'the data'}</strong>. Look for patterns, clusters, and outliers to extract actionable insights."


# ── Core chart builder (robust, no silent failures) ───────────────────────────
def _build_chart(chart_type, x, y, color, df, title=None):
    """
    Build a Plotly figure. Returns (fig, error_msg) tuple.
    error_msg is None on success, a string on failure.
    """
    if title is None:
        title = f"{chart_type} — {x or ''}" + (f" vs {y}" if y else "")

    # Sample large datasets for speed
    df_plot = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df

    try:
        if chart_type == "Scatter Plot":
            if not x or not y:
                return None, "Scatter Plot requires both X and Y columns."
            fig = px.scatter(df_plot, x=x, y=y, color=color, title=title,
                             trendline="ols", opacity=0.72,
                             color_discrete_sequence=CHART_COLORS["scatter"])

        elif chart_type == "Histogram":
            if not x:
                return None, "Histogram requires an X column."
            n_bins = min(50, max(10, int(df_plot[x].nunique() / 5)))
            fig = px.histogram(df_plot, x=x, nbins=n_bins, marginal="box",
                               title=title, color_discrete_sequence=CHART_COLORS["histogram"])
            fig.update_xaxes(tickangle=45, nticks=20)

        elif chart_type == "Bar Chart":
            if not x:
                return None, "Bar Chart requires an X column."
            top = df_plot[x].value_counts().head(20).reset_index()
            top.columns = [x, "Count"]
            fig = px.bar(top, x=x, y="Count", title=title, color=x,
                         color_discrete_sequence=CHART_COLORS["bar"])
            fig.update_xaxes(tickangle=45, tickfont=dict(size=10))

        elif chart_type == "Pie Chart":
            if not x:
                return None, "Pie Chart requires an X column."
            top = df_plot[x].value_counts().head(10).reset_index()
            top.columns = [x, "Count"]
            fig = px.pie(top, names=x, values="Count", title=title,
                         color_discrete_sequence=CHART_COLORS["pie"],
                         hole=0.3)

        elif chart_type == "Line Chart":
            if y:
                df_ln = df_plot[[x, y]].dropna().sort_values(x) if x else df_plot
                if x:
                    fig = px.line(df_ln, x=x, y=y, title=title,
                                  color_discrete_sequence=CHART_COLORS["line"])
                else:
                    fig = px.line(df_ln, y=y, title=title,
                                  color_discrete_sequence=CHART_COLORS["line"])
            else:
                fig = px.line(df_plot.reset_index(), x="index", y=x, title=title,
                              color_discrete_sequence=CHART_COLORS["line"])
            fig.update_traces(line=dict(width=2))

        elif chart_type == "Box Plot":
            if not y:
                return None, "Box Plot requires a Y (numeric) column."
            fig = px.box(df_plot, x=x, y=y, color=color if color else x,
                         title=title, color_discrete_sequence=CHART_COLORS["box"])
            if x:
                fig.update_xaxes(tickangle=45)

        elif chart_type == "Violin Plot":
            if not y:
                return None, "Violin Plot requires a Y (numeric) column."
            fig = px.violin(df_plot, x=x, y=y, color=color if color else x,
                            box=True, points="outliers", title=title,
                            color_discrete_sequence=CHART_COLORS["violin"])
            if x:
                fig.update_xaxes(tickangle=45)

        elif chart_type == "Grouped Bar":
            if not x or not y:
                return None, "Grouped Bar requires X (category) and Y (numeric)."
            agg = df_plot.groupby(x)[y].mean().reset_index().sort_values(y, ascending=False).head(20)
            fig = px.bar(agg, x=x, y=y, title=title,
                         color=x, color_discrete_sequence=CHART_COLORS["bar"])
            fig.update_xaxes(tickangle=45)

        elif chart_type == "Treemap":
            if not x:
                return None, "Treemap requires an X (category) column."
            if y:
                agg_df = df_plot.groupby(x)[y].sum().reset_index()
                agg_df = agg_df[agg_df[y] > 0]
                fig = px.treemap(agg_df, path=[x], values=y, title=title,
                                 color_discrete_sequence=CHART_COLORS["treemap"])
            else:
                vc = df_plot[x].value_counts().reset_index()
                vc.columns = [x, "Count"]
                fig = px.treemap(vc, path=[x], values="Count", title=title,
                                 color_discrete_sequence=CHART_COLORS["treemap"])

        elif chart_type == "Area Chart":
            col_to_use = y or x
            if not col_to_use:
                return None, "Area Chart requires a numeric column."
            fig = px.area(df_plot.reset_index(), x="index", y=col_to_use,
                          title=title, color_discrete_sequence=CHART_COLORS["area"])

        elif chart_type == "Correlation Heatmap":
            num_cols = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])]
            if len(num_cols) < 2:
                return None, "Need at least 2 numeric columns for heatmap."
            corr = df_plot[num_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title=title,
                            color_continuous_scale=CHART_COLORS["heatmap"],
                            range_color=[-1, 1])
            fig.update_layout(height=max(400, len(num_cols) * 40))

        elif chart_type == "Bubble Chart":
            if not x or not y:
                return None, "Bubble Chart requires X and Y columns."
            num_cols = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])]
            size_col = [c for c in num_cols if c not in [x, y]]
            fig = px.scatter(df_plot, x=x, y=y,
                             size=size_col[0] if size_col else None,
                             color=color,
                             title=title,
                             color_discrete_sequence=CHART_COLORS["scatter"],
                             size_max=30, opacity=0.7)

        elif chart_type == "Density Heatmap":
            if not x or not y:
                return None, "Density Heatmap requires X and Y numeric columns."
            fig = px.density_heatmap(df_plot, x=x, y=y, title=title,
                                     color_continuous_scale="Viridis")

        elif chart_type == "Funnel":
            if not x or not y:
                return None, "Funnel requires category (X) and numeric (Y)."
            agg = df_plot.groupby(x)[y].sum().reset_index().sort_values(y, ascending=False).head(10)
            fig = px.funnel(agg, x=y, y=x, title=title,
                            color_discrete_sequence=PALETTE)

        else:
            return None, f"Unknown chart type: {chart_type}"

        _style(fig)
        return fig, None

    except Exception as e:
        return None, str(e)


# ── Generate ALL charts for the dataset ──────────────────────────────────────
def _generate_all_charts(df, col_types):
    """
    Exhaustively generate charts for all important column combinations.
    Returns list of dicts: {chart_type, x, y, color, title, reason}
    """
    charts = []
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]

    df_s = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df

    # ── 1. Histograms for every numeric column ──
    for col in num_cols[:8]:
        s = df_s[col].dropna()
        if len(s) < 3:
            continue
        skew = s.skew()
        charts.append({
            "chart_type": "Histogram", "x": col, "y": None, "color": None,
            "title": f"📊 Distribution — {col}",
            "reason": f"Shows frequency distribution of <strong>{col}</strong>. "
                      f"Skewness = {skew:.2f} ({'right-skewed' if skew > 0.5 else 'left-skewed' if skew < -0.5 else 'symmetric'}).",
            "priority": 9,
        })

    # ── 2. Bar charts for every categorical column ──
    for col in cat_cols[:6]:
        n_uniq = df_s[col].nunique()
        if n_uniq < 1 or n_uniq > 50:
            continue
        top_val = df_s[col].value_counts().index[0]
        top_pct = df_s[col].value_counts().iloc[0] / len(df_s[col].dropna()) * 100
        charts.append({
            "chart_type": "Bar Chart", "x": col, "y": None, "color": None,
            "title": f"📋 Category Counts — {col}",
            "reason": f"<strong>{col}</strong> has {n_uniq} categories. Most common: <strong>{top_val}</strong> ({top_pct:.1f}%).",
            "priority": 8,
        })

    # ── 3. Pie charts for low-cardinality categoricals ──
    for col in cat_cols[:4]:
        n_uniq = df_s[col].nunique()
        if 2 <= n_uniq <= 10:
            charts.append({
                "chart_type": "Pie Chart", "x": col, "y": None, "color": None,
                "title": f"🥧 Proportions — {col}",
                "reason": f"Pie chart shows each category's share in <strong>{col}</strong> ({n_uniq} categories).",
                "priority": 7,
            })

    # ── 4. Top correlated scatter plots ──
    if len(num_cols) >= 2:
        try:
            corr = df_s[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.unstack().dropna().sort_values(ascending=False)
            for (ca, cb), r_val in pairs.head(5).items():
                cat_color = cat_cols[0] if cat_cols else None
                charts.append({
                    "chart_type": "Scatter Plot", "x": ca, "y": cb, "color": cat_color,
                    "title": f"📈 Scatter — {ca} vs {cb}",
                    "reason": f"Correlation r = {r_val:.2f} between <strong>{ca}</strong> and <strong>{cb}</strong>. "
                              f"{'Strong relationship — great for predictive modelling.' if r_val > 0.7 else 'Moderate relationship — explore patterns.'}",
                    "priority": 10 if r_val > 0.7 else 8,
                })
        except Exception:
            pass

    # ── 5. Box plots — numeric vs best categorical ──
    if cat_cols and num_cols:
        # Pick categorical with reasonable cardinality
        best_cats = [c for c in cat_cols if 2 <= df_s[c].nunique() <= 15][:2]
        for cat_col in best_cats:
            for num_col in num_cols[:3]:
                charts.append({
                    "chart_type": "Box Plot", "x": cat_col, "y": num_col, "color": None,
                    "title": f"📦 Box Plot — {num_col} by {cat_col}",
                    "reason": f"Shows how <strong>{num_col}</strong> varies across <strong>{cat_col}</strong> groups. Reveals median, spread, and outliers per category.",
                    "priority": 7,
                })

    # ── 6. Violin plots — numeric vs categorical ──
    if cat_cols and num_cols:
        best_cats = [c for c in cat_cols if 2 <= df_s[c].nunique() <= 10][:1]
        for cat_col in best_cats:
            for num_col in num_cols[:2]:
                charts.append({
                    "chart_type": "Violin Plot", "x": cat_col, "y": num_col, "color": None,
                    "title": f"🎻 Violin — {num_col} by {cat_col}",
                    "reason": f"Violin plot reveals full distribution shape of <strong>{num_col}</strong> within each <strong>{cat_col}</strong> group.",
                    "priority": 6,
                })

    # ── 7. Grouped bar (aggregated cat × num) ──
    if cat_cols and num_cols:
        best_cats = [c for c in cat_cols if 2 <= df_s[c].nunique() <= 20][:2]
        for cat_col in best_cats:
            for num_col in num_cols[:2]:
                charts.append({
                    "chart_type": "Grouped Bar", "x": cat_col, "y": num_col, "color": None,
                    "title": f"📊 Avg {num_col} by {cat_col}",
                    "reason": f"Average <strong>{num_col}</strong> per <strong>{cat_col}</strong> category — easy comparison bar chart.",
                    "priority": 7,
                })

    # ── 8. Line charts ──
    if date_cols and num_cols:
        date_col = date_cols[0]
        for num_col in num_cols[:3]:
            charts.append({
                "chart_type": "Line Chart", "x": date_col, "y": num_col, "color": None,
                "title": f"📈 Trend — {num_col} over Time",
                "reason": f"Date column detected — tracks <strong>{num_col}</strong> over time. Look for trends, seasonality, or sudden shifts.",
                "priority": 9,
            })
    elif num_cols:
        for num_col in num_cols[:2]:
            charts.append({
                "chart_type": "Line Chart", "x": None, "y": num_col, "color": None,
                "title": f"📈 Trend — {num_col}",
                "reason": f"Line chart shows how <strong>{num_col}</strong> changes across rows. Useful for detecting trends.",
                "priority": 5,
            })

    # ── 9. Treemap ──
    if cat_cols and num_cols:
        charts.append({
            "chart_type": "Treemap", "x": cat_cols[0], "y": num_cols[0], "color": None,
            "title": f"🗺️ Treemap — {cat_cols[0]} by {num_cols[0]}",
            "reason": f"Treemap reveals which <strong>{cat_cols[0]}</strong> categories contribute most to total <strong>{num_cols[0]}</strong>.",
            "priority": 6,
        })
    elif cat_cols:
        charts.append({
            "chart_type": "Treemap", "x": cat_cols[0], "y": None, "color": None,
            "title": f"🗺️ Treemap — {cat_cols[0]}",
            "reason": f"Treemap shows proportional area for each <strong>{cat_cols[0]}</strong> category.",
            "priority": 5,
        })

    # ── 10. Area chart ──
    if num_cols:
        charts.append({
            "chart_type": "Area Chart", "x": None, "y": num_cols[0], "color": None,
            "title": f"📉 Area Chart — {num_cols[0]}",
            "reason": f"Area chart emphasises the cumulative magnitude of <strong>{num_cols[0]}</strong> across records.",
            "priority": 5,
        })

    # ── 11. Bubble chart (3 numeric dims) ──
    if len(num_cols) >= 3:
        charts.append({
            "chart_type": "Bubble Chart", "x": num_cols[0], "y": num_cols[1],
            "color": cat_cols[0] if cat_cols else None,
            "title": f"🫧 Bubble — {num_cols[0]} vs {num_cols[1]}",
            "reason": f"Bubble chart adds a 3rd numeric dimension (size) to the scatter of <strong>{num_cols[0]}</strong> vs <strong>{num_cols[1]}</strong>.",
            "priority": 6,
        })

    # ── 12. Density Heatmap ──
    if len(num_cols) >= 2:
        charts.append({
            "chart_type": "Density Heatmap", "x": num_cols[0], "y": num_cols[1],
            "color": None,
            "title": f"🌡️ Density Heatmap — {num_cols[0]} vs {num_cols[1]}",
            "reason": f"Density heatmap shows where data concentrates in the <strong>{num_cols[0]}</strong> × <strong>{num_cols[1]}</strong> space.",
            "priority": 6,
        })

    # ── 13. Correlation Heatmap (if enough numeric) ──
    if len(num_cols) >= 3:
        charts.append({
            "chart_type": "Correlation Heatmap", "x": None, "y": None, "color": None,
            "title": f"🔗 Correlation Matrix ({len(num_cols)} numeric cols)",
            "reason": f"Heatmap of all {len(num_cols)} numeric columns. Instantly reveals which pairs are strongly correlated.",
            "priority": 10,
        })

    # Sort by priority
    charts.sort(key=lambda c: c["priority"], reverse=True)
    return charts


# ── Render AI tab ─────────────────────────────────────────────────────────────
def _render_ai_tab(df, col_types):
    st.markdown(
        """<div style="background:linear-gradient(135deg,#eef2ff,#e0e7ff);border-radius:14px;
        padding:1rem 1.2rem;margin-bottom:1.2rem;font-size:0.88rem;color:#3730a3;font-weight:500;">
        🤖 <strong>AI Smart Charts</strong> — Automatically generated based on your dataset's structure,
        correlations, and column types. Each chart includes an AI insight explaining what to look for.
        </div>""",
        unsafe_allow_html=True,
    )

    with st.spinner("⏳ Loading charts..."):
        charts = _generate_all_charts(df, col_types)

    if not charts:
        st.warning("Not enough columns to generate charts. Upload a dataset with numeric or categorical columns.")
        return

    df_sample = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df

    st.markdown(
        f'<div style="font-size:0.8rem;color:#6b7280;margin-bottom:1rem;">'
        f'Showing <strong>{len(charts)} charts</strong> tailored to your dataset</div>',
        unsafe_allow_html=True,
    )

    # Render charts in 2-column grid
    for i in range(0, len(charts), 2):
        row_cols = st.columns(2)
        for j, ui_col in enumerate(row_cols):
            idx = i + j
            if idx >= len(charts):
                break
            rec = charts[idx]
            with ui_col:
                fig, err = _build_chart(
                    rec["chart_type"], rec["x"], rec["y"], rec.get("color"),
                    df, title=rec["title"]
                )
                if fig is not None:
                    render_chart(fig, f"ai_{idx}")
                    insight = _ai_insight(rec["chart_type"], rec["x"], rec["y"], df_sample)
                    st.markdown(
                        f'<div style="background:#f8faff;border-left:3px solid #6366f1;'
                        f'border-radius:0 10px 10px 0;padding:0.55rem 0.85rem;'
                        f'font-size:0.82rem;color:#111827;line-height:1.5;">'
                        f'<span style="font-weight:700;color:#6366f1;">🤖 AI:</span> {insight}</div>',
                        unsafe_allow_html=True,
                    )
                elif err:
                    st.markdown(
                        f'<div style="background:#fef2f2;border:1px dashed #fca5a5;border-radius:10px;'
                        f'padding:0.8rem;font-size:0.8rem;color:#991b1b;">'
                        f'⚠️ Could not render <strong>{rec["chart_type"]}</strong>: {err}</div>',
                        unsafe_allow_html=True,
                    )


# ── Render Custom tab ─────────────────────────────────────────────────────────
def _render_custom_tab(df, col_types):
    st.markdown('<div class="section-title">🎛️ Custom Chart Builder</div>', unsafe_allow_html=True)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    all_cols = df.columns.tolist()

    with st.container():
        chart_type = st.selectbox("Chart Type", [
            "Bar Chart", "Histogram", "Scatter Plot", "Line Chart", "Pie Chart",
            "Box Plot", "Violin Plot", "Area Chart", "Bubble Chart", "Treemap",
            "Grouped Bar", "Density Heatmap", "Correlation Heatmap",
            "Funnel", "Parallel Coordinates", "Waterfall",
        ], key="cust_chart_type")

        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X-axis / Category", [""] + all_cols, key="cust_x")
        with c2:
            y_col = st.selectbox("Y-axis / Value", [""] + num_cols, key="cust_y")
        with c3:
            color_col = st.selectbox("Color by", [None] + all_cols, key="cust_color")

        title_custom = st.text_input(
            "Chart Title",
            value=f"{chart_type} — {x_col or 'data'}",
            key="cust_title"
        )

    # Handle special chart types that need extra inputs
    extra_cols = None
    if chart_type == "Parallel Coordinates":
        extra_cols = st.multiselect(
            "Numeric dimensions",
            num_cols,
            default=num_cols[:min(4, len(num_cols))],
            key="cust_pc"
        )
    elif chart_type == "Waterfall":
        pass  # handled below

    if st.button("📊 Generate Chart", type="primary", key="gen_chart_btn"):
        df_plot = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df
        fig = None
        err_msg = None

        try:
            if chart_type == "Parallel Coordinates":
                dims = extra_cols or num_cols[:4]
                if dims:
                    fig_pc = px.parallel_coordinates(
                        df_plot, dimensions=dims,
                        color=dims[0] if dims else None,
                        title=title_custom,
                        color_continuous_scale="Viridis"
                    )
                    _style(fig_pc)
                    fig = fig_pc
                else:
                    err_msg = "Select at least 1 numeric column."
            elif chart_type == "Waterfall":
                val_col = y_col or (num_cols[0] if num_cols else None)
                lbl_col = x_col or (all_cols[0] if all_cols else None)
                if val_col and lbl_col:
                    vals = df_plot[val_col].dropna().head(12).tolist()
                    lbls = df_plot[lbl_col].dropna().head(12).astype(str).tolist()
                    fig_wf = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=[("absolute" if i == 0 else "relative") for i in range(len(vals))],
                        x=lbls, y=vals,
                        text=[f"{v:+.2f}" for v in vals],
                        connector={"line": {"color": "#9ca3af"}},
                        increasing={"marker": {"color": "#22c55e"}},
                        decreasing={"marker": {"color": "#ef4444"}},
                    ))
                    fig_wf.update_layout(title=title_custom, waterfallgap=0.3)
                    _style(fig_wf)
                    fig = fig_wf
                else:
                    err_msg = "Select X and Y columns for Waterfall."
            elif chart_type == "Correlation Heatmap":
                hm_cols = num_cols if num_cols else []
                if len(hm_cols) >= 2:
                    corr = df_plot[hm_cols].corr().round(2)
                    fig_hm = px.imshow(corr, text_auto=True, aspect="auto",
                                       title=title_custom,
                                       color_continuous_scale="RdBu_r",
                                       range_color=[-1, 1])
                    _style(fig_hm, height=max(400, len(hm_cols) * 45))
                    fig = fig_hm
                else:
                    err_msg = "Need at least 2 numeric columns for Correlation Heatmap."
            else:
                fig, err_msg = _build_chart(
                    chart_type,
                    x_col or None,
                    y_col or None,
                    color_col,
                    df,
                    title=title_custom
                )
        except Exception as e:
            err_msg = str(e)

        if fig:
            render_chart(fig, "custom_fig")
            insight = _ai_insight(chart_type, x_col or None, y_col or None, df.head(500))
            st.markdown(
                f'<div style="background:#f8faff;border-left:4px solid #0f3460;'
                f'border-radius:0 12px 12px 0;padding:0.8rem 1rem;margin-top:0.5rem;'
                f'font-size:0.88rem;color:#374151;line-height:1.6;">'
                f'<span style="font-weight:700;color:#0f3460;">🤖 AI Insight:</span> {insight}</div>',
                unsafe_allow_html=True,
            )
        elif err_msg:
            st.error(f"Could not generate chart: {err_msg}")


# ── Main render ───────────────────────────────────────────────────────────────
def render():
    st.markdown('<div class="section-title">🎨 AI Visualizations</div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df
    col_types = detect_column_types(df)

    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    # Dataset summary line
    st.markdown(
        f'<div style="font-size:0.82rem;color:#6b7280;margin-bottom:1rem;">'
        f'Dataset: <strong>{len(df):,} rows</strong> · '
        f'<strong>{len(num_cols)}</strong> numeric · '
        f'<strong>{len(cat_cols)}</strong> categorical columns'
        f'</div>',
        unsafe_allow_html=True,
    )

    tab_ai, tab_custom = st.tabs(["🤖 AI Smart Charts", "🎛️ Custom Chart Builder"])

    with tab_ai:
        _render_ai_tab(df, col_types)

    with tab_custom:
        _render_custom_tab(df, col_types)

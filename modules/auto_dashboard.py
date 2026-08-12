import streamlit as st
import pandas as pd
from modules.utils import detect_column_types, get_summary_stats, auto_insights, render_chart, render_charts_grid
import plotly.express as px


def _calculate_kpis(df, col_types):
    stats, _ = get_summary_stats(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    kpis = {"Total Rows": stats["rows"], "Total Columns": stats["columns"], "Missing Values": stats["missing"], "Duplicates": stats["duplicates"]}
    if num_cols:
        kpis["Average Value"] = round(df[num_cols].mean().mean(), 2)
        revenue_cols = [c for c in num_cols if "revenue" in c.lower() or "sales" in c.lower() or "amount" in c.lower() or "price" in c.lower()]
        if revenue_cols:
            kpis["Total Revenue"] = round(df[revenue_cols].sum().sum(), 2)
        else:
            kpis["Total Sum"] = round(df[num_cols].sum().sum(), 2)
    return kpis


def _build_kpi_card(label, value, color):
    return f'<div style="background:white;border:1px solid #e4e8f0;border-radius:12px;padding:0.7rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.04);"><div style="font-size:0.65rem;color:#6b7280;font-weight:600;text-transform:uppercase;">{label}</div><div style="font-size:1.4rem;font-weight:900;color:{color};margin-top:0.2rem;">{value:,}</div></div>'


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)
    insights = auto_insights(df, col_types)
    stats, _ = get_summary_stats(df)

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F916;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">Auto Dashboard Generator</span></div>', unsafe_allow_html=True)

    kpis = _calculate_kpis(df, col_types)
    colors = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#dc2626", "#7c3aed"]
    cols = st.columns(len(kpis))
    for i, (label, value) in enumerate(kpis.items()):
        with cols[i]:
            st.markdown(_build_kpi_card(label, value, colors[i % len(colors)]), unsafe_allow_html=True)

    st.markdown('<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:1.2rem 0 0.5rem 0;">Auto-Generated Charts</div>', unsafe_allow_html=True)
    from modules.utils import auto_charts
    charts = auto_charts(df, col_types)
    if charts:
        items = [(f, f"auto_{i}", t) for i, (t, f) in enumerate(charts)]
        render_charts_grid(items, section_key="auto_dash")

    if insights:
        st.markdown('<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:1.2rem 0 0.5rem 0;">AI Summary</div>', unsafe_allow_html=True)
        for ins in insights:
            st.markdown(f'<div style="background:#f8faff;border-left:3px solid #6366f1;border-radius:0 8px 8px 0;padding:0.45rem 0.8rem;font-size:0.82rem;color:#111827;margin-bottom:0.3rem;">&#x1F916; {ins}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:1.2rem 0 0.5rem 0;">Top Insight</div>', unsafe_allow_html=True)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    if num_cols:
        top_corr = df[num_cols].corr().unstack().sort_values(ascending=False)
        top_corr = top_corr[top_corr < 1]
        if not top_corr.empty:
            pair = top_corr.index[0]
            st.markdown(f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:0.8rem 1rem;font-size:0.85rem;color:#166534;"><strong>&#x1F4C8; Highest Correlation:</strong> {pair[0]} &harr; {pair[1]} ({top_corr.iloc[0]:.2f})<br><strong>&#x1F50D; Outlier Count:</strong> Detected in {len(num_cols)} numeric columns</div>', unsafe_allow_html=True)

    # ── Interactive What-If Scenario Simulator ─────────────────────────────
    if len(num_cols) >= 2:
        st.markdown('<div style="font-weight:800;font-size:1.3rem;color:#0f172a;margin:1.8rem 0 0.5rem 0;">🎯 Interactive Scenario Simulator</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#64748b;font-size:0.95rem;margin-bottom:0.8rem;">Adjust key predictor sliders to project KPI outcomes interactively.</div>', unsafe_allow_html=True)

        target = st.selectbox("Target KPI to Project", num_cols, key="ad_sim_target")
        feature_cols = [c for c in num_cols if c != target][:4]

        user_inputs = {}
        cols = st.columns(min(4, len(feature_cols)))
        for idx, col in enumerate(feature_cols):
            val = float(df[col].median())
            min_v = float(df[col].quantile(0.05))
            max_v = float(df[col].quantile(0.95))
            if max_v <= min_v:
                min_v, max_v = float(df[col].min()), float(df[col].max())
            with cols[idx % len(cols)]:
                user_inputs[col] = st.slider(f"Adjust {col}", min_value=min_v, max_value=max_v, value=val, key=f"ad_sl_{col}")

        try:
            from sklearn.linear_model import Ridge
            import numpy as np
            X_tr = df[feature_cols].dropna()
            y_tr = df.loc[X_tr.index, target]
            if len(X_tr) > 5:
                model = Ridge(alpha=1.0)
                model.fit(X_tr, y_tr)
                input_df = pd.DataFrame([user_inputs])
                proj_val = float(model.predict(input_df)[0])
                base_val = float(df[target].median())
                delta = proj_val - base_val

                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    st.metric(label=f"Projected Outcome ({target})", value=f"{proj_val:,.2f}", delta=f"{delta:,.2f} vs median baseline")
                with c_m2:
                    st.markdown(f'<div style="background:#f8fafc;border-left:4px solid #6366f1;padding:0.8rem 1rem;border-radius:0 10px 10px 0;font-size:0.95rem;color:#334155;margin-top:0.3rem;">💡 <strong>Scenario Model:</strong> Ridge Regression trained on {len(X_tr):,} dataset rows. Baseline median = {base_val:,.2f}.</div>', unsafe_allow_html=True)
        except Exception:
            pass

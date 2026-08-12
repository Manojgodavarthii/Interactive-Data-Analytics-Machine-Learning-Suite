import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.utils import detect_column_types, render_chart
from modules.ai_engine import correlation_insight, chart_explanation


def _style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
    return fig


def _mutual_info_cat_num(df, cat_col, num_col):
    """Simple approximation of association: eta-squared from one-way ANOVA."""
    groups = [df[num_col][df[cat_col] == v].dropna() for v in df[cat_col].unique()]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0
    grand_mean = df[num_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((df[num_col].dropna() - grand_mean) ** 2).sum()
    return round(ss_between / ss_total, 4) if ss_total > 0 else 0.0


def render():
    st.markdown('<div class="section-title">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df
    col_types = detect_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    tab_heatmap, tab_pairs, tab_catnum, tab_ranked, tab_test = st.tabs([
        "🌡️ Heatmap", "📈 Scatter Pairs", "🏷️ Cat ↔ Numeric", "📊 Ranked Pairs", "🧪 Stats Test"
    ])

    # ─── TAB 1: Heatmap ───────────────────────────────────────────────────
    with tab_heatmap:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")
        else:
            method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"], key="corr_method")
            sel_cols = st.multiselect(
                "Select numeric columns",
                num_cols,
                default=num_cols[:min(10, len(num_cols))],
                key="corr_sel_cols",
            )
            if len(sel_cols) >= 2:
                corr_df = df[sel_cols].corr(method=method).round(3)
                fig = px.imshow(
                    corr_df, text_auto=True, aspect="auto",
                    title=f"{method.capitalize()} Correlation Heatmap",
                    color_continuous_scale="RdBu_r", range_color=[-1, 1],
                )
                fig.update_layout(height=550, paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Inter, sans-serif"))
                render_chart(fig, "corr_heatmap")
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#1a1a2e0d,#0f346008);border-left:4px solid #0f3460;'
                    f'border-radius:0 12px 12px 0;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.88rem;color:#374151;">'
                    f'<span style="font-weight:700;color:#0f3460;">🤖 AI Explanation:</span> '
                    f'{chart_explanation("Correlation Heatmap", "", "", df)}</div>',
                    unsafe_allow_html=True,
                )
                st.subheader("Correlation Matrix")
                st.dataframe(corr_df, use_container_width=True)
                st.download_button("📥 Download Matrix", corr_df.to_csv().encode(), "correlation.csv", "text/csv")

    # ─── TAB 2: Scatter Pairs ─────────────────────────────────────────────
    with tab_pairs:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            corr_ref = df[num_cols].corr("pearson").abs()
            upper = corr_ref.where(np.triu(np.ones(corr_ref.shape), k=1).astype(bool))
            pairs = upper.unstack().dropna().sort_values(ascending=False)
            top_pairs = list(pairs.head(8).index)

            st.markdown(
                '<div style="font-size:0.88rem;color:#6b7280;margin-bottom:1rem;">'
                'Scatter plots for the top correlated numeric column pairs, with OLS trend lines.</div>',
                unsafe_allow_html=True,
            )
            color_col = st.selectbox("Color by (optional)", [None] + cat_cols, key="scatter_color")
            df_sample = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df

            for i in range(0, len(top_pairs), 2):
                cols_ui = st.columns(2)
                for j, col_ui in enumerate(cols_ui):
                    if i + j >= len(top_pairs):
                        break
                    ca, cb = top_pairs[i + j]
                    corr_val = df[num_cols].corr("pearson").loc[ca, cb]
                    with col_ui:
                        fig = px.scatter(
                            df_sample, x=ca, y=cb,
                            color=color_col,
                            trendline="ols",
                            title=f"{ca} vs {cb}  (r={corr_val:.2f})",
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            opacity=0.7,
                        )
                        _style(fig)
                        render_chart(fig, f"scatter_{i}_{j}")
                        insight = correlation_insight(ca, cb, corr_val, "pearson")
                        st.markdown(
                            f'<div style="background:#6366f10d;border-left:3px solid #6366f1;'
                            f'border-radius:0 10px 10px 0;padding:0.5rem 0.8rem;margin-bottom:1rem;'
                            f'font-size:0.8rem;color:#374151;">{insight}</div>',
                            unsafe_allow_html=True,
                        )

    # ─── TAB 3: Categorical ↔ Numeric ─────────────────────────────────────
    with tab_catnum:
        if not cat_cols or not num_cols:
            st.warning("Need at least one categorical and one numeric column.")
        else:
            st.markdown(
                '<div style="font-size:0.88rem;color:#6b7280;margin-bottom:1rem;">'
                'Measures association between categorical and numeric columns using Eta-Squared '
                '(from one-way ANOVA). Values closer to 1 mean stronger association.</div>',
                unsafe_allow_html=True,
            )
            cat_sel = st.selectbox("Categorical column", cat_cols, key="cn_cat")
            num_sel = st.selectbox("Numeric column", num_cols, key="cn_num")

            eta2 = _mutual_info_cat_num(df, cat_sel, num_sel)
            strength = "Strong" if eta2 > 0.14 else "Moderate" if eta2 > 0.06 else "Weak"
            color = "#22c55e" if eta2 > 0.14 else "#f59e0b" if eta2 > 0.06 else "#6b7280"

            st.markdown(
                f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem 1.3rem;'
                f'box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid {color};margin-bottom:1rem;">'
                f'<div style="font-size:1.8rem;font-weight:800;color:{color};">{eta2}</div>'
                f'<div style="font-size:0.8rem;color:#6b7280;font-weight:600;">Eta-Squared — {strength} Association</div>'
                f'<div style="font-size:0.78rem;color:#374151;margin-top:0.4rem;">'
                f'{cat_sel} explains <strong>{eta2*100:.1f}%</strong> of variance in {num_sel}.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Box plot per category
            fig = px.box(df, x=cat_sel, y=num_sel, color=cat_sel,
                         title=f"{num_sel} distribution by {cat_sel}",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_xaxes(tickangle=45)
            _style(fig)
            render_chart(fig, "cat_num_box")

            # All cat x num matrix
            st.markdown("**Full Association Matrix (Eta-Squared)**")
            matrix = {}
            for cc in cat_cols:
                row = {}
                for nc in num_cols:
                    row[nc] = _mutual_info_cat_num(df, cc, nc)
                matrix[cc] = row
            matrix_df = pd.DataFrame(matrix).T
            fig2 = px.imshow(matrix_df, text_auto=True, aspect="auto",
                             title="Eta-Squared: Categorical × Numeric",
                             color_continuous_scale="YlOrRd", range_color=[0, 1])
            fig2.update_layout(height=300 + len(cat_cols) * 30,
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter, sans-serif"))
            render_chart(fig2, "eta_matrix")

    # ─── TAB 4: Ranked Pairs ──────────────────────────────────────────────
    with tab_ranked:
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            method_r = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="rank_method")
            corr_df_r = df[num_cols].corr(method=method_r).round(3)
            upper_r = corr_df_r.where(np.triu(np.ones(corr_df_r.shape), k=1).astype(bool))
            pairs_r = upper_r.unstack().dropna().reset_index()
            pairs_r.columns = ["Column A", "Column B", "Correlation"]
            pairs_r["Abs Correlation"] = pairs_r["Correlation"].abs()
            pairs_r = pairs_r.sort_values("Abs Correlation", ascending=False).reset_index(drop=True)
            pairs_r["Strength"] = pairs_r["Abs Correlation"].apply(
                lambda v: "🔴 Very Strong" if v > 0.8 else "🟠 Strong" if v > 0.6 else "🟡 Moderate" if v > 0.4 else "⚪ Weak"
            )

            c1, c2 = st.columns(2)
            strong = pairs_r[pairs_r["Abs Correlation"] > 0.7]
            weak = pairs_r[pairs_r["Abs Correlation"] < 0.3]
            c1.markdown(f'<div class="insight-box">💪 Strong correlations (>0.7): <strong>{len(strong)}</strong></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="insight-box">🤏 Weak correlations (<0.3): <strong>{len(weak)}</strong></div>', unsafe_allow_html=True)

            st.dataframe(pairs_r[["Column A", "Column B", "Correlation", "Strength"]], use_container_width=True, hide_index=True)

            # Select pair for deep dive
            st.markdown("---")
            st.markdown("**🔍 Deep Dive a Pair**")
            pair_label = st.selectbox(
                "Select pair",
                [f"{r['Column A']} × {r['Column B']}" for _, r in pairs_r.iterrows()],
                key="rank_pair",
            )
            if pair_label:
                ca, cb = pair_label.split(" × ")
                cv = corr_df_r.loc[ca, cb]
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#4CAF5010,#2196F310);border-left:4px solid #2196F3;'
                    f'border-radius:0 12px 12px 0;padding:0.8rem 1rem;margin-bottom:0.8rem;font-size:0.88rem;color:#374151;">'
                    f'{correlation_insight(ca, cb, cv, method_r)}</div>',
                    unsafe_allow_html=True,
                )
                df_s = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df
                fig2 = px.scatter(df_s, x=ca, y=cb, trendline="ols", title=f"{ca} vs {cb}")
                _style(fig2)
                render_chart(fig2, "ranked_deep")

    # ─── TAB 5: Statistical Test ──────────────────────────────────────────
    with tab_test:
        st.markdown(
            '<div style="font-size:0.88rem;color:#6b7280;margin-bottom:1rem;">'
            'Test whether the correlation between two columns is statistically significant.</div>',
            unsafe_allow_html=True,
        )
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            tc1, tc2 = st.columns(2)
            with tc1:
                col_a = st.selectbox("Column A", num_cols, key="test_ca")
            with tc2:
                col_b = st.selectbox("Column B", [c for c in num_cols if c != col_a], key="test_cb")
            method_t = st.selectbox("Method", ["pearson", "spearman"], key="test_method")

            a_vals = df[col_a].dropna()
            b_vals = df[col_b].dropna()
            common = a_vals.index.intersection(b_vals.index)
            a_vals, b_vals = a_vals[common], b_vals[common]

            from scipy import stats as scipy_stats
            if method_t == "pearson":
                r, p = scipy_stats.pearsonr(a_vals, b_vals)
            else:
                r, p = scipy_stats.spearmanr(a_vals, b_vals)

            sig = p < 0.05
            sig_color = "#22c55e" if sig else "#ef4444"
            sig_label = "✅ Statistically Significant" if sig else "❌ Not Significant"

            r_c, p_c, s_c = st.columns(3)
            for label, val, color in [
                ("Correlation (r)", f"{r:.4f}", "#6366f1"),
                ("p-value", f"{p:.6f}", "#f59e0b"),
                ("Significance (α=0.05)", sig_label, sig_color),
            ]:
                pass  # rendered below
            st.markdown(
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1rem;">'
                f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid #6366f1;text-align:center;">'
                f'<div style="font-size:1.5rem;font-weight:800;color:#6366f1;">{r:.4f}</div>'
                f'<div style="font-size:0.72rem;color:#6b7280;font-weight:600;text-transform:uppercase;">Correlation (r)</div></div>'
                f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid #f59e0b;text-align:center;">'
                f'<div style="font-size:1.5rem;font-weight:800;color:#f59e0b;">{p:.6f}</div>'
                f'<div style="font-size:0.72rem;color:#6b7280;font-weight:600;text-transform:uppercase;">p-value</div></div>'
                f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid {sig_color};text-align:center;">'
                f'<div style="font-size:1.1rem;font-weight:800;color:{sig_color};">{sig_label}</div>'
                f'<div style="font-size:0.72rem;color:#6b7280;font-weight:600;text-transform:uppercase;">α = 0.05</div></div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="insight-box">{correlation_insight(col_a, col_b, r, method_t)}</div>',
                unsafe_allow_html=True,
            )
            # Sample scatter
            df_s = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df
            fig = px.scatter(df_s, x=col_a, y=col_b, trendline="ols",
                             title=f"{col_a} vs {col_b}",
                             color_discrete_sequence=["#0f3460"])
            _style(fig)
            render_chart(fig, "test_scatter")

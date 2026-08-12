import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.utils import detect_column_types, render_chart
from modules.ai_engine import column_insight


def _style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=55, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
    return fig


def _insight_box(text, color="#0f3460"):
    return (
        f'<div style="background:linear-gradient(135deg,{color}0d,{color}08);'
        f'border-left:4px solid {color};border-radius:0 12px 12px 0;'
        f'padding:0.8rem 1rem;margin-bottom:0.6rem;font-size:0.88rem;color:#374151;">'
        f'{text}</div>'
    )


def render():
    from scipy import stats as scipy_stats
    st.markdown('<div class="section-title">📈 Statistical Analysis</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df
    col_types = detect_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if not num_cols:
        st.warning("No numeric columns found for statistical analysis.")
        return

    tab_summary, tab_dist, tab_norm, tab_hypo, tab_cat = st.tabs([
        "📋 Summary Stats", "📊 Distributions", "🔔 Normality Tests", "🧪 Hypothesis Tests", "🏷️ Categorical Stats"
    ])

    # ─── TAB 1: Summary Stats ─────────────────────────────────────────────
    with tab_summary:
        sel_cols = st.multiselect(
            "Select columns for analysis", num_cols,
            default=num_cols[:min(5, len(num_cols))],
            key="stat_sel"
        )
        if not sel_cols:
            st.info("Select at least one column above.")
        else:
            stat_data = []
            for col in sel_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                mad = (s - s.mean()).abs().mean()
                cv = s.std() / s.mean() if s.mean() != 0 else np.nan
                ci95_lo = s.mean() - 1.96 * s.std() / np.sqrt(len(s))
                ci95_hi = s.mean() + 1.96 * s.std() / np.sqrt(len(s))
                stat_data.append({
                    "Column": col,
                    "Count": len(s),
                    "Missing": int(df[col].isnull().sum()),
                    "Mean": round(s.mean(), 4),
                    "Median": round(s.median(), 4),
                    "Mode": round(s.mode().iloc[0], 4) if len(s.mode()) > 0 else "N/A",
                    "Min": round(s.min(), 4),
                    "Max": round(s.max(), 4),
                    "Range": round(s.max() - s.min(), 4),
                    "Std Dev": round(s.std(), 4),
                    "Variance": round(s.var(), 4),
                    "MAD": round(mad, 4),
                    "CV (%)": round(cv * 100, 2) if not np.isnan(cv) else "N/A",
                    "Q1 (25%)": round(q1, 4),
                    "Q3 (75%)": round(q3, 4),
                    "IQR": round(iqr, 4),
                    "P5": round(s.quantile(0.05), 4),
                    "P95": round(s.quantile(0.95), 4),
                    "Skewness": round(s.skew(), 4),
                    "Kurtosis": round(s.kurtosis(), 4),
                    "95% CI Lower": round(ci95_lo, 4),
                    "95% CI Upper": round(ci95_hi, 4),
                    "Outliers (3σ)": int((abs((s - s.mean()) / s.std()) > 3).sum()) if s.std() > 0 else 0,
                })
            stat_df = pd.DataFrame(stat_data)
            st.dataframe(stat_df.set_index("Column"), use_container_width=True)
            st.download_button(
                "📥 Download Full Statistics",
                stat_df.to_csv(index=False).encode(),
                "statistics.csv", "text/csv",
            )
            st.markdown("---")
            st.markdown("**🤖 AI Interpretations**")
            for col in sel_cols:
                s = df[col].dropna()
                ins = column_insight(col, s, "numeric")
                st.markdown(_insight_box(f"<span style='font-weight:700;color:#0f3460;'>🤖 {col}:</span> {ins}"), unsafe_allow_html=True)

    # ─── TAB 2: Distributions ─────────────────────────────────────────────
    with tab_dist:
        dist_col = st.selectbox("Select column", num_cols, key="dist_col")
        s = df[dist_col].dropna()

        c_left, c_right = st.columns(2)
        with c_left:
            n_bins = min(60, max(10, int(s.nunique() / 5)))
            fig1 = px.histogram(df, x=dist_col, marginal="box", nbins=n_bins,
                                title=f"Histogram + Boxplot — {dist_col}",
                                color_discrete_sequence=["#6366f1"])
            _style(fig1)
            render_chart(fig1, f"hist_{dist_col}")

        with c_right:
            # Violin
            fig2 = px.violin(df, y=dist_col, box=True, points="outliers",
                             title=f"Violin Plot — {dist_col}",
                             color_discrete_sequence=["#0f3460"])
            _style(fig2)
            render_chart(fig2, f"violin_{dist_col}")

        # Density (KDE approximation via histogram with kde)
        fig3 = px.histogram(df, x=dist_col, histnorm="probability density", nbins=60,
                            title=f"Density Plot (KDE) — {dist_col}", marginal="rug",
                            color_discrete_sequence=["#10b981"])
        _style(fig3)
        render_chart(fig3, f"density_{dist_col}")

        # Stats summary
        skew = s.skew()
        kurt = s.kurtosis()
        direction = "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "symmetric")
        tail = "heavy-tailed (leptokurtic)" if kurt > 2 else ("light-tailed (platykurtic)" if kurt < -2 else "normal-tailed (mesokurtic)")
        cv = s.std() / s.mean() if s.mean() != 0 else 0
        variability = "very low (<10%)" if cv < 0.1 else ("moderate" if cv < 0.3 else "high (>30%)")
        st.markdown(_insight_box(
            f"<strong>{dist_col}:</strong> Distribution is <strong>{direction}</strong> (skewness={skew:.3f}), "
            f"{tail} (kurtosis={kurt:.3f}), and has <strong>{variability}</strong> variability (CV={cv*100:.1f}%)."
        ), unsafe_allow_html=True)

        # Comparison table across all columns
        if len(num_cols) > 1:
            st.markdown("---")
            st.markdown("**📊 Comparison Across All Numeric Columns**")
            comp_rows = []
            for col in num_cols:
                sv = df[col].dropna()
                if len(sv) == 0:
                    continue
                comp_rows.append({
                    "Column": col, "Mean": round(sv.mean(), 3), "Std Dev": round(sv.std(), 3),
                    "Skewness": round(sv.skew(), 3), "Kurtosis": round(sv.kurtosis(), 3),
                    "Outliers": int((abs((sv - sv.mean()) / sv.std()) > 3).sum()) if sv.std() > 0 else 0,
                })
            st.dataframe(pd.DataFrame(comp_rows).set_index("Column"), use_container_width=True)

    # ─── TAB 3: Normality Tests ───────────────────────────────────────────
    with tab_norm:
        st.markdown(
            '<div style="font-size:0.88rem;color:#6b7280;margin-bottom:1rem;">'
            'Normality tests check whether a column follows a normal distribution. '
            '<strong>p > 0.05</strong> → data is likely normally distributed.</div>',
            unsafe_allow_html=True,
        )
        norm_sel = st.multiselect(
            "Select columns", num_cols,
            default=num_cols[:min(6, len(num_cols))],
            key="norm_sel"
        )
        if norm_sel:
            norm_rows = []
            for col in norm_sel:
                s_n = df[col].dropna()
                if len(s_n) < 8:
                    continue
                # Shapiro-Wilk (best for n < 5000)
                sw_s = min(5000, len(s_n))
                try:
                    sw_stat, sw_p = scipy_stats.shapiro(s_n.sample(sw_s, random_state=42) if len(s_n) > sw_s else s_n)
                except Exception:
                    sw_stat, sw_p = np.nan, np.nan
                # D'Agostino K²
                try:
                    da_stat, da_p = scipy_stats.normaltest(s_n)
                except Exception:
                    da_stat, da_p = np.nan, np.nan
                # Kolmogorov-Smirnov
                try:
                    ks_stat, ks_p = scipy_stats.kstest(s_n, "norm", args=(s_n.mean(), s_n.std()))
                except Exception:
                    ks_stat, ks_p = np.nan, np.nan

                norm_rows.append({
                    "Column": col,
                    "Shapiro-Wilk stat": round(sw_stat, 4) if not np.isnan(sw_stat) else "N/A",
                    "Shapiro p": round(sw_p, 6) if not np.isnan(sw_p) else "N/A",
                    "D'Agostino stat": round(da_stat, 4) if not np.isnan(da_stat) else "N/A",
                    "D'Agostino p": round(da_p, 6) if not np.isnan(da_p) else "N/A",
                    "KS stat": round(ks_stat, 4) if not np.isnan(ks_stat) else "N/A",
                    "KS p": round(ks_p, 6) if not np.isnan(ks_p) else "N/A",
                    "Normal?": "✅ Yes" if (sw_p > 0.05 if not np.isnan(sw_p) else False) else "❌ No",
                })

            norm_df = pd.DataFrame(norm_rows).set_index("Column")
            st.dataframe(norm_df, use_container_width=True)

            st.markdown("---")
            # QQ Plot
            qq_col = st.selectbox("QQ Plot for column", norm_sel, key="qq_col")
            s_qq = df[qq_col].dropna()
            theoretical_q = np.sort(scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(s_qq))))
            sample_q = np.sort(s_qq)
            qq_df = pd.DataFrame({"Theoretical Quantiles": theoretical_q, "Sample Quantiles": sample_q})
            fig_qq = px.scatter(
                qq_df, x="Theoretical Quantiles", y="Sample Quantiles",
                title=f"QQ Plot — {qq_col} vs Normal Distribution",
                trendline="ols", color_discrete_sequence=["#0f3460"]
            )
            _style(fig_qq)
            render_chart(fig_qq, "qq_plot")
            st.markdown(
                _insight_box(
                    "Points lying close to the diagonal line indicate normally distributed data. "
                    "Deviations at the tails indicate skewness or heavy tails.",
                    "#6366f1"
                ),
                unsafe_allow_html=True,
            )

    # ─── TAB 4: Hypothesis Tests ──────────────────────────────────────────
    with tab_hypo:
        test_type = st.selectbox("Select Test", [
            "Independent t-test (compare 2 groups)",
            "One-sample t-test (vs known mean)",
            "Mann-Whitney U (non-parametric)",
            "ANOVA (compare 3+ groups)",
            "Chi-Square (categorical independence)",
        ])

        if test_type == "Independent t-test (compare 2 groups)":
            if len(num_cols) < 1 or len(cat_cols) < 1:
                st.warning("Need numeric and categorical columns.")
            else:
                num_c = st.selectbox("Numeric column", num_cols, key="tt_num")
                cat_c = st.selectbox("Group column", cat_cols, key="tt_cat")
                unique_cats = df[cat_c].dropna().unique()
                if len(unique_cats) >= 2:
                    g1_label = st.selectbox("Group 1", unique_cats, key="tt_g1")
                    g2_label = st.selectbox("Group 2", [v for v in unique_cats if v != g1_label], key="tt_g2")
                    g1 = df[df[cat_c] == g1_label][num_c].dropna()
                    g2 = df[df[cat_c] == g2_label][num_c].dropna()
                    if st.button("Run t-test"):
                        t_stat, p_val = scipy_stats.ttest_ind(g1, g2)
                        sig = p_val < 0.05
                        sig_col = "#22c55e" if sig else "#ef4444"
                        st.markdown(
                            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin:1rem 0;">'
                            f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid #6366f1;text-align:center;">'
                            f'<div style="font-size:1.5rem;font-weight:800;color:#6366f1;">{t_stat:.4f}</div>'
                            f'<div style="font-size:0.72rem;color:#6b7280;font-weight:600;text-transform:uppercase;">t-statistic</div></div>'
                            f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid #f59e0b;text-align:center;">'
                            f'<div style="font-size:1.5rem;font-weight:800;color:#f59e0b;">{p_val:.6f}</div>'
                            f'<div style="font-size:0.72rem;color:#6b7280;font-weight:600;text-transform:uppercase;">p-value</div></div>'
                            f'<div style="background:rgba(255,255,255,0.6);border-radius:14px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);border-left:4px solid {sig_col};text-align:center;">'
                            f'<div style="font-size:1.1rem;font-weight:800;color:{sig_col};">{"✅ Significant" if sig else "❌ Not Significant"}</div>'
                            f'<div style="font-size:0.72rem;color:#6b7280;font-weight:600;text-transform:uppercase;">α = 0.05</div></div></div>',
                            unsafe_allow_html=True
                        )
                        conclusion = (
                            f"There <strong>is</strong> a statistically significant difference in <strong>{num_c}</strong> between <strong>{g1_label}</strong> (mean={g1.mean():.3f}) and <strong>{g2_label}</strong> (mean={g2.mean():.3f})."
                            if sig else
                            f"There is <strong>no</strong> statistically significant difference in <strong>{num_c}</strong> between <strong>{g1_label}</strong> and <strong>{g2_label}</strong> (p={p_val:.4f} > 0.05)."
                        )
                        st.markdown(_insight_box(conclusion), unsafe_allow_html=True)
                        fig = px.box(df[df[cat_c].isin([g1_label, g2_label])], x=cat_c, y=num_c, color=cat_c,
                                     title=f"{num_c}: {g1_label} vs {g2_label}",
                                     color_discrete_sequence=["#6366f1", "#f59e0b"])
                        _style(fig)
                        render_chart(fig, "t_test_box")

        elif test_type == "One-sample t-test (vs known mean)":
            num_c = st.selectbox("Numeric column", num_cols, key="ost_num")
            known_mean = st.number_input("Hypothesised mean", value=float(df[num_c].mean()), key="ost_mean")
            if st.button("Run one-sample t-test"):
                s_ost = df[num_c].dropna()
                t_stat, p_val = scipy_stats.ttest_1samp(s_ost, known_mean)
                sig = p_val < 0.05
                sig_col = "#22c55e" if sig else "#ef4444"
                st.markdown(_insight_box(
                    f"Sample mean = <strong>{s_ost.mean():.4f}</strong>, hypothesised mean = <strong>{known_mean}</strong>. "
                    f"t={t_stat:.4f}, p={p_val:.6f} → "
                    f"{'<strong>Significant difference</strong> from the hypothesised mean.' if sig else 'No significant difference from the hypothesised mean.'}"
                , sig_col), unsafe_allow_html=True)

        elif test_type == "Mann-Whitney U (non-parametric)":
            if len(num_cols) < 1 or len(cat_cols) < 1:
                st.warning("Need numeric and categorical columns.")
            else:
                num_c = st.selectbox("Numeric column", num_cols, key="mw_num")
                cat_c = st.selectbox("Group column", cat_cols, key="mw_cat")
                unique_cats = df[cat_c].dropna().unique()
                if len(unique_cats) >= 2:
                    g1_l = st.selectbox("Group 1", unique_cats, key="mw_g1")
                    g2_l = st.selectbox("Group 2", [v for v in unique_cats if v != g1_l], key="mw_g2")
                    if st.button("Run Mann-Whitney U"):
                        g1 = df[df[cat_c] == g1_l][num_c].dropna()
                        g2 = df[df[cat_c] == g2_l][num_c].dropna()
                        u_stat, p_val = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
                        sig = p_val < 0.05
                        st.markdown(_insight_box(
                            f"Mann-Whitney U = {u_stat:.2f}, p = {p_val:.6f}. "
                            f"{'<strong>Significant</strong> difference in distributions.' if sig else 'No significant difference.'}"
                        , "#22c55e" if sig else "#ef4444"), unsafe_allow_html=True)

        elif test_type == "ANOVA (compare 3+ groups)":
            if len(num_cols) < 1 or len(cat_cols) < 1:
                st.warning("Need numeric and categorical columns.")
            else:
                num_c = st.selectbox("Numeric column", num_cols, key="anova_num")
                cat_c = st.selectbox("Group column", cat_cols, key="anova_cat")
                if st.button("Run One-Way ANOVA"):
                    groups = [df[df[cat_c] == v][num_c].dropna() for v in df[cat_c].unique() if len(df[df[cat_c] == v][num_c].dropna()) > 0]
                    f_stat, p_val = scipy_stats.f_oneway(*groups)
                    sig = p_val < 0.05
                    st.markdown(_insight_box(
                        f"F-statistic = {f_stat:.4f}, p = {p_val:.6f}. "
                        f"{'<strong>Significant</strong> differences exist between at least two groups.' if sig else 'No significant differences between groups.'}"
                    , "#22c55e" if sig else "#ef4444"), unsafe_allow_html=True)
                    fig = px.box(df, x=cat_c, y=num_c, color=cat_c,
                                 title=f"ANOVA: {num_c} by {cat_c}",
                                 color_discrete_sequence=px.colors.qualitative.Bold)
                    _style(fig)
                    fig.update_xaxes(tickangle=45)
                    render_chart(fig, "anova_box")

        elif test_type == "Chi-Square (categorical independence)":
            if len(cat_cols) < 2:
                st.warning("Need at least 2 categorical columns.")
            else:
                cc1 = st.selectbox("Column 1", cat_cols, key="chi_c1")
                cc2 = st.selectbox("Column 2", [c for c in cat_cols if c != cc1], key="chi_c2")
                if st.button("Run Chi-Square Test"):
                    ct = pd.crosstab(df[cc1], df[cc2])
                    chi2, p_val, dof, _ = scipy_stats.chi2_contingency(ct)
                    sig = p_val < 0.05
                    cramer_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1))) if len(df) * (min(ct.shape) - 1) > 0 else 0
                    st.markdown(_insight_box(
                        f"χ² = {chi2:.4f}, dof = {dof}, p = {p_val:.6f}, Cramér's V = {cramer_v:.4f}. "
                        f"{'<strong>Significant</strong> association between columns.' if sig else 'No significant association.'} "
                        f"Association strength: {'strong' if cramer_v > 0.5 else 'moderate' if cramer_v > 0.3 else 'weak'}."
                    , "#22c55e" if sig else "#ef4444"), unsafe_allow_html=True)
                    st.dataframe(ct, use_container_width=True)

    # ─── TAB 5: Categorical Stats ─────────────────────────────────────────
    with tab_cat:
        if not cat_cols:
            st.warning("No categorical columns found.")
        else:
            cat_sel_s = st.selectbox("Select categorical column", cat_cols, key="cat_stats_sel")
            freq = df[cat_sel_s].value_counts().reset_index()
            freq.columns = [cat_sel_s, "Count"]
            freq["Percentage (%)"] = (freq["Count"] / freq["Count"].sum() * 100).round(2)
            freq["Cumulative (%)"] = freq["Percentage (%)"].cumsum().round(2)
            st.dataframe(freq, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(freq.head(20), x=cat_sel_s, y="Count", color=cat_sel_s,
                             title=f"Frequency — {cat_sel_s}",
                             color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_xaxes(tickangle=45)
                _style(fig)
                render_chart(fig, f"cat_freq_{cat_sel_s}")
            with c2:
                top_n = freq.head(min(10, len(freq)))
                fig2 = px.pie(top_n, names=cat_sel_s, values="Count",
                              title=f"Top {len(top_n)} — {cat_sel_s}",
                              color_discrete_sequence=px.colors.qualitative.Bold)
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
                render_chart(fig2, f"cat_pie_{cat_sel_s}")

            mode_val = df[cat_sel_s].mode().iloc[0] if len(df[cat_sel_s].mode()) > 0 else "N/A"
            entropy = -(freq["Percentage (%)"] / 100 * np.log(freq["Percentage (%)"] / 100 + 1e-10)).sum()
            st.markdown(_insight_box(
                f"<strong>{cat_sel_s}</strong>: {df[cat_sel_s].nunique()} unique values, "
                f"most common = <strong>{mode_val}</strong> ({freq['Percentage (%)'].iloc[0]:.1f}%). "
                f"Shannon entropy = {entropy:.3f} (higher = more uniform distribution)."
            ), unsafe_allow_html=True)

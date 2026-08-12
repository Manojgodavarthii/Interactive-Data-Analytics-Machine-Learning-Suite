import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types, get_summary_stats, render_chart


def _anomaly_decomposition(df, col_types):
    """Isolation Forest anomaly detection + feature importance (SHAP-style) ranking."""
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    if len(num_cols) < 2:
        return None, "Need at least 2 numeric columns for anomaly decomposition."

    model_df = df[num_cols].dropna().reset_index(drop=True)
    if len(model_df) < 30:
        return None, "Need at least 30 complete rows for reliable anomaly detection."

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(model_df.values)
        iso = IsolationForest(contamination=min(0.1, max(0.01, 5 / len(model_df))), random_state=42, n_estimators=200)
        labels = iso.fit_predict(X)
        model_df["anomaly"] = labels == -1
        n_anom = int(model_df["anomaly"].sum())

        # Feature importance ranking via RandomForest on raw features
        rf = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X, np.arange(len(X)))
        importances = rf.feature_importances_
        ranked = sorted(zip(num_cols, importances), key=lambda x: -x[1])

        # SHAP explanation on the anomaly model if available
        shap_ok = False
        shap_top = []
        try:
            import shap
            explainer = shap.Explainer(rf, X[:200])
            shap_vals = explainer(X[:200])
            mean_shap = np.abs(shap_vals.values).mean(axis=0)
            shap_top = sorted(zip(num_cols, mean_shap), key=lambda x: -x[1])
            shap_ok = True
        except Exception:
            shap_ok = False

        result = {
            "df": model_df,
            "n_anomalies": n_anom,
            "pct": round(n_anom / len(model_df) * 100, 2),
            "num_cols": num_cols,
            "importance": ranked,
            "shap": shap_top if shap_ok else ranked,
            "method": "SHAP (TreeExplainer)" if shap_ok else "RandomForest Importance",
        }
        return result, None
    except Exception as e:
        return None, f"Anomaly decomposition failed: {e}"


def _analyze_root_cause(df, col_types):
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    stats, _ = get_summary_stats(df)
    causes = []

    if stats["missing"] > 0:
        max_miss = df.isnull().sum().idxmax()
        causes.append({"factor": "Data Quality", "cause": f"Missing values in '{max_miss}'", "detail": f"{df[max_miss].isnull().sum():,} records missing ({df[max_miss].isnull().mean()*100:.1f}%)", "severity": "High"})

    if stats["duplicates"] > 0:
        causes.append({"factor": "Data Quality", "cause": f"Duplicate records ({stats['duplicates']} rows)", "detail": "Duplicates can skew analysis results and inflate metrics", "severity": "Medium"})

    for col in num_cols[:3]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        if n_out > len(df) * 0.02:
            causes.append({"factor": "Outliers", "cause": f"Unusual values in '{col}'", "detail": f"{n_out} outliers detected ({n_out/len(df)*100:.1f}% of data)", "severity": "Medium"})

    if cat_cols and num_cols:
        for cat in cat_cols[:2]:
            top = df[cat].value_counts().index[0]
            for num in num_cols[:1]:
                group_means = df.groupby(cat)[num].mean()
                if len(group_means) > 1:
                    highest = group_means.idxmax()
                    lowest = group_means.idxmin()
                    diff_pct = abs(group_means[highest] - group_means[lowest]) / max(abs(group_means[lowest]), 0.01) * 100
                    if diff_pct > 20:
                        causes.append({"factor": "Category Impact", "cause": f"'{cat}' significantly affects {num}", "detail": f"Difference between '{highest}' and '{lowest}': {diff_pct:.0f}%", "severity": "High"})

    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if abs(corr.iloc[i, j]) > 0.7:
                    causes.append({"factor": "Strong Correlation", "cause": f"'{num_cols[i]}' strongly correlated with '{num_cols[j]}'", "detail": f"Correlation: {corr.iloc[i, j]:.2f}. These variables may share underlying factors.", "severity": "Medium"})

    if not causes:
        causes.append({"factor": "Analysis Complete", "cause": "No significant issues detected", "detail": "Dataset appears normal with no root causes identified", "severity": "Low"})

    return causes[:8]


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F50D;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">Root Cause Analysis</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.85rem;margin-bottom:1rem;">Identify underlying factors behind unusual patterns in your data</div>', unsafe_allow_html=True)

    causes = _analyze_root_cause(df, col_types)

    for c in causes:
        sev_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#6b7280"}.get(c["severity"], "#6b7280")
        with st.container():
            st.markdown(f'<div style="background:white;border:1px solid #e4e8f0;border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.6rem;"><div style="display:flex;align-items:center;gap:0.5rem;"><span style="background:{sev_color};width:8px;height:8px;border-radius:50%;display:inline-block;"></span><span style="font-weight:700;font-size:0.9rem;color:#1a1a2e;">{c["factor"]}</span><span style="margin-left:auto;font-size:0.65rem;background:{sev_color}20;color:{sev_color};padding:0.1rem 0.4rem;border-radius:4px;font-weight:600;">{c["severity"]}</span></div><div style="font-size:0.88rem;color:#374151;margin-top:0.25rem;">{c["cause"]}</div><div style="font-size:0.75rem;color:#6b7280;margin-top:0.15rem;">{c["detail"]}</div></div>', unsafe_allow_html=True)

    # ── Anomaly Decomposition (Isolation Forest + SHAP) ────────────────────
    st.markdown('<div style="font-weight:700;font-size:1.1rem;color:#1a1a2e;margin:1.5rem 0 0.5rem 0;">&#x1F50D;&#xFE0F; Anomaly Decomposition (Isolation Forest + SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.82rem;margin-bottom:0.8rem;">Detects unusual records and ranks which variables most contribute to them</div>', unsafe_allow_html=True)

    if st.button("🚨 Run Anomaly Decomposition", type="primary", key="anom_btn"):
        result, err = _anomaly_decomposition(df, col_types)
        if err:
            st.warning(err)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div style="background:rgba(239,68,68,0.08);border:1px solid #fecaca;border-radius:12px;padding:0.8rem;text-align:center;"><div style="font-size:1.6rem;font-weight:800;color:#ef4444;">{result["n_anomalies"]}</div><div style="font-size:0.7rem;color:#6b7280;">Anomalous Records</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div style="background:rgba(99,102,241,0.08);border:1px solid #c7d2fe;border-radius:12px;padding:0.8rem;text-align:center;"><div style="font-size:1.6rem;font-weight:800;color:#6366f1;">{result["pct"]}%</div><div style="font-size:0.7rem;color:#6b7280;">of Dataset</div></div>', unsafe_allow_html=True)

            st.markdown(f'<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:1rem 0 0.3rem 0;">Top Drivers of Anomalies ({result["method"]})</div>', unsafe_allow_html=True)
            import plotly.express as px
            imp_df = pd.DataFrame(result["importance"], columns=["Variable", "Importance"]).head(8)
            fig = px.bar(imp_df, x="Importance", y="Variable", orientation="h",
                         title="Variable Contribution to Anomalies",
                         color="Importance", color_continuous_scale="Reds",
                         text_auto=".4f")
            fig.update_layout(height=320, yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=50, b=30))
            render_chart(fig, "anomaly_importance")

            anom_df = result["df"][result["df"]["anomaly"]].drop(columns=["anomaly"])
            with st.expander(f"📋 View {result['n_anomalies']} Anomalous Records"):
                st.dataframe(anom_df.head(50), use_container_width=True)

            top_driver = result["importance"][0][0]
            st.markdown(f'<div style="background:#fef2f2;border-left:4px solid #ef4444;border-radius:0 12px 12px 0;padding:0.8rem 1rem;font-size:0.88rem;color:#374151;margin-top:0.8rem;"><strong>Root-cause driver:</strong> <strong>{top_driver}</strong> is the leading variable separating anomalies from normal records. Investigate unusual values in this column first, then cross-check the next drivers in the chart above.</div>', unsafe_allow_html=True)

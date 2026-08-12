import streamlit as st
import pandas as pd
import numpy as np


def _safe_div(a, b):
    return round(a / b * 100, 1) if b else 0


def render():
    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F504;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">Dataset Comparison</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.85rem;margin-bottom:0.5rem;">Compare your current dataset with another uploaded dataset</div>', unsafe_allow_html=True)

    df = st.session_state.df

    uploaded = st.file_uploader("Upload comparison dataset (CSV/Excel)", type=["csv", "xlsx"], key="comp_uploader")
    if uploaded:
        try:
            compare_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.session_state._compare_df = compare_df
            st.success(f"Loaded comparison dataset: {len(compare_df):,} rows, {len(compare_df.columns):,} columns")
        except Exception as e:
            st.error(f"Error loading comparison file: {e}")

    if st.session_state.get("_compare_df") is not None:
        comp = st.session_state._compare_df
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Current Rows", len(df))
        with c2:
            st.metric("Comparison Rows", len(comp))
        with c3:
            st.metric("Current Columns", len(df.columns))
        with c4:
            st.metric("Comparison Columns", len(comp.columns))

        common_cols = list(set(df.columns) & set(comp.columns))
        new_cols = [c for c in comp.columns if c not in df.columns]
        removed_cols = [c for c in df.columns if c not in comp.columns]
        if new_cols:
            st.markdown(f'<div style="margin-top:0.5rem;font-size:0.85rem;">&#x2795; New columns in comparison: {", ".join(new_cols)}</div>', unsafe_allow_html=True)
        if removed_cols:
            st.markdown(f'<div style="font-size:0.85rem;color:#dc2626;">&#x2796; Removed columns in comparison: {", ".join(removed_cols)}</div>', unsafe_allow_html=True)

        if common_cols:
            st.markdown(f'<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:0.8rem 0 0.3rem 0;">Column-by-Column Comparison ({len(common_cols)} common columns)</div>', unsafe_allow_html=True)
            diff_rows = []
            for col in common_cols:
                df_count = df[col].nunique()
                comp_count = comp[col].nunique()
                df_miss = int(df[col].isnull().sum())
                comp_miss = int(comp[col].isnull().sum())
                is_diff = "Yes" if (df_count != comp_count or df_miss != comp_miss) else "No"
                diff_rows.append({"Column": col, "Current Unique": df_count, "Comparison Unique": comp_count, "Current Missing": df_miss, "Comparison Missing": comp_miss, "Changed": is_diff})
            diff_df = pd.DataFrame(diff_rows)
            changed = diff_df[diff_df["Changed"] == "Yes"]
            st.dataframe(diff_df, use_container_width=True, hide_index=True)
            if len(changed) > 0:
                st.markdown(f'<div style="color:#dc2626;font-size:0.85rem;margin-top:0.3rem;">&#x26A0; {len(changed)} column(s) have changed between datasets</div>', unsafe_allow_html=True)

            num_common = [c for c in common_cols if df[c].dtype in ("int64", "float64") and comp[c].dtype in ("int64", "float64")]
            if num_common:
                st.markdown('<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:0.8rem 0 0.3rem 0;">KPI Comparison (Numeric Columns)</div>', unsafe_allow_html=True)
                kpi_rows = []
                for col in num_common:
                    df_mean = df[col].mean()
                    comp_mean = comp[col].mean()
                    diff_pct = _safe_div(comp_mean - df_mean, df_mean)
                    direction = "&#x2191;" if diff_pct > 0 else "&#x2193;" if diff_pct < 0 else "&#x2192;"
                    kpi_rows.append({"Column": col, "Current Mean": round(df_mean, 2), "Comparison Mean": round(comp_mean, 2), "Delta %": f"{direction} {abs(diff_pct)}%", "Change": "Increase" if diff_pct > 1 else "Decrease" if diff_pct < -1 else "Stable"})
                st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

        if st.button("Clear Comparison Dataset", use_container_width=True):
            del st.session_state._compare_df
            st.rerun()
    else:
        st.info("Upload a CSV or Excel file above to compare it with the current dataset.")

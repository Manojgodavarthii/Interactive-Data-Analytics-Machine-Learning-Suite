import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.utils import detect_column_types, render_chart
from modules.ai_engine import cleaning_recommendations, data_quality_score
from modules.version_history import record_change


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


def _quality_gauge(score):
    color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    label = "Excellent" if score >= 80 else "Fair" if score >= 60 else "Poor"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Data Quality Score — {label}", "font": {"size": 16, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 60], "color": "rgba(239,68,68,0.1)"},
                {"range": [60, 80], "color": "rgba(245,158,11,0.1)"},
                {"range": [80, 100], "color": "rgba(34,197,94,0.1)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"suffix": "/100", "font": {"size": 32, "color": color}},
    ))
    fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif"), margin=dict(l=20, r=20, t=40, b=10))
    return fig


def _auto_clean_all(df, col_types):
    """Apply all safe automatic cleaning operations."""
    actions = []
    # 1. Remove pure duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        actions.append(f"🗑️ Removed {removed} duplicate rows")

    # 2. Fill numeric missing with median
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    for col in num_cols:
        miss = df[col].isnull().sum()
        if miss > 0 and miss / len(df) < 0.5:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            actions.append(f"🔢 Filled {miss} missing values in '{col}' with median ({median_val:.4f})")

    # 3. Fill categorical missing with mode
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    for col in cat_cols:
        miss = df[col].isnull().sum()
        if miss > 0 and miss / len(df) < 0.5:
            mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown"
            df[col] = df[col].fillna(mode_val)
            actions.append(f"🏷️ Filled {miss} missing values in '{col}' with mode ('{mode_val}')")

    # 4. Drop columns where >50% are missing
    high_miss = [c for c in df.columns if df[c].isnull().sum() / len(df) > 0.5]
    if high_miss:
        df = df.drop(columns=high_miss)
        actions.append(f"❌ Dropped columns with >50% missing: {', '.join(high_miss)}")

    if not actions:
        actions.append("✅ Dataset is already clean — no automatic fixes needed.")

    return df, actions


def render():
    st.markdown('<div class="section-title">🧹 Data Cleaning</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df.copy()
    col_types = detect_column_types(df)
    if "cleaning_history" not in st.session_state:
        st.session_state.cleaning_history = []

    # ── Quality Score ──────────────────────────────────────────────────────
    score, breakdown = data_quality_score(df, col_types)
    c_gauge, c_breakdown = st.columns([1, 1])
    with c_gauge:
        render_chart(_quality_gauge(score), "quality_gauge")
    with c_breakdown:
        st.markdown("**Score Breakdown**")
        for key, val in [
            ("Missing Values (max 40pt)", breakdown["Missing Values"]),
            ("Duplicates (max 30pt)", breakdown["Duplicates"]),
            ("Outliers (max 30pt)", breakdown["Outliers"]),
        ]:
            bar_pct = (val / {"Missing Values (max 40pt)": 40, "Duplicates (max 30pt)": 30, "Outliers (max 30pt)": 30}[key]) * 100
            color = "#22c55e" if bar_pct > 75 else "#f59e0b" if bar_pct > 40 else "#ef4444"
            st.markdown(
                f'<div style="margin-bottom:0.6rem;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;font-weight:600;color:#374151;margin-bottom:0.2rem;">'
                f'<span>{key}</span><span style="color:{color};">{val:.1f}</span></div>'
                f'<div style="background:#f0f0f0;border-radius:8px;height:8px;">'
                f'<div style="background:{color};width:{bar_pct:.0f}%;height:8px;border-radius:8px;transition:width 0.3s;"></div></div></div>',
                unsafe_allow_html=True
            )
        st.markdown(
            f'<div style="margin-top:0.5rem;font-size:0.75rem;color:#6b7280;">'
            f'Missing: <strong>{breakdown["missing_pct"]:.1f}%</strong> Duplicates: <strong>{breakdown["dup_pct"]:.1f}%</strong> Outliers: <strong>{breakdown["outlier_pct"]:.2f}%</strong></div>',
            unsafe_allow_html=True
        )

    # ── Issues Detected ────────────────────────────────────────────────────
    missing_count = df.isnull().sum().sum()
    dup_count = df.duplicated().sum()

    st.markdown('<div class="section-title">⚠️ Detected Issues</div>', unsafe_allow_html=True)
    if missing_count > 0 or dup_count > 0:
        c1, c2 = st.columns(2)
        with c1:
            if missing_count > 0:
                st.markdown(
                    f'<div class="insight-box">🔴 <strong>{missing_count:,}</strong> missing values across dataset</div>',
                    unsafe_allow_html=True,
                )
        with c2:
            if dup_count > 0:
                st.markdown(
                    f'<div class="insight-box">🟡 <strong>{dup_count:,}</strong> duplicate rows detected</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div style="background:white;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);border:1px solid #e4e8f0;border-left:4px solid #4CAF50;text-align:center;padding:1rem;">✅ No data quality issues detected. Your dataset is clean!</div>',
            unsafe_allow_html=True,
        )

    # Missing value heatmap
    if missing_count > 0:
        miss_cols = df.columns[df.isnull().any()].tolist()
        miss_pcts = df[miss_cols].isnull().mean() * 100
        miss_df_plot = miss_pcts.reset_index()
        miss_df_plot.columns = ["Column", "Missing (%)"]
        miss_df_plot = miss_df_plot.sort_values("Missing (%)", ascending=False)
        fig_miss = px.bar(miss_df_plot, x="Column", y="Missing (%)",
                          title="Missing Value % per Column",
                          color="Missing (%)", color_continuous_scale="Reds",
                          range_color=[0, 100])
        fig_miss.update_xaxes(tickangle=45)
        _style(fig_miss)
        fig_miss.update_layout(height=320)
        render_chart(fig_miss, "missing_heatmap")

    # ── AI Recommendations + Auto-Clean ───────────────────────────────────
    st.markdown('<div class="section-title">🤖 AI Cleaning Recommendations</div>', unsafe_allow_html=True)

    c_recs, c_auto = st.columns([3, 1])
    with c_recs:
        for rec in cleaning_recommendations(df, col_types):
            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
    with c_auto:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🪄 Auto-Clean All", use_container_width=True, type="primary", key="auto_clean_btn"):
            cleaned_df, actions = _auto_clean_all(df.copy(), col_types)
            st.session_state.df = cleaned_df
            for a in actions:
                st.session_state.cleaning_history.append(a)
            record_change(cleaned_df, "Auto-Clean", "; ".join(actions))
            st.success(f"✅ Auto-clean applied {len(actions)} operation(s)!")
            st.rerun()
        st.markdown(
            '<div style="font-size:0.72rem;color:#9ca3af;text-align:center;margin-top:0.3rem;">'
            'Fills missing with median/mode, removes duplicates & high-missing columns</div>',
            unsafe_allow_html=True,
        )

    # ── Manual Cleaning Tools ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🔧 Manual Cleaning Tools</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Missing Values", "Duplicates", "Type Conversion", "Outliers", "Column Ops"])

    with tab1:
        miss_cols = df.columns[df.isnull().any()].tolist()
        if miss_cols:
            sel_col = st.selectbox("Select column", miss_cols, key="mv_col")
            affected = df[sel_col].isnull().sum()
            st.info(f"Affected values: {affected} ({affected/len(df)*100:.1f}%)")
            method = st.selectbox("Fill Method", ["Mean", "Median", "Mode", "Custom Value", "Forward Fill", "Backward Fill", "Remove Rows", "Remove Column"])
            custom_val = ""
            if method == "Custom Value":
                custom_val = st.text_input("Enter custom value")
            if st.button("Apply", key="mv_apply"):
                dtype = col_types.get(sel_col, "text")
                if method == "Mean" and dtype == "numeric":
                    df[sel_col] = df[sel_col].fillna(df[sel_col].mean())
                elif method == "Median" and dtype == "numeric":
                    df[sel_col] = df[sel_col].fillna(df[sel_col].median())
                elif method == "Mode":
                    df[sel_col] = df[sel_col].fillna(df[sel_col].mode().iloc[0])
                elif method == "Custom Value" and custom_val:
                    df[sel_col] = df[sel_col].fillna(custom_val)
                elif method == "Forward Fill":
                    df[sel_col] = df[sel_col].ffill()
                elif method == "Backward Fill":
                    df[sel_col] = df[sel_col].bfill()
                elif method == "Remove Rows":
                    df = df.dropna(subset=[sel_col])
                elif method == "Remove Column":
                    df = df.drop(columns=[sel_col])
                st.success(f"Applied {method} on '{sel_col}'")
                st.session_state.cleaning_history.append(f"Missing: {method} on '{sel_col}'")
                st.session_state.df = df
                record_change(df, f"Fill Missing: {sel_col}", f"{method} on '{sel_col}'")
                st.rerun()
        else:
            st.success("No missing values found.")

    with tab2:
        if dup_count > 0:
            st.info(f"{dup_count} duplicate rows found.")
            if st.button("View Duplicates"):
                dups = df[df.duplicated(keep=False)]
                st.dataframe(dups, use_container_width=True)
            keep_opt = st.radio("Keep", ["First", "Last"], horizontal=True)
            if st.button("Remove Duplicates"):
                before = len(df)
                df = df.drop_duplicates(keep="first" if keep_opt == "First" else "last")
                removed = before - len(df)
                st.success(f"Removed {removed} duplicate rows.")
                st.session_state.cleaning_history.append(f"Duplicates: Removed {removed} (keep={keep_opt})")
                st.session_state.df = df
                record_change(df, "Remove Duplicates", f"Removed {removed} rows (keep={keep_opt})")
                st.rerun()
        else:
            st.success("No duplicate rows found.")

    with tab3:
        conv_col = st.selectbox("Select column", df.columns, key="dt_col")
        current_type = col_types.get(conv_col, "unknown")
        st.info(f"Current type: **{current_type}** | Pandas dtype: **{df[conv_col].dtype}**")
        new_type = st.selectbox("Convert to", ["Integer", "Float", "Text", "Date", "Category", "Boolean"])
        if st.button("Convert"):
            try:
                if new_type == "Integer":
                    df[conv_col] = pd.to_numeric(df[conv_col], errors="coerce").astype("Int64")
                elif new_type == "Float":
                    df[conv_col] = pd.to_numeric(df[conv_col], errors="coerce")
                elif new_type == "Text":
                    df[conv_col] = df[conv_col].astype(str)
                elif new_type == "Date":
                    df[conv_col] = pd.to_datetime(df[conv_col], errors="coerce")
                elif new_type == "Category":
                    df[conv_col] = df[conv_col].astype("category")
                elif new_type == "Boolean":
                    df[conv_col] = df[conv_col].astype(bool)
                st.success(f"Converted '{conv_col}' to {new_type}")
                st.session_state.cleaning_history.append(f"Convert: '{conv_col}' → {new_type}")
                st.session_state.df = df
                record_change(df, f"Convert {conv_col}", f"'{conv_col}' → {new_type}")
                st.rerun()
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    with tab4:
        num_cols_ol = [c for c, t in col_types.items() if t == "numeric"]
        if num_cols_ol:
            o_col = st.selectbox("Select numeric column", num_cols_ol, key="out_col")
            method_ol = st.radio("Detection Method", ["IQR", "Z-score"], horizontal=True)
            action_ol = st.radio("Action", ["Remove", "Cap (Winsorize)"], horizontal=True)

            if method_ol == "IQR":
                Q1 = df[o_col].quantile(0.25)
                Q3 = df[o_col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (df[o_col] < Q1 - 1.5 * IQR) | (df[o_col] > Q3 + 1.5 * IQR)
            else:
                from scipy import stats as scipy_stats
                z = np.abs(scipy_stats.zscore(df[o_col].dropna()))
                mask = pd.Series(False, index=df.index)
                mask.loc[df[o_col].dropna().index[z > 3]] = True
            outliers = df[mask]

            st.info(f"Outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

            # Before/after chart
            if len(outliers) > 0:
                c1b, c2b = st.columns(2)
                with c1b:
                    fig_b = px.box(df, y=o_col, title="Before", color_discrete_sequence=["#6366f1"])
                    _style(fig_b)
                    fig_b.update_layout(height=280)
                    render_chart(fig_b, "outlier_box")
                with st.expander("View outlier records"):
                    st.dataframe(outliers, use_container_width=True)

                if st.button("Apply Outlier Treatment"):
                    if action_ol == "Remove":
                        before = len(df)
                        df = df.drop(outliers.index)
                        st.success(f"Removed {before - len(df)} outliers.")
                        st.session_state.cleaning_history.append(f"Outliers: Removed from '{o_col}' ({method_ol})")
                    else:
                        if method_ol == "IQR":
                            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                        else:
                            lower = df[o_col].mean() - 3 * df[o_col].std()
                            upper = df[o_col].mean() + 3 * df[o_col].std()
                        df[o_col] = df[o_col].clip(lower, upper)
                        st.success(f"Capped '{o_col}' to [{lower:.4f}, {upper:.4f}]")
                        st.session_state.cleaning_history.append(f"Outliers: Capped '{o_col}' ({method_ol})")
                    st.session_state.df = df
                    record_change(df, f"Outlier Treatment: {o_col}", f"{action_ol} ({method_ol})")
                    st.rerun()
        else:
            st.info("No numeric columns available.")

    with tab5:
        op = st.selectbox("Operation", ["Rename Column", "Delete Column", "Reorder Columns", "Add Derived Column"])
        if op == "Rename Column":
            rcol = st.selectbox("Select column", df.columns, key="rn_col")
            new_name = st.text_input("New name")
            if st.button("Rename") and new_name:
                df = df.rename(columns={rcol: new_name})
                st.session_state.cleaning_history.append(f"Renamed: '{rcol}' → '{new_name}'")
                st.session_state.df = df
                record_change(df, f"Rename {rcol}", f"'{rcol}' → '{new_name}'")
                st.rerun()
        elif op == "Delete Column":
            dcol = st.multiselect("Select columns to delete", df.columns)
            if st.button("Delete") and dcol:
                df = df.drop(columns=dcol)
                st.session_state.cleaning_history.append(f"Deleted: {dcol}")
                st.session_state.df = df
                record_change(df, "Delete Columns", f"Removed: {', '.join(dcol)}")
                st.rerun()
        elif op == "Reorder Columns":
            rorder = st.multiselect("Select order", df.columns.tolist(), default=df.columns.tolist())
            if st.button("Reorder") and rorder:
                remaining = [c for c in df.columns if c not in rorder]
                df = df[rorder + remaining]
                st.session_state.df = df
                st.session_state.cleaning_history.append("Reordered columns")
                record_change(df, "Reorder Columns", "Reordered column layout")
                st.rerun()
        elif op == "Add Derived Column":
            num_cols_dc = [c for c, t in col_types.items() if t == "numeric"]
            if len(num_cols_dc) >= 2:
                c_a = st.selectbox("Column A", num_cols_dc, key="dc_ca")
                op_dc = st.selectbox("Operation", ["+", "-", "*", "/", "log(A)", "sqrt(A)"], key="dc_op")
                c_b = st.selectbox("Column B", num_cols_dc, key="dc_cb") if op_dc in ["+", "-", "*", "/"] else None
                new_col_name = st.text_input("New column name", value=f"{c_a}_{op_dc}_{c_b or ''}".replace("/", "div"))
                if st.button("Add Column") and new_col_name:
                    try:
                        if op_dc == "+":
                            df[new_col_name] = df[c_a] + df[c_b]
                        elif op_dc == "-":
                            df[new_col_name] = df[c_a] - df[c_b]
                        elif op_dc == "*":
                            df[new_col_name] = df[c_a] * df[c_b]
                        elif op_dc == "/":
                            df[new_col_name] = df[c_a] / df[c_b].replace(0, np.nan)
                        elif op_dc == "log(A)":
                            df[new_col_name] = np.log(df[c_a].replace(0, np.nan))
                        elif op_dc == "sqrt(A)":
                            df[new_col_name] = np.sqrt(df[c_a].clip(0))
                        st.success(f"Added column '{new_col_name}'")
                        st.session_state.cleaning_history.append(f"Derived: '{new_col_name}' = {c_a} {op_dc} {c_b or ''}")
                        st.session_state.df = df
                        record_change(df, f"Add Column {new_col_name}", f"{new_col_name} = {c_a} {op_dc} {c_b or ''}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Need at least 2 numeric columns for derived columns.")

    # ── Cleaning History ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">📜 Cleaning History</div>', unsafe_allow_html=True)
    if st.session_state.cleaning_history:
        for i, h in enumerate(reversed(st.session_state.cleaning_history), 1):
            st.markdown(
                f'<div style="background:white;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);border:1px solid #e4e8f0;padding:0.4rem 1rem;font-size:0.85rem;">'
                f'<span style="color:#6b7280;font-size:0.75rem;">#{len(st.session_state.cleaning_history)-i+1}</span> {h}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No cleaning operations performed yet.")

    col_reset_a, col_reset_b = st.columns(2)
    with col_reset_a:
        if st.button("🔄 Reset to Original Dataset", use_container_width=True):
            if "original_df" in st.session_state and st.session_state.original_df is not None:
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state.cleaning_history = []
                st.session_state._version_history = []
                st.rerun()
    with col_reset_b:
        if st.button("📥 Download Cleaned Dataset", use_container_width=True):
            csv = st.session_state.df.to_csv(index=False).encode()
            st.download_button("Click to Download", csv, "cleaned_dataset.csv", "text/csv")

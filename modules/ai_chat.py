import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types, get_summary_stats
from modules.ai_engine import translate_nl_to_code, safe_eval_df, validate_expression


def _answer_query(query, df, col_types):
    query_lower = query.lower()
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    stats, _ = get_summary_stats(df)

    if any(w in query_lower for w in ["top", "highest", "maximum", "max"]):
        col = None
        for c in num_cols:
            if c.lower() in query_lower:
                col = c
                break
        if not col and num_cols:
            col = num_cols[0]
        if col:
            top = df.nlargest(min(10, len(df)), col)
            return {"type": "table", "title": f"Top {min(10, len(df))} by {col}", "data": top[[col] + cat_cols[:2]] if cat_cols else top[[col]], "insight": f"Highest value: {top[col].max():.2f}"}

    if "missing" in query_lower:
        miss = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
        if not miss.empty:
            return {"type": "table", "title": "Missing Values", "data": miss.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}), "insight": f"Total missing: {int(miss.sum())}"}
        return {"type": "text", "title": "Missing Values", "message": "No missing values found in the dataset."}

    if any(w in query_lower for w in ["outlier", "anomaly", "abnormal"]):
        results = []
        for col in num_cols[:5]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = ((df[col] < low) | (df[col] > high)).sum()
            results.append({"Column": col, "Outliers": n, "Lower Bound": round(low, 2), "Upper Bound": round(high, 2)})
        return {"type": "table", "title": "Outlier Detection", "data": pd.DataFrame(results), "insight": "Values outside 1.5x IQR are flagged as outliers."}

    if any(w in query_lower for w in ["correlation", "correlate", "relationship"]):
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().round(2)
            max_pair = corr.unstack().sort_values(ascending=False)
            max_pair = max_pair[max_pair < 1]
            top_pair = max_pair.index[0] if not max_pair.empty else ("", "")
            return {"type": "table", "title": "Correlation Matrix", "data": corr, "insight": f"Strongest correlation: {top_pair[0]} vs {top_pair[1]} ({max_pair.iloc[0]:.2f})" if top_pair[0] else "No strong correlations found."}
        return {"type": "text", "title": "Correlation", "message": "Need at least 2 numeric columns for correlation analysis."}

    if any(w in query_lower for w in ["describe", "explain", "summary", "overview"]):
        desc = df[num_cols].describe().round(2) if num_cols else pd.DataFrame()
        return {"type": "table", "title": "Dataset Summary", "data": desc, "insight": f"{stats['rows']:,} rows, {stats['columns']} columns, {stats['missing']} missing"}

    if any(w in query_lower for w in ["salary", "compensation", "pay"]):
        if "salary" in df.columns.str.lower().values or any("salary" in c.lower() for c in df.columns):
            sal_col = [c for c in df.columns if "salary" in c.lower()]
            dept_col = [c for c in cat_cols if "dept" in c.lower() or "department" in c.lower()]
            if sal_col:
                result = df[sal_col + dept_col].head(10) if dept_col else df[sal_col].head(10)
                avg_sal = df[sal_col[0]].mean()
                return {"type": "table", "title": "Salary Data", "data": result, "insight": f"Average salary: {avg_sal:,.2f} | Total: {df[sal_col[0]].sum():,.2f}"}
        return {"type": "text", "title": "Salary", "message": "No salary column found in this dataset."}

    if any(w in query_lower for w in ["department", "dept", "division"]):
        dept_col = [c for c in cat_cols if "dept" in c.lower() or "department" in c.lower() or "division" in c.lower()]
        if dept_col:
            counts = df[dept_col[0]].value_counts().reset_index()
            counts.columns = [dept_col[0], "Count"]
            return {"type": "table", "title": "Department Distribution", "data": counts, "insight": f"{len(counts)} departments. Largest: {counts.iloc[0][dept_col[0]]} ({counts.iloc[0]['Count']})"}
        return {"type": "text", "title": "Department", "message": "No department column detected."}

    if any(w in query_lower for w in ["trend", "change", "growth"]):
        if date_cols and num_cols:
            return {"type": "text", "title": "Trend Analysis", "message": f"Date columns: {', '.join(date_cols[:3])}. Numeric columns for trend: {', '.join(num_cols[:3])}. Use 'Visualizations' page for detailed trend charts."}
        return {"type": "text", "title": "Trend", "message": "No date column available for trend analysis."}

    if any(w in query_lower for w in ["compare", "male", "female", "gender"]):
        gender_col = [c for c in df.columns if "gender" in c.lower() or "sex" in c.lower()]
        if gender_col and num_cols:
            comp = df.groupby(gender_col[0])[num_cols[0]].describe().round(2)
            return {"type": "table", "title": "Gender Comparison", "data": comp, "insight": f"Comparing {num_cols[0]} by {gender_col[0]}"}
        return {"type": "text", "title": "Comparison", "message": "No gender column or numeric data found for comparison."}

    if any(w in query_lower for w in ["report", "generate"]):
        return {"type": "text", "title": "Report", "message": f"Dataset: {stats['size']}. Quality issues: {stats['missing']} missing, {stats['duplicates']} duplicates. Use 'Report Generation' page for full export."}

    if any(w in query_lower for w in ["dashboard", "kpi"]):
        return {"type": "text", "title": "Dashboard", "message": f"Stats: {stats['rows']:,} rows, {stats['columns']} cols, {stats['missing']} missing. Use 'Auto Dashboard' page for full dashboard view."}

    return {"type": "text", "title": "AI Response", "message": f"I found {stats['rows']:,} rows and {stats['columns']} columns in your dataset. Try asking about: top values, missing data, outliers, correlation, salary, departments, trends, or gender comparison."}


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F4AC;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">AI Chat with Dataset</span></div>', unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "result" in msg:
                r = msg["result"]
                if r.get("type") == "table" and r.get("data") is not None:
                    st.dataframe(r["data"], use_container_width=True, height=min(300, 40 * (len(r["data"]) + 1)))
                if r.get("insight"):
                    st.caption(f"💡 {r['insight']}")

    suggestions = ["Show top 10 salaries", "Which department has highest salary?", "Find missing values", "Predict next month's sales", "Explain the dataset", "Create dashboard", "Show correlation", "Find outliers", "Compare males and females", "Generate report"]
    with st.expander("Suggested Questions", expanded=False):
        for i, sq in enumerate(suggestions):
            if st.button(sq, key=f"sq_suggest_btn_{i}_{abs(hash(sq))}", use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": sq})
                result = _answer_query(sq, df, col_types)
                msg = result.get("message", result.get("insight", ""))
                st.session_state.chat_messages.append({"role": "assistant", "content": msg, "result": result})
                st.rerun()

    query = st.chat_input("Ask anything about your dataset...")
    if query:
        st.session_state.chat_messages.append({"role": "user", "content": query})
        result = _answer_query(query, df, col_types)
        msg = result.get("message", result.get("insight", ""))
        st.session_state.chat_messages.append({"role": "assistant", "content": msg, "result": result})
        st.rerun()

    # ── Safe Natural Language → Pandas Sandbox ─────────────────────────────
    st.markdown('<div style="font-weight:700;font-size:1.1rem;color:#1a1a2e;margin:1.5rem 0 0.3rem 0;">&#x1F6E1;&#xFE0F; Safe NL → Pandas Sandbox</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.82rem;margin-bottom:0.5rem;">Ask in plain English — the app translates it to sandboxed pandas code, validates it with an AST whitelist (no imports, no file/system access), and runs it against <code>_df</code>.</div>', unsafe_allow_html=True)

    nl_query = st.text_input("Natural language query", key="nl_query", placeholder="e.g., top 5 rows by salary, or average of revenue")
    c_sand, c_sand2 = st.columns([1, 1])
    with c_sand:
        run_nl = st.button("🪄 Translate & Run", type="primary", key="run_nl", use_container_width=True)
    with c_sand2:
        show_code = st.checkbox("Show generated code", value=True, key="show_nl_code")

    if run_nl and nl_query.strip():
        code, label = translate_nl_to_code(nl_query, df, col_types)
        if code is None:
            st.warning("Could not translate this query. Try: 'top 5 by <column>', 'average of <column>', 'sum of <column>', 'missing values', 'correlation', 'describe', 'value counts of <column>', 'row count'.")
        else:
            if show_code:
                st.markdown(f'<div style="background:#1e1e2e;color:#cdd6f4;border-radius:8px;padding:0.6rem 0.9rem;font-family:monospace;font-size:0.8rem;margin-bottom:0.4rem;">{code}</div>', unsafe_allow_html=True)
                st.caption(f"✅ Passed AST whitelist validation — safe to execute")
            try:
                result = safe_eval_df(code, df)
                st.markdown(f'<div style="font-weight:600;font-size:0.9rem;color:#1a1a2e;margin-top:0.3rem;">Result: {label}</div>', unsafe_allow_html=True)
                if isinstance(result, (pd.Series, pd.DataFrame)):
                    st.dataframe(result if isinstance(result, pd.DataFrame) else result.reset_index(), use_container_width=True, height=min(320, 35 * (len(result) + 1)))
                else:
                    st.markdown(f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:0.8rem;font-size:1.1rem;font-weight:700;color:#166534;text-align:center;">{result:,.4f}</div>' if isinstance(result, (int, float)) else st.markdown(f'<div style="font-size:0.9rem;">{result}</div>'))
            except Exception as e:
                st.error(f"Blocked / failed: {e}")

    st.markdown('<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:1.2rem 0 0.3rem 0;">&#x1F4BB; Advanced: Write Safe Pandas Code</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.82rem;margin-bottom:0.4rem;">Use <code>_df</code> to reference your dataset. Only whitelisted pandas/numpy operations run — imports, files, and system calls are blocked.</div>', unsafe_allow_html=True)
    custom_code = st.text_area("Pandas expression", key="custom_code", height=80, placeholder="_df.groupby('department')['salary'].mean().sort_values(ascending=False)")
    if st.button("▶️ Execute Sandboxed Code", key="exec_code", type="primary", use_container_width=True) and custom_code.strip():
        ok, err = validate_expression(custom_code)
        if not ok:
            st.error(f"❌ Blocked: {err}")
        else:
            try:
                result = safe_eval_df(custom_code, df)
                if isinstance(result, (pd.Series, pd.DataFrame)):
                    st.dataframe(result if isinstance(result, pd.DataFrame) else result.reset_index(), use_container_width=True, height=min(320, 35 * (len(result) + 1)))
                else:
                    st.markdown(f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:0.8rem;font-size:1.1rem;font-weight:700;color:#166534;">{result}</div>' if isinstance(result, (int, float)) else st.markdown(f'<div style="font-size:0.9rem;">{result}</div>'))
            except Exception as e:
                st.error(f"❌ {e}")

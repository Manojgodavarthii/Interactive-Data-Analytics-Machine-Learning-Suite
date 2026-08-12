import streamlit as st
import pandas as pd
import numpy as np
import duckdb


def render():
    df = st.session_state.df

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F4BE;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">SQL Query Editor</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.85rem;margin-bottom:0.5rem;">Run SQL queries directly on your dataset. The table is named <code>dataset</code>.</div>', unsafe_allow_html=True)

    with st.expander("Sample Queries", expanded=False):
        samples = [
            "SELECT * FROM dataset LIMIT 10",
            "SELECT COUNT(*) as total_rows FROM dataset",
            "SELECT column_name, COUNT(*) as count FROM dataset GROUP BY column_name ORDER BY count DESC LIMIT 5",
            "SELECT AVG(numeric_column) as avg_value FROM dataset",
            "SELECT col1, col2, col1 * 1.1 as calculated FROM dataset LIMIT 20",
        ]
        for i, s in enumerate(samples):
            if st.button(s, key=f"sql_sample_btn_{i}_{abs(hash(s))}", use_container_width=True):
                st.session_state._sql_query = s

    default_sql = st.session_state.get("_sql_query", "SELECT * FROM dataset LIMIT 100")
    sql = st.text_area("SQL Query", value=default_sql, height=100, placeholder="SELECT * FROM dataset LIMIT 100")

    c1, c2 = st.columns([1, 3])
    with c1:
        run = st.button("Run Query", type="primary", use_container_width=True)
    with c2:
        show_schema = st.checkbox("Show Schema", value=True)

    if show_schema:
        with st.expander("Dataset Schema", expanded=False):
            schema_rows = []
            for col in df.columns:
                schema_rows.append({"Column": col, "Type": str(df[col].dtype), "Sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"})
            st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    if run or st.session_state.get("_sql_query"):
        if run:
            st.session_state._sql_query = sql
        try:
            result = duckdb.query(sql.replace("dataset", "df")).df()
            st.markdown(f'<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.3rem;">&#x2705; Query executed successfully &middot; {len(result):,} rows &times; {len(result.columns)} cols</div>', unsafe_allow_html=True)
            st.dataframe(result, use_container_width=True, height=min(400, 35 * (len(result) + 1)))

            if len(result.columns) >= 2 and len(result) > 1:
                st.markdown('<div style="font-weight:700;font-size:0.9rem;color:#1a1a2e;margin-top:0.5rem;">Visualize Results</div>', unsafe_allow_html=True)
                num_cols = result.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter", "Area"])
                    x_col = st.selectbox("X axis", result.columns.tolist())
                    y_col = st.selectbox("Y axis", num_cols)
                    import plotly.express as px
                    fig_map = {"Bar": px.bar, "Line": px.line, "Scatter": px.scatter, "Area": px.area}
                    fig = fig_map[chart_type](result, x=x_col, y=y_col, title=f"{chart_type} Chart of {y_col} by {x_col}")
                    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=60))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
            st.session_state._sql_query = ""
        except Exception as e:
            st.error(f"Query Error: {e}")
            st.session_state._sql_query = ""

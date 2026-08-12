import streamlit as st
import pandas as pd
from modules.utils import *

def render():
    st.markdown('<div class="section-title">📄 Raw Dataset</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state.df.copy()
    col_options = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist())
    per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
    search_term = st.text_input("🔎 Search records", placeholder="Search across all columns...")
    df_display = df[col_options] if col_options else df
    if search_term:
        mask = df_display.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        df_display = df_display[mask]
    st.markdown(f'<div style="color:#6b7280;font-size:0.85rem;margin-bottom:0.5rem;">Showing {len(df_display)} rows | {len(df_display.columns)} columns</div>', unsafe_allow_html=True)
    st.dataframe(df_display, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download as CSV", csv_data, "raw_dataset.csv", "text/csv")
    with col2:
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("📥 Download as Excel", buf.getvalue(), "raw_dataset.xlsx")
    with col3:
        json_data = df.to_json(orient="records").encode("utf-8")
        st.download_button("📥 Download as JSON", json_data, "raw_dataset.json", "application/json")
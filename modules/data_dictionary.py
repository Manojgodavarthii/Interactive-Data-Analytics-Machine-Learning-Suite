import streamlit as st
import pandas as pd
import numpy as np


def render():
    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F504;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">Data Dictionary</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6b7280;font-size:0.85rem;margin-bottom:1rem;">Auto-generated column reference for the current dataset</div>', unsafe_allow_html=True)

    df = st.session_state.df
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        unique = df[col].nunique()
        sample_vals = ", ".join(str(v) for v in df[col].dropna().unique()[:3])
        dtype_map = {"int64": "Integer", "float64": "Float", "object": "Text", "datetime64[ns]": "Date", "bool": "Boolean"}
        col_type = dtype_map.get(dtype, dtype)
        total = len(df)
        completeness = round((1 - missing / max(total, 1)) * 100, 1)
        is_id = unique == total and total > 100
        is_cat = unique <= 20 and total > 100
        is_num = dtype in ("int64", "float64")
        inference = "ID / Primary Key" if is_id else "Categorical" if is_cat else "Numeric" if is_num else "Text / Description"
        rows.append({"Column": col, "Type": col_type, "Inferred Role": inference, "Missing": missing, "Completeness": f"{completeness}%", "Unique": unique, "Sample": sample_vals})
    dict_df = pd.DataFrame(rows)
    st.dataframe(dict_df, use_container_width=True, hide_index=True)
    st.download_button("Download as CSV", dict_df.to_csv(index=False).encode("utf-8"), "data_dictionary.csv", "text/csv", use_container_width=True)

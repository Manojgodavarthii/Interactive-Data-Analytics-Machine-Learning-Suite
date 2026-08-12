import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import *

def render():
    st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state.df
    col_types = detect_column_types(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dataset Name", st.session_state.get("filename", "N/A"))
    with c2:
        ext = "CSV" if st.session_state.get("filename", "").endswith(".csv") else "Excel"
        st.metric("File Type", ext)
    with c3:
        st.metric("Records", len(df))
    with c4:
        st.metric("Columns", len(df.columns))
    st.markdown('<div class="section-title">🔍 Column Details</div>', unsafe_allow_html=True)
    search_col = st.text_input("🔎 Search column", placeholder="Type column name...")
    type_filter = st.multiselect("Filter by type", ["numeric", "categorical", "date", "text", "boolean"])
    rows_data = []
    for col in df.columns:
        dtype = col_types[col]
        if type_filter and dtype not in type_filter:
            continue
        if search_col and search_col.lower() not in col.lower():
            continue
        nulls = int(df[col].isnull().sum())
        null_pct = round(nulls / len(df) * 100, 2)
        uniq = df[col].nunique()
        dups = len(df) - uniq
        row = {"Column": col, "Type": dtype, "Nulls": nulls, "Missing %": null_pct, "Unique": uniq, "Duplicates": dups}
        if dtype == "numeric":
            row["Min"] = round(df[col].min(), 2) if df[col].notna().any() else "N/A"
            row["Max"] = round(df[col].max(), 2) if df[col].notna().any() else "N/A"
            row["Mean"] = round(df[col].mean(), 2) if df[col].notna().any() else "N/A"
            row["Median"] = round(df[col].median(), 2) if df[col].notna().any() else "N/A"
            row["Std"] = round(df[col].std(), 2) if df[col].notna().any() else "N/A"
        elif dtype == "text":
            non_null = df[col].dropna().astype(str)
            row["Longest"] = non_null.map(len).max() if len(non_null) > 0 else "N/A"
            row["Shortest"] = non_null.map(len).min() if len(non_null) > 0 else "N/A"
            row["Most Common"] = non_null.mode().iloc[0] if len(non_null) > 0 else "N/A"
        elif dtype == "date":
            dates = pd.to_datetime(df[col], errors="coerce").dropna()
            row["Earliest"] = str(dates.min().date()) if len(dates) > 0 else "N/A"
            row["Latest"] = str(dates.max().date()) if len(dates) > 0 else "N/A"
        rows_data.append(row)
    summary_df = pd.DataFrame(rows_data)
    if search_col and not summary_df.empty:
        summary_df = summary_df[summary_df["Column"].str.contains(search_col, case=False)]
    st.dataframe(summary_df, use_container_width=True)
    csv_data = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Overview as CSV", csv_data, "dataset_overview.csv", "text/csv")
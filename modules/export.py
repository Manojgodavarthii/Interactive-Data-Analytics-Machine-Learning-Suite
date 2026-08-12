import streamlit as st
import pandas as pd
from modules.utils import detect_column_types, get_summary_stats, auto_insights
from zipfile import ZipFile
from io import BytesIO
from modules.ai_engine import analyze_dataset


def render():
    st.markdown('<div style="display:flex;align-items:center;gap:0.75rem;padding:1.2rem 0 0.5rem 0;"><span style="font-size:2rem;">💾</span><span style="font-weight:800;font-size:1.7rem;color:#0f172a;">Instant Data & Analytics Exporter</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.95rem;margin-bottom:1.2rem;">Direct 1-click exports for raw datasets, cleaned data, statistical analyses, correlation matrices, and complete ZIP packages.</div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df
    col_types = detect_column_types(df)
    stats, _ = get_summary_stats(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]

    # 1. Raw Dataset Exports
    st.markdown('<div style="font-weight:800;font-size:1.25rem;color:#0f172a;margin:1.4rem 0 0.6rem 0;">📄 Raw Dataset Exports</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Raw CSV", csv_data, "raw_dataset.csv", "text/csv", use_container_width=True)
    with c2:
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("📥 Export Raw Excel", buf.getvalue(), "raw_dataset.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with c3:
        json_data = df.to_json(orient="records").encode("utf-8")
        st.download_button("📥 Export Raw JSON", json_data, "raw_dataset.json", "application/json", use_container_width=True)

    # 2. Data Cleaning & Sanitized Dataset
    st.markdown('<div style="font-weight:800;font-size:1.25rem;color:#0f172a;margin:1.8rem 0 0.6rem 0;">🧹 Cleaned Dataset & Cleaning Audit</div>', unsafe_allow_html=True)
    cl1, cl2 = st.columns(2)
    with cl1:
        cleaned_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned Dataset (CSV)", cleaned_csv, "cleaned_dataset.csv", "text/csv", use_container_width=True)
    with cl2:
        cleaning_history = st.session_state.get("cleaning_history", ["No cleaning transformations performed."])
        audit_log = "Cleaned Audit History:\n" + "\n".join(f"- {item}" for item in cleaning_history)
        st.download_button("📥 Download Cleaning Audit Log", audit_log.encode("utf-8"), "cleaning_audit.txt", "text/plain", use_container_width=True)

    # 3. Statistical Analysis & Correlation Matrix
    st.markdown('<div style="font-weight:800;font-size:1.25rem;color:#0f172a;margin:1.8rem 0 0.6rem 0;">📈 Statistical & Correlation Matrices</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        if num_cols:
            stat_df = df[num_cols].describe().round(3)
            csv_stat = stat_df.to_csv().encode("utf-8")
            st.download_button("📥 Download Statistical Analysis CSV", csv_stat, "statistical_analysis.csv", "text/csv", use_container_width=True)
        else:
            st.info("No numeric columns for statistical export.")
    with s2:
        if len(num_cols) >= 2:
            corr_df = df[num_cols].corr().round(3)
            csv_corr = corr_df.to_csv().encode("utf-8")
            st.download_button("📥 Download Correlation Matrix CSV", csv_corr, "correlation_matrix.csv", "text/csv", use_container_width=True)
        else:
            st.info("Requires at least 2 numeric columns for correlation export.")

    # 4. Complete Package Archive (ZIP)
    st.markdown('<div style="font-weight:800;font-size:1.25rem;color:#0f172a;margin:2rem 0 0.6rem 0;">📦 Complete Analytics Archive (ZIP)</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.92rem;margin-bottom:0.8rem;">Package raw data, cleaned dataset, statistical metrics, correlation matrix, and AI insights into a single ZIP file.</div>', unsafe_allow_html=True)

    zip_buf = BytesIO()
    insights = auto_insights(df, col_types)
    with ZipFile(zip_buf, "w") as zf:
        zf.writestr("dataset.csv", df.to_csv(index=False))
        excel_buf = BytesIO()
        df.to_excel(excel_buf, index=False)
        zf.writestr("dataset.xlsx", excel_buf.getvalue())
        zf.writestr("dataset.json", df.to_json(orient="records"))
        if num_cols:
            zf.writestr("statistical_analysis.csv", df[num_cols].describe().round(3).to_csv())
        if len(num_cols) >= 2:
            zf.writestr("correlation_matrix.csv", df[num_cols].corr().round(3).to_csv())
        if insights:
            zf.writestr("ai_insights.txt", "\n".join(insights))

    st.download_button(
        "📥 Package & Download Complete ZIP Archive",
        zip_buf.getvalue(),
        "complete_analytics_package.zip",
        "application/zip",
        type="primary",
        use_container_width=True
    )
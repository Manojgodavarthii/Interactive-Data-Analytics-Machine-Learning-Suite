import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types


def _compute_dqi(df, col_types):
    """Dynamic Data Quality Index — aggregate of 4 sub-metrics (0-100% each)."""
    n = len(df)
    total_cells = n * len(df.columns)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    dqi = {}

    # 1. Completeness
    missing = int(df.isnull().sum().sum())
    dqi["Completeness"] = round(max(0, min(100, (1 - missing / max(total_cells, 1)) * 100)), 1)

    # 2. Uniqueness
    dup_pct = df.duplicated().sum() / max(n, 1)
    uniq_ratio = df.nunique().mean() / max(n, 1) if n > 0 else 0
    dqi["Uniqueness"] = round(max(0, min(100, (1 - dup_pct) * 70 + min(uniq_ratio * 30, 30))), 1)

    # 3. Type Validity
    invalid = 0
    for col, t in col_types.items():
        s = df[col].dropna()
        if len(s) == 0:
            continue
        if t == "numeric":
            invalid += int((np.abs(s - s.mean()) > 6 * s.std()).sum())
        elif t == "date":
            invalid += int(s.apply(lambda v: pd.isna(pd.to_datetime(v, errors="coerce"))).sum())
        elif t == "boolean":
            invalid += int(~s.astype(str).str.lower().isin(["true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"]))
    dqi["Type Validity"] = round(max(0, min(100, 100 - invalid / max(total_cells, 1) * 100)), 1)

    # 4. Outlier Ratio
    outliers = 0
    for col in num_cols:
        s = df[col].dropna()
        if len(s) > 1:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers += int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    dqi["Outlier Ratio"] = round(max(0, min(100, 100 - outliers / max(len(df), 1) * 5)), 1)

    weights = {"Completeness": 0.35, "Uniqueness": 0.15, "Type Validity": 0.25, "Outlier Ratio": 0.25}
    dqi["DQI"] = round(sum(dqi[k] * w for k, w in weights.items()), 1)
    return dqi


def _audit_trail(df, col_types):
    """Audit checks: zero-variance, target leakage, multicollinearity (VIF>10)."""
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    findings = []

    for col in df.columns:
        if df[col].nunique() <= 1:
            findings.append({"type": "Zero-Variance", "column": col, "severity": "High",
                             "detail": f"Constant value '{df[col].iloc[0]}' — no predictive power, may cause model instability"})

    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c = corr.iloc[i, j]
                if abs(c) > 0.95:
                    findings.append({"type": "Target Leakage", "column": num_cols[j], "severity": "High",
                                     "detail": f"'{num_cols[j]}' nearly identical to '{num_cols[i]}' (r={c:.3f}) — leak risk if both used as features"})

    if len(num_cols) >= 2:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.tools.tools import add_constant
            X = df[num_cols].dropna()
            if len(X) > len(num_cols) and len(X) > 0:
                Xc = add_constant(X)
                if Xc.shape[1] > 2:
                    for idx, col in enumerate(Xc.columns[1:], start=1):
                        vif = variance_inflation_factor(Xc.values, idx)
                        if vif > 10:
                            findings.append({"type": "Multicollinearity", "column": col, "severity": "Medium",
                                             "detail": f"VIF = {vif:.1f} (>10) — {col} is highly explained by other columns"})
        except Exception:
            pass

    return findings[:12]


def _compute_health_scores(df):
    total = len(df)
    total_cells = total * len(df.columns)
    scores = {}

    missing_count = int(df.isnull().sum().sum())
    scores["completeness"] = round(100 - (missing_count / max(total_cells, 1)) * 100, 1)
    scores["completeness"] = max(0, min(100, scores["completeness"]))

    dup_count = int(df.duplicated().sum())
    scores["duplicates"] = round(100 - (dup_count / max(total, 1)) * 100, 1)

    num_cols = df.select_dtypes(include=np.number).columns
    outlier_count = 0
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count += int(((df[col] < low) | (df[col] > high)).sum())
    scores["outliers"] = round(100 - (outlier_count / max(total_cells, 1)) * 100, 1)

    col_types = detect_column_types(df)
    type_issues = sum(1 for c, t in col_types.items() if t == "text" and df[c].nunique() < 5)
    scores["consistency"] = round(100 - (type_issues / max(len(df.columns), 1)) * 50, 1)

    null_issues = sum(1 for c in df.columns if df[c].isnull().sum() / max(len(df), 1) > 0.5)
    scores["null_values"] = round(100 - (null_issues / max(len(df.columns), 1)) * 100, 1)
    scores["null_values"] = max(0, min(100, scores["null_values"]))

    invalid_nums = 0
    for col in num_cols:
        invalid_nums += int((df[col] < 0).sum()) if col not in ["temperature", "balance"] else 0
    scores["data_types"] = round(100 - (invalid_nums / max(total_cells, 1)) * 50, 1)

    unique_ratios = [df[c].nunique() / max(len(df), 1) for c in df.columns]
    avg_ratio = np.mean(unique_ratios) if unique_ratios else 0
    scores["unique_values"] = round(min(100, avg_ratio * 100 + 20), 1)

    fmt_issues = 0
    for c in df.columns:
        try:
            if df[c].dtype == "object" and df[c].astype(str).str.contains(r"[^\w\s\-\.]", na=False).sum() > len(df) * 0.1:
                fmt_issues += 1
        except Exception:
            pass
    scores["formatting"] = round(100 - fmt_issues * 10, 1)
    scores["formatting"] = max(0, min(100, scores["formatting"]))

    weights = {"completeness": 20, "consistency": 15, "duplicates": 15, "outliers": 15, "data_types": 10, "null_values": 10, "unique_values": 8, "formatting": 7}
    overall = sum(scores[k] * weights[k] / 100 for k in weights)
    scores["overall"] = round(overall, 1)

    problems = []
    if missing_count > 0:
        problems.append(f"{missing_count:,} Missing Values Detected")
    if dup_count > 0:
        problems.append(f"{dup_count:,} Duplicate Rows Detected")
    for col in num_cols:
        neg_count = int((df[col] < 0).sum())
        if neg_count > 0 and col not in ["temperature", "balance"]:
            problems.append(f"Column '{col}' has {neg_count:,} negative values")
    return scores, problems[:6]


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)
    scores, problems = _compute_health_scores(df)

    st.markdown('<div style="display:flex;align-items:center;gap:0.75rem;padding:1.2rem 0 0.5rem 0;"><span style="font-size:2.2rem;">❤️</span><span style="font-weight:800;font-size:1.8rem;color:#0f172a;">Dataset Health & Quality Audit</span></div>', unsafe_allow_html=True)

    overall = scores["overall"]
    color = "#22c55e" if overall >= 80 else "#f59e0b" if overall >= 60 else "#ef4444"
    st.markdown(
        f'<div style="text-align:center;padding:2.5rem 1.5rem;background:linear-gradient(135deg,#ffffff 0%,#f0f4ff 100%);border-radius:24px;margin:1.2rem 0;border:1px solid #cbd5e1;box-shadow:0 8px 30px rgba(15,23,42,0.06);">'
        f'<div style="font-size:4.8rem;font-weight:900;color:{color};line-height:1;letter-spacing:-1px;">{overall}%</div>'
        f'<div style="font-size:1.35rem;color:#1e293b;font-weight:800;margin-top:0.6rem;">Overall Dataset Health Score</div>'
        f'<div style="width:300px;height:12px;background:#e2e8f0;border-radius:6px;margin:1.2rem auto;overflow:hidden;"><div style="width:{overall}%;height:100%;background:{color};border-radius:6px;transition:width 0.5s;"></div></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div style="font-weight:800;font-size:1.4rem;color:#0f172a;margin:1.6rem 0 1rem 0;">📊 Quality Metrics Breakdown</div>', unsafe_allow_html=True)
    cats = ["completeness", "consistency", "duplicates", "outliers", "data_types", "null_values", "unique_values", "formatting"]
    labels = ["Completeness", "Consistency", "Duplicates", "Outliers", "Data Types", "Null Values", "Unique Values", "Formatting"]
    for cat, label in zip(cats, labels):
        val = scores[cat]
        c = "#22c55e" if val >= 80 else "#f59e0b" if val >= 60 else "#ef4444"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem;background:#ffffff;padding:0.75rem 1.2rem;border-radius:12px;border:1px solid #e2e8f0;">'
            f'<div style="flex:1.2;font-size:1.1rem;font-weight:700;color:#1e293b;">{label}</div>'
            f'<div style="flex:2.5;height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden;"><div style="width:{val}%;height:100%;background:{c};border-radius:5px;"></div></div>'
            f'<div style="font-size:1.15rem;font-weight:800;color:{c};min-width:60px;text-align:right;">{val}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    if problems:
        st.markdown('<div style="font-weight:800;font-size:1.35rem;color:#dc2626;margin:1.8rem 0 0.8rem 0;">⚠️ Critical Health Warnings</div>', unsafe_allow_html=True)
        for p in problems:
            st.markdown(f'<div style="background:#fef2f2;border:1.5px solid #fecaca;border-radius:12px;padding:0.75rem 1.1rem;font-size:1.05rem;font-weight:600;color:#991b1b;margin-bottom:0.5rem;">⚠️ {p}</div>', unsafe_allow_html=True)

    # ── Dynamic Data Quality Index (DQI) ───────────────────────────────────
    st.markdown('<div style="font-weight:800;font-size:1.4rem;color:#0f172a;margin:2rem 0 0.8rem 0;">📈 Dynamic Data Quality Index (DQI)</div>', unsafe_allow_html=True)
    dqi = _compute_dqi(df, col_types)
    dqi_color = "#22c55e" if dqi["DQI"] >= 80 else "#f59e0b" if dqi["DQI"] >= 60 else "#ef4444"
    st.markdown(
        f'<div style="text-align:center;padding:1.8rem;background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);border-radius:20px;margin:0.8rem 0 1.5rem 0;border:1.5px solid #86efac;">'
        f'<div style="font-size:0.95rem;color:#166534;letter-spacing:1px;font-weight:800;text-transform:uppercase;">Aggregate Data Reliability Index</div>'
        f'<div style="font-size:4rem;font-weight:900;color:{dqi_color};margin:0.3rem 0;">{dqi["DQI"]}%</div>'
        f'<div style="font-size:1.05rem;color:#15803d;font-weight:600;">Evaluates Completeness · Uniqueness · Type Validity · Outlier Ratio</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    dqi_items = [("Completeness", "🧩"), ("Uniqueness", "🔁"), ("Type Validity", "✅"), ("Outlier Ratio", "📉")]
    dqi_cols = st.columns(4)
    for ci, (k, icon) in enumerate(dqi_items):
        v = dqi[k]
        c = "#22c55e" if v >= 80 else "#f59e0b" if v >= 60 else "#ef4444"
        with dqi_cols[ci]:
            st.markdown(
                f'<div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:1.1rem;text-align:center;box-shadow:0 4px 14px rgba(15,23,42,0.04);">'
                f'<div style="font-size:1.8rem;margin-bottom:0.2rem;">{icon}</div>'
                f'<div style="font-size:0.95rem;color:#475569;font-weight:700;">{k}</div>'
                f'<div style="font-size:1.6rem;font-weight:900;color:{c};margin-top:0.2rem;">{v}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Governance Audit ───────────────────────────────────────────────────
    st.markdown('<div style="font-weight:800;font-size:1.4rem;color:#0f172a;margin:2rem 0 0.8rem 0;">🔍 Governance Audit & Vulnerabilities</div>', unsafe_allow_html=True)
    audit = _audit_trail(df, col_types)
    if audit:
        for item in audit:
            sev_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#6b7280"}.get(item["severity"], "#6b7280")
            st.markdown(
                f'<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:0.95rem 1.25rem;margin-bottom:0.65rem;box-shadow:0 2px 8px rgba(15,23,42,0.03);">'
                f'<div style="display:flex;align-items:center;gap:0.6rem;">'
                f'<span style="background:{sev_color};width:10px;height:10px;border-radius:50%;display:inline-block;"></span>'
                f'<span style="font-weight:800;font-size:1.1rem;color:#0f172a;">{item["type"]}</span>'
                f'<span style="margin-left:auto;font-size:0.8rem;background:{sev_color}1a;color:{sev_color};padding:0.2rem 0.6rem;border-radius:6px;font-weight:800;">{item["severity"].upper()} SEVERITY</span></div>'
                f'<div style="font-size:1rem;color:#334155;margin-top:0.4rem;line-height:1.6;"><code style="font-size:1rem;background:#f1f5f9;padding:0.1rem 0.4rem;border-radius:4px;">{item["column"]}</code> &mdash; {item["detail"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown('<div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:12px;padding:0.85rem 1.2rem;font-size:1.05rem;color:#166534;font-weight:700;">✅ Governance audit passed clean: No zero-variance columns, target leakage, or multicollinearity issues.</div>', unsafe_allow_html=True)

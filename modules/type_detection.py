import streamlit as st
import pandas as pd
import numpy as np
import re


def _pattern_detect_column(series):
    """Semantic/structural classification of a string column via regex."""
    s = series.dropna().astype(str)
    n = len(s)
    if n == 0:
        return "Empty", 0.0

    # Date patterns — ISO 8601
    iso_dates = s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}([T ]\d{1,2}:\d{1,2}(:\d{1,2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$", na=False).sum()
    if iso_dates / n > 0.85:
        return "Date (ISO 8601)", round(iso_dates / n * 100, 1)
    # Date patterns — slash/dash
    date_matches = s.str.match(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$", na=False).sum()
    if date_matches / n > 0.85:
        return "Date (Common Format)", round(date_matches / n * 100, 1)
    # RFC 2822 email
    email_matches = s.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False).sum()
    if email_matches / n > 0.85:
        return "Email Address", round(email_matches / n * 100, 1)
    # Geographic coordinates
    coord_matches = s.str.match(r"^-?\d{1,3}(\.\d+)?\s*[,;\s]\s*-?\d{1,3}(\.\d+)?$", na=False).sum()
    if coord_matches / n > 0.85:
        return "Geographic Coordinates (Lat, Lon)", round(coord_matches / n * 100, 1)
    # Currency strings
    currency_matches = s.str.match(r"^[\$€£¥]\s?-?[\d,]+(\.\d{1,2})?$", na=False).sum()
    if currency_matches / n > 0.85:
        return "Currency String", round(currency_matches / n * 100, 1)
    # Boolean equivalents
    bool_matches = s.str.lower().str.strip().isin(["yes", "no", "y", "n", "true", "false", "t", "f", "1", "0"]).sum()
    if bool_matches / n > 0.85:
        return "Boolean Equivalent", round(bool_matches / n * 100, 1)
    # Phone numbers
    phone_matches = s.str.match(r"^\+?[\d\s\-()]{7,20}$", na=False).sum()
    if phone_matches / n > 0.85:
        return "Phone Number", round(phone_matches / n * 100, 1)
    # Low-cardinality string → categorical
    if series.nunique() <= 10:
        return "Categorical / Ordinal", 100.0
    # Free text
    return "Free Text", 100.0


def _pattern_inference_matrix(df):
    """Per-column semantic type matrix using regex patterns."""
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            uniq = df[col].nunique()
            if uniq <= 10:
                rows.append({"Column": col, "Pandas Type": str(df[col].dtype), "Semantic Type": "Ordinal / Binned Numeric", "Evidence": f"{uniq} unique values (≤10) — treat as categorical", "Confidence": "95.0%"})
            else:
                rows.append({"Column": col, "Pandas Type": str(df[col].dtype), "Semantic Type": "Continuous Numeric", "Evidence": f"{uniq} unique values (>10) — treat as continuous", "Confidence": "95.0%"})
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            rows.append({"Column": col, "Pandas Type": "datetime64", "Semantic Type": "Date / Time", "Evidence": "Native datetime dtype", "Confidence": "100.0%"})
        elif pd.api.types.is_bool_dtype(df[col]):
            rows.append({"Column": col, "Pandas Type": "bool", "Semantic Type": "Boolean", "Evidence": "Native boolean dtype", "Confidence": "100.0%"})
        else:
            sem, conf = _pattern_detect_column(df[col])
            rows.append({"Column": col, "Pandas Type": str(df[col].dtype), "Semantic Type": sem, "Evidence": "Regex pattern match" if "(" in sem else "Low cardinality / text heuristic", "Confidence": f"{conf}%"})
    return pd.DataFrame(rows)


def _detect_type(df):
    cols_lower = [c.lower().strip() for c in df.columns]
    cols_str = " ".join(cols_lower)
    keywords = {
        "HR": ["employee", "salary", "department", "hire", "attrition", "experience", "gender", "age", "manager", "bonus", "performance", "job", "position", "tenure"],
        "Finance": ["revenue", "profit", "cost", "expense", "income", "budget", "transaction", "amount", "account", "invoice", "payment", "tax", "asset", "liability", "cash"],
        "Sales": ["sales", "customer", "product", "order", "region", "quantity", "price", "discount", "revenue", "channel", "lead", "conversion", "pipeline"],
        "Healthcare": ["patient", "diagnosis", "treatment", "doctor", "hospital", "symptom", "disease", "medication", "blood", "age", "bmi", "insurance", "claim"],
        "Education": ["student", "grade", "score", "course", "teacher", "class", "enrollment", "attendance", "subject", "semester", "exam", "assignment", "gpa"],
        "Retail": ["product", "price", "stock", "category", "brand", "supplier", "store", "inventory", "sales", "customer", "discount", "quantity"],
        "Marketing": ["campaign", "click", "impression", "conversion", "channel", "budget", "roi", "traffic", "lead", "email", "social", "engagement", "ad"],
        "Customer Support": ["ticket", "issue", "priority", "status", "agent", "response", "resolution", "satisfaction", "feedback", "complaint", "category"],
        "Manufacturing": ["product", "batch", "defect", "production", "machine", "quality", "material", "supplier", "inventory", "assembly", "waste", "downtime"],
        "Survey": ["response", "rating", "feedback", "question", "satisfaction", "demographic", "survey", "opinion", "score", "comment"],
        "E-commerce": ["order", "product", "customer", "cart", "checkout", "payment", "shipping", "review", "rating", "category", "inventory", "discount"],
    }
    scores = {}
    for dtype, kws in keywords.items():
        score = sum(2 for kw in kws if kw in cols_str)
        for col in cols_lower:
            for kw in kws:
                if kw in col:
                    score += 1
        num_cols = df.select_dtypes(include=np.number).columns.str.lower()
        num_matches = sum(1 for c in num_cols for kw in kws if kw in c)
        score += num_matches * 0.5
        scores[dtype] = score
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = round(scores[best] / max(total, 1) * 100, 1) if total > 0 else 0
    secondary = sorted(scores.items(), key=lambda x: -x[1])[1:4]
    return best, confidence, secondary, scores


def render():
    df = st.session_state.df
    detected, confidence, alternatives, all_scores = _detect_type(df)

    st.markdown('<div style="display:flex;align-items:center;gap:0.75rem;padding:1.2rem 0 0.5rem 0;"><span style="font-size:2.2rem;">🏷️</span><span style="font-weight:800;font-size:1.8rem;color:#0f172a;">Dataset Domain & Type Classifier</span></div>', unsafe_allow_html=True)

    conf_color = "#22c55e" if confidence > 70 else "#f59e0b" if confidence > 40 else "#ef4444"
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#f0f4ff 0%,#e0e7ff 100%);border-radius:24px;padding:2.2rem;text-align:center;margin:1.2rem 0;border:1.5px solid #c7d2fe;box-shadow:0 8px 30px rgba(99,102,241,0.08);">'
        f'<div style="font-size:1.05rem;color:#475569;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.4rem;">CLASSIFIED DATASET DOMAIN</div>'
        f'<div style="font-size:3.2rem;font-weight:900;color:#312e81;letter-spacing:-0.5px;">{detected}</div>'
        f'<div style="display:flex;align-items:center;justify-content:center;gap:0.5rem;margin-top:0.8rem;">'
        f'<span style="font-size:1.15rem;color:#475569;font-weight:700;">Classification Confidence:</span>'
        f'<span style="font-size:1.4rem;font-weight:900;color:{conf_color};">{confidence}%</span></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div style="font-weight:800;font-size:1.4rem;color:#0f172a;margin:1.8rem 0 0.8rem 0;">📊 Alternative Domain Likelihoods</div>', unsafe_allow_html=True)
    for dtype, score in alternatives:
        pct = round(score / max(sum(all_scores.values()), 1) * 100, 1)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.6rem;background:#ffffff;padding:0.75rem 1.2rem;border-radius:12px;border:1px solid #e2e8f0;">'
            f'<span style="flex:1.2;font-size:1.1rem;font-weight:700;color:#1e293b;">{dtype}</span>'
            f'<div style="flex:2.5;height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden;"><div style="width:{min(100, pct * 3)}%;height:100%;background:#818cf8;border-radius:5px;"></div></div>'
            f'<span style="font-size:1.1rem;font-weight:800;color:#4f46e5;min-width:60px;text-align:right;">{pct}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    recs = {
        "HR": ["Salary & Compensation Analysis", "Employee Tenure & Experience", "Departmental Headcount", "Attrition & Turnover Risk", "Gender Diversity Metrics"],
        "Finance": ["Revenue & Income Trends", "Cost & Expense Breakdown", "Budget vs Actual Variance", "Profit Margin Optimization", "Cash Flow Forecasting"],
        "Sales": ["Sales Velocity Trends", "Revenue by Region / Territory", "Product Performance Matrix", "Monthly Growth Tracking", "Pipeline Conversion Funnel"],
        "Healthcare": ["Patient Demographics", "Disease & Diagnosis Distribution", "Treatment Cost Analysis", "Hospital Readmission Rates", "Symptom Correlation"],
        "Education": ["Grade Distribution Analysis", "Course Enrollment Trends", "Student Attendance Metrics", "Exam Performance Analysis", "Graduation Rates"],
        "Retail": ["Sales by Category & Brand", "Inventory Turnover Ratio", "Customer Segmentation", "Pricing Strategy Analysis", "Stockout Risk Analysis"],
        "Marketing": ["Campaign ROI Tracking", "Channel Performance Comparison", "Conversion Funnel Drops", "Lead Quality Scorecard", "Ad Spend Allocation"],
        "Customer Support": ["Ticket Volume Trends", "First Response Time", "Agent Resolution Rate", "CSAT Score Distribution", "Root Issue Categories"],
        "Manufacturing": ["Defect Rate & Quality Metrics", "Production Yield Trends", "Machine Downtime Analysis", "Material Waste Optimization", "Supplier Compliance"],
        "Survey": ["Response Rate Breakdown", "NPS & Satisfaction Scores", "Demographic Cross-tabs", "Sentiment & Feedback Matrix", "Question Ranking"],
        "E-commerce": ["Order Velocity Trends", "Customer Lifetime Value (LTV)", "Product Bundle Analysis", "Cart Abandonment Rate", "Review & Rating Matrix"],
    }
    if detected in recs:
        st.markdown(f'<div style="font-weight:800;font-size:1.4rem;color:#0f172a;margin:2rem 0 0.8rem 0;">💡 Recommended Workflows for {detected}</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, r in enumerate(recs[detected]):
            with cols[i % 2]:
                st.markdown(
                    f'<div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:14px;padding:0.9rem 1.2rem;font-size:1.1rem;font-weight:700;color:#166534;margin-bottom:0.6rem;box-shadow:0 2px 8px rgba(34,197,94,0.04);">'
                    f'📊 {r}</div>',
                    unsafe_allow_html=True
                )

    # ── Pattern-Based Type Inference Matrix ────────────────────────────────
    st.markdown('<div style="font-weight:800;font-size:1.4rem;color:#0f172a;margin:2.2rem 0 0.4rem 0;">🧮 Multi-Engine Semantic Type Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:1.02rem;margin-bottom:0.8rem;">Regex pattern matching + cardinality heuristic classification for every dataset column</div>', unsafe_allow_html=True)
    matrix = _pattern_inference_matrix(df)
    st.dataframe(matrix, use_container_width=True, hide_index=True)

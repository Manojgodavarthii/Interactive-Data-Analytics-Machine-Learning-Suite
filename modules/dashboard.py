import streamlit as st
import pandas as pd
import hashlib
from modules.utils import read_dataset, detect_column_types, get_summary_stats, auto_insights, auto_clean_type, render_chart, view_all_button, render_charts_grid
from modules.ai_engine import analyze_dataset, important_columns


def _get_df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


@st.cache_data(show_spinner=False, ttl=600)
def _cached_ai_stats(df_hash, rows, cols, missing, duplicates, num_cols, cat_cols, date_cols, text_cols, bool_cols, missing_pct, dup_pct):
    """Cache quality stats — pass primitives instead of DataFrame to avoid serialization issues."""
    quality = "excellent"
    if missing_pct > 20 or dup_pct > 20:
        quality = "poor"
    elif missing_pct > 5 or dup_pct > 5:
        quality = "fair"
    return {
        "rows": rows, "cols": cols, "missing": missing, "duplicates": duplicates,
        "num_cols": num_cols, "cat_cols": cat_cols, "date_cols": date_cols,
        "text_cols": text_cols, "bool_cols": bool_cols,
        "missing_pct": missing_pct, "duplicate_pct": dup_pct,
        "quality": quality,
    }


@st.cache_data(show_spinner=False, ttl=600)
def _cached_important_cols(df_hash, _df):
    col_types = detect_column_types(_df)
    return important_columns(_df, col_types)


@st.cache_data(show_spinner=False, ttl=600)
def _cached_insights(df_hash, _df):
    col_types = detect_column_types(_df)
    return auto_insights(_df, col_types)


@st.cache_data(show_spinner=False, ttl=600)
def _cached_quick_charts(df_hash, _df):
    """Generate a small set of quick overview charts for the dashboard."""
    import plotly.express as px
    col_types = detect_column_types(_df)
    charts = []
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    df_s = _df.sample(min(3000, len(_df)), random_state=42) if len(_df) > 3000 else _df

    # Missing value chart
    miss = _df.isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        miss_df = miss.reset_index()
        miss_df.columns = ["Column", "Missing"]
        fig = px.bar(miss_df, x="Column", y="Missing",
                     title="⚠️ Missing Values by Column",
                     color_discrete_sequence=["#f59e0b"])
        fig.update_xaxes(tickangle=45)
        charts.append(("Missing Values", fig))

    # Histogram for each numeric column (up to 3)
    for col in num_cols[:3]:
        fig = px.histogram(df_s, x=col, marginal="box", nbins=40,
                           title=f"📊 Distribution — {col}",
                           color_discrete_sequence=["#6366f1"])
        fig.update_xaxes(tickangle=45)
        charts.append((f"{col} Dist.", fig))

    # Bar chart for each categorical column (up to 2)
    for col in cat_cols[:2]:
        top = df_s[col].value_counts().head(12).reset_index()
        top.columns = [col, "Count"]
        fig = px.bar(top, x=col, y="Count",
                     title=f"📋 Top Categories — {col}",
                     color=col, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_xaxes(tickangle=45)
        charts.append((f"{col}", fig))

    # Correlation heatmap
    if len(num_cols) >= 2:
        corr = df_s[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="🔗 Correlation Heatmap",
                        color_continuous_scale="RdBu_r", range_color=[-1, 1])
        charts.append(("Correlation", fig))

    return charts


def render():
    if st.session_state.df is None:
        _render_upload()
    else:
        _render_dashboard()


def _render_upload():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0f172a,#1e1b4b,#0f3460,#1e293b);'
        'border-radius:24px;padding:3rem 2rem;text-align:center;margin-bottom:1.8rem;'
        'box-shadow:0 20px 60px rgba(15,52,96,0.35);">'
        '<h1 style="color:white;font-size:2.6rem;font-weight:900;margin:0 0 0.8rem;letter-spacing:-1px;">'
        'Smart Dataset Analysis Platform</h1>'
        '<p style="color:rgba(255,255,255,0.7);font-size:1.05rem;max-width:540px;margin:0 auto;line-height:1.6;">'
        'Upload any dataset to unlock AI insights, automated analytics, predictive ML, and interactive dashboards.</p>'
        '<div style="color:rgba(255,255,255,0.4);font-size:0.85rem;margin-top:0.6rem;font-weight:600;">'
        'Supports CSV / Excel (.xlsx/.xls) / Up to 200 MB</div></div>',
        unsafe_allow_html=True
    )

    # ── Feature chips ──────────────────────────────────────────────────────
    features = [
        ("🤖", "AI Smart Charts"),
        ("📊", "Statistical Tests"),
        ("🔗", "Correlation Matrix"),
        ("🔮", "Forecasting"),
        ("🧹", "Auto Data Cleaning"),
        ("🚀", "Advanced Analytics"),
    ]
    cols = st.columns(len(features))
    for col, (icon, label) in zip(cols, features):
        with col:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.6);border-radius:12px;padding:0.8rem 0.5rem;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.06);border:1px solid #e4e8f4;">'
                f'<div style="font-size:1.4rem;margin-bottom:0.25rem;">{icon}</div>'
                f'<div style="font-size:0.72rem;font-weight:600;color:#374151;">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── File uploader ──────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂  Drop your file here or click to browse",
        type=["csv", "xlsx", "xls"],
        help="Supported: CSV, Excel (XLSX/XLS). Max 200MB.",
        label_visibility="visible",
    )

    if uploaded_file and st.session_state.get("_last_loaded_filename") != uploaded_file.name:
        # ── Loading state UI ───────────────────────────────────────────────
        progress_bar = st.progress(0)
        status_box = st.empty()

        try:
            status_box.markdown(
                '<div style="background:linear-gradient(135deg,#eef2ff,#f0f4ff);border-radius:14px;padding:1rem 1.4rem;border-left:4px solid #6366f1;font-size:0.9rem;color:#374151;font-weight:500;">⚡ Processing dataset...</div>',
                unsafe_allow_html=True
            )
            progress_bar.progress(30)
            df = read_dataset(uploaded_file)
            col_types = detect_column_types(df)
            progress_bar.progress(70)
            df_clean = auto_clean_type(df, col_types)

            st.session_state.df = df_clean
            st.session_state.original_df = df_clean.copy()
            st.session_state.filename = uploaded_file.name
            st.session_state._last_loaded_filename = uploaded_file.name
            st.session_state.cleaning_history = []
            st.session_state.pop("ml_train_token", None)

            # Kick off background ML training so the Predictive Modeling page is ready to use
            from modules.advanced_analytics import start_background_training
            st.session_state["ml_train_token"] = start_background_training(df_clean)

            progress_bar.progress(100)
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_box.empty()
            st.error(f"❌ Error loading file: {e}")




def _render_dashboard():
    df = st.session_state.df
    df_hash = _get_df_hash(df)
    stats, col_types = get_summary_stats(df)

    # Compute AI stats using only primitives (fast, no serialization)
    missing_pct = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if len(df) > 0 else 0
    dup_pct = round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0
    ai_stats = _cached_ai_stats(
        df_hash, stats["rows"], stats["columns"], stats["missing"], stats["duplicates"],
        stats["numeric_cols"], stats["categorical_cols"], stats["date_cols"],
        sum(1 for v in col_types.values() if v == "text"),
        sum(1 for v in col_types.values() if v == "boolean"),
        missing_pct, dup_pct,
    )

    # ── Header ────────────────────────────────────────────────────────────
    quality_color = "#22c55e" if ai_stats["quality"] == "excellent" else "#f59e0b" if ai_stats["quality"] == "fair" else "#ef4444"
    top_col1, top_col2, top_col3 = st.columns([3, 1, 1])
    with top_col1:
        st.markdown(f'<div style="font-size:1.5rem;font-weight:800;color:#1a1a2e;">📊 Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.82rem;color:#6b7280;">'
            f'{st.session_state.filename} · {stats["rows"]:,} rows · {stats["columns"]} columns · '
            f'<strong>Quality: <span style="color:{quality_color};">{ai_stats["quality"].upper()}</span></strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with top_col2:
        if st.button("📁 Upload New", use_container_width=True):
            for key in ["df", "original_df", "filename"]:
                st.session_state[key] = None
            st.session_state.cleaning_history = []
            st.session_state.pop("ml_train_token", None)
            st.session_state.page = "Dashboard"
            st.rerun()
    with top_col3:
        if st.button("🔄 Reset Data", use_container_width=True):
            if st.session_state.original_df is not None:
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state.cleaning_history = []
                st.session_state.pop("ml_train_token", None)
                from modules.advanced_analytics import start_background_training
                st.session_state["ml_train_token"] = start_background_training(st.session_state.df)
                st.rerun()

    # ── Metric Cards ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    for (label, val, icon, color), col in zip([
        ("Total Rows", f"{stats['rows']:,}", "📋", "#3b82f6"),
        ("Total Columns", stats["columns"], "🔢", "#8b5cf6"),
        ("Missing Values", f"{stats['missing']:,}", "⚠️", "#f59e0b"),
        ("Duplicate Rows", f"{stats['duplicates']:,}", "♻️", "#ef4444"),
    ], [col1, col2, col3, col4]):
        with col:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.6);border-radius:16px;padding:1.2rem 1.3rem;box-shadow:0 2px 14px rgba(0,0,0,0.05);border-left:4px solid {color};margin-bottom:0.6rem;">'
                f'<div style="font-size:1.4rem;">{icon}</div>'
                f'<div style="font-size:1.8rem;font-weight:800;color:#1a1a2e;line-height:1.2;margin:0.3rem 0;">{val}</div>'
                f'<div style="font-size:0.72rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;">{label}</div></div>',
                unsafe_allow_html=True
            )

    c1, c2, c3, c4 = st.columns(4)
    for (label, val, icon, color), col in zip([
        ("Numeric Cols", stats["numeric_cols"], "📊", "#10b981"),
        ("Categorical Cols", stats["categorical_cols"], "🏷️", "#6366f1"),
        ("Date Cols", stats["date_cols"], "📅", "#f97316"),
        ("Memory Usage", stats["memory"], "💾", "#14b8a6"),
    ], [c1, c2, c3, c4]):
        with col:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.6);border-radius:12px;padding:0.85rem 1rem;border-left:3px solid {color};margin-bottom:0.6rem;box-shadow:0 1px 6px rgba(0,0,0,0.05);">'
                f'<div style="font-size:1.1rem;">{icon}</div>'
                f'<div style="font-size:1.25rem;font-weight:700;color:#111827;margin:0.2rem 0;">{val}</div>'
                f'<div style="font-size:0.68rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.4px;">{label}</div></div>',
                unsafe_allow_html=True
            )

    # ── AI Insights ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div style='font-size:1.1rem;font-weight:700;color:#1a1a2e;margin-bottom:0.7rem;'>🤖 AI-Generated Insights</div>", unsafe_allow_html=True)
    insights = _cached_insights(df_hash, df)
    for i, ins in enumerate(insights):
        c = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"][i % 5]
        st.markdown(
            f'<div style="background:{c}0d;border-left:3px solid {c};border-radius:0 12px 12px 0;padding:0.7rem 1rem;margin-bottom:0.5rem;font-size:0.87rem;color:#111827;">'
            f'<span style="font-weight:600;color:{c};">→</span> {ins}</div>',
            unsafe_allow_html=True
        )

    # ── Important Columns ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div style='font-size:1.1rem;font-weight:700;color:#1a1a2e;margin-bottom:0.7rem;'>⭐ Important Columns (AI-Ranked)</div>", unsafe_allow_html=True)
    imp = _cached_important_cols(df_hash, df)
    if imp:
        imp_cols_ui = st.columns(min(5, len(imp)))
        for i, (col_name, score, dtype) in enumerate(imp):
            colors = ["#6366f1", "#8b5cf6", "#3b82f6", "#10b981", "#f59e0b"]
            c = colors[i % len(colors)]
            with imp_cols_ui[i % len(imp_cols_ui)]:
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#0f172a,#1e293b);'
                    f'color:white;border-radius:14px;padding:1rem 0.8rem;'
                    f'text-align:center;box-shadow:0 4px 16px rgba(0,0,0,0.15);'
                    f'border-top:3px solid {c};">'
                    f'<div style="font-size:1.8rem;font-weight:900;color:{c};line-height:1;">'
                    f'{score}</div>'
                    f'<div style="font-size:0.75rem;font-weight:600;color:rgba(255,255,255,0.85);'
                    f'margin-top:0.3rem;word-break:break-word;">{col_name}</div>'
                    f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.4);'
                    f'margin-top:0.1rem;text-transform:uppercase;">{dtype}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Quick Overview Charts ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div style='font-size:1.1rem;font-weight:700;color:#1a1a2e;margin-bottom:0.7rem;'>📈 Quick Overview Charts</div>", unsafe_allow_html=True)

    charts = _cached_quick_charts(df_hash, df)
    if charts:
        items = []
        for title, fig in charts:
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=12),
                height=360,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
            fig.update_yaxes(showgrid=True, gridcolor="#dde3f0", zerolinecolor="#e4e8f0")
            items.append((fig, f"dash_{len(items)}", title))
        render_charts_grid(items, section_key="dash_quick")

    # ── Quick Actions ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div style='font-size:1.1rem;font-weight:700;color:#1a1a2e;margin-bottom:0.8rem;'>⚡ Quick Actions</div>", unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    for (key, label, page), col in zip([
        ("a", "🧹 Start Cleaning", "Data Cleaning"),
        ("b", "📄 View Raw Data", "Raw Dataset"),
        ("c", "🎨 Generate Charts", "Visualizations"),
        ("d", "📝 Download Report", "Report Generation"),
    ], [col_a, col_b, col_c, col_d]):
        with col:
            if st.button(label, use_container_width=True, key=f"qa_{key}"):
                st.session_state.page = page
                st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import hashlib
import plotly.io as pio

def _df_hash(df):
    if df is None:
        return "none"
    return f"{id(df)}_{len(df)}_{len(df.columns)}"

@st.cache_data(show_spinner=False, ttl=300, hash_funcs={pd.DataFrame: _df_hash})
def detect_column_types(df):
    types = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            types[col] = "date"
        elif pd.api.types.is_bool_dtype(dtype):
            types[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(dtype):
            if df[col].nunique() <= 10:
                is_int = all(df[col].dropna() == df[col].dropna().astype(int))
                if is_int:
                    types[col] = "categorical"
                else:
                    types[col] = "numeric"
            else:
                types[col] = "numeric"
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            if df[col].nunique() < 20:
                types[col] = "categorical"
            else:
                types[col] = "text"
        else:
            types[col] = "text"
    return types

@st.cache_data(show_spinner=False, ttl=300, hash_funcs={pd.DataFrame: _df_hash})
def get_summary_stats(df):
    col_types = detect_column_types(df)
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_cols": sum(1 for v in col_types.values() if v == "numeric"),
        "categorical_cols": sum(1 for v in col_types.values() if v == "categorical"),
        "date_cols": sum(1 for v in col_types.values() if v == "date"),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
        "size": f"{df.shape[0]} x {df.shape[1]}"
    }
    return stats, col_types

@st.cache_data(show_spinner=False, ttl=300, hash_funcs={pd.DataFrame: _df_hash})
def auto_insights(df, col_types=None):
    col_types = col_types or detect_column_types(df)
    insights = []
    stats, _ = get_summary_stats(df)
    if stats["missing"] > 0:
        max_miss = df.isnull().sum().idxmax()
        insights.append(f"Column '{max_miss}' has the highest missing values ({df.isnull().sum().max()}).")
    unique_counts = {c: df[c].nunique() for c in df.columns}
    if unique_counts:
        top_unique = max(unique_counts, key=unique_counts.get)
        insights.append(f"Column '{top_unique}' has the most unique values ({unique_counts[top_unique]}).")
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    if num_cols:
        df_num = df[num_cols]
        max_val = df_num.max().max()
        min_val = df_num.min().min()
        max_col = df_num.max().idxmax()
        min_col = df_num.min().idxmin()
        insights.append(f"Largest numeric value: {max_val} in '{max_col}'.")
        insights.append(f"Smallest numeric value: {min_val} in '{min_col}'.")
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    if len(cat_cols) >= 1:
        insights.append(f"Categorical columns suitable for visualization: {', '.join(cat_cols[:5])}.")
    return insights

@st.cache_data(show_spinner=False, ttl=300, hash_funcs={pd.DataFrame: _df_hash})
def auto_charts(df, col_types=None):
    import plotly.express as px
    col_types = col_types or detect_column_types(df)
    charts = []
    stats, _ = get_summary_stats(df)
    if stats["missing"] > 0:
        miss_df = df.isnull().sum().reset_index()
        miss_df.columns = ["Column", "Missing Count"]
        miss_df = miss_df[miss_df["Missing Count"] > 0]
        if not miss_df.empty:
            fig = px.bar(miss_df, x="Column", y="Missing Count", title="Missing Values by Column")
            fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
            charts.append(("Missing Value Chart", fig))
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    for col in num_cols[:3]:
        n_bins = min(50, max(10, int(df[col].nunique() / 5)))
        fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="box", nbins=n_bins)
        fig.update_xaxes(tickangle=45, nticks=20, tickfont=dict(size=9))
        charts.append((f"{col} Distribution", fig))
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    for col in cat_cols[:3]:
        top = df[col].value_counts().head(10).reset_index()
        top.columns = [col, "count"]
        fig = px.bar(top, x=col, y="count", title=f"Top Categories in '{col}'")
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        charts.append((f"{col} Categories", fig))
    if len(num_cols) >= 2:
        corr_df = df[num_cols].corr().round(2)
        fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation Heatmap")
        charts.append(("Correlation", fig))
    return charts

def read_dataset(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def auto_clean_type(df, col_types):
    for col, dtype in col_types.items():
        if dtype == "date":
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
        elif dtype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def export_df(df, fmt="csv"):
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif fmt == "excel":
        buf = BytesIO()
        df.to_excel(buf, index=False)
        return buf.getvalue()
    elif fmt == "json":
        return df.to_json(orient="records").encode("utf-8")

def format_bytes(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def render_chart(fig, key, insight=None):
    """Render a Plotly chart with native mode bar (zoom/pan on hover).
    Data view accessible via an inline popover button.
    Full-screen available via the 'All Charts' page.
    """
    fig.update_layout(
        dragmode="pan",
        font=dict(size=14),
        title_font=dict(size=17),
    )
    cfg = {
        "displayModeBar": True, "displaylogo": False,
        "modeBarButtonsToRemove": ["sendDataToCloud", "lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "filename": "chart", "height": 750, "width": 1100},
        "scrollZoom": True,
    }
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key}", config=cfg)
    if insight:
        st.markdown(f'<div style="background:#f8fafc;border-left:4px solid #6366f1;padding:0.6rem 0.9rem;border-radius:0 10px 10px 0;font-size:1.02rem;color:#334155;margin-bottom:0.5rem;">💡 {insight}</div>', unsafe_allow_html=True)
    with st.popover("📊 View Raw Chart Data", help="Click to expand chart underlying data"):
        try:
            _show_chart_data(fig, key)
        except Exception as e:
            st.caption(f"Could not extract chart data: {e}")


def _show_chart_data(fig, key):
    """Extract and display data from a Plotly figure."""
    rows = []
    for trace in fig.data:
        try:
            name = trace.name or "Series"
            if _has(trace, "x") and _has(trace, "y"):
                for xi, yi in zip(list(trace.x), list(trace.y)):
                    rows.append({"Series": name, "X": xi, "Y": yi})
            elif _has(trace, "labels") and _has(trace, "values"):
                for li, vi in zip(list(trace.labels), list(trace.values)):
                    rows.append({"Series": name, "Label": li, "Value": vi})
            elif _has(trace, "z"):
                import numpy as np
                arr = np.array(trace.z)
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        rows.append({"Row": i, "Col": j, "Value": arr[i, j]})
            elif _has(trace, "y"):
                for yi in list(trace.y):
                    rows.append({"Series": name, "Value": yi})
            elif _has(trace, "x"):
                for xi in list(trace.x):
                    rows.append({"Series": name, "Value": xi})
            elif _has(trace, "text") and _has(trace, "value"):
                for ti, vi in zip(list(trace.text), list(trace.value)):
                    rows.append({"Series": name, "Label": ti, "Value": vi})
        except Exception:
            continue
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)
    else:
        st.caption("No extractable data for this chart type.")


def _has(trace, attr):
    """Check if a trace has a non-None attribute."""
    return hasattr(trace, attr) and getattr(trace, attr) is not None


def view_all_button(chart_figs, section_key):
    """Render a button at the top of a charts section to view all charts full-screen."""
    if st.button("🖥️ View All Charts in Full Screen", key=f"vall_{section_key}", type="primary"):
        st.session_state._view_all_figs = [pio.to_json(f) for f in chart_figs]
        st.session_state._view_all = True
        st.rerun()


def render_charts_grid(chart_items, section_key=None):
    """Render charts in a 2-column grid with optional view-all button.
    Each item: (fig, key) or (fig, key, insight_or_None).
    """
    figs = [item[0] for item in chart_items]
    if section_key and figs:
        view_all_button(figs, section_key)
    for i in range(0, len(chart_items), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx >= len(chart_items):
                break
            fig, key = chart_items[idx][0], chart_items[idx][1]
            insight = chart_items[idx][2] if len(chart_items[idx]) > 2 else None
            with cols[j]:
                render_chart(fig, key, insight)

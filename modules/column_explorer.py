import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types
import plotly.express as px


def render():
    df = st.session_state.df
    col_types = detect_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">&#x1F517;</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">Column Relationship Explorer</span></div>', unsafe_allow_html=True)

    all_cols = num_cols + cat_cols
    selected = st.selectbox("Select a column to explore relationships", all_cols if all_cols else df.columns.tolist())

    if selected in num_cols:
        corr = df[num_cols].corr()[selected].drop(selected).sort_values(ascending=False)
        st.markdown(f'<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:0.5rem 0;">Strong Positive Relationships</div>', unsafe_allow_html=True)
        pos = corr[corr > 0.3]
        if not pos.empty:
            for col, val in pos.items():
                st.markdown(f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:0.35rem 0.7rem;font-size:0.82rem;color:#166534;margin-bottom:0.2rem;">{col} &rarr; correlation: {val:.2f}</div>', unsafe_allow_html=True)
        else:
            st.caption("No strong positive relationships found.")

        st.markdown(f'<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:0.5rem 0;">Strong Negative Relationships</div>', unsafe_allow_html=True)
        neg = corr[corr < -0.3]
        if not neg.empty:
            for col, val in neg.items():
                st.markdown(f'<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:0.35rem 0.7rem;font-size:0.82rem;color:#991b1b;margin-bottom:0.2rem;">{col} &rarr; correlation: {val:.2f}</div>', unsafe_allow_html=True)
        else:
            st.caption("No strong negative relationships found.")

        st.markdown(f'<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:0.5rem 0;">Weak / No Relationship</div>', unsafe_allow_html=True)
        weak = corr[(corr >= -0.3) & (corr <= 0.3)]
        if not weak.empty:
            st.markdown(f'<div style="font-size:0.78rem;color:#6b7280;background:#f9fafb;border-radius:8px;padding:0.35rem 0.7rem;">{", ".join(weak.index.tolist())}</div>', unsafe_allow_html=True)

        st.markdown(f'<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:0.8rem 0 0.3rem 0;">Correlation Chart for {selected}</div>', unsafe_allow_html=True)
        fig = px.bar(corr.reset_index(), x="index", y=selected, title=f"Correlation of {selected} with other columns", color=selected, color_continuous_scale="RdYlGn", text_auto=".2f")
        fig.update_layout(height=350, xaxis_title="Column", yaxis_title="Correlation", margin=dict(l=20, r=20, t=40, b=60))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

    elif selected in cat_cols:
        st.markdown(f'<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:0.5rem 0;">Distribution of {selected}</div>', unsafe_allow_html=True)
        counts = df[selected].value_counts().head(10).reset_index()
        counts.columns = [selected, "Count"]
        fig = px.bar(counts, x=selected, y="Count", title=f"Top Categories in {selected}", text_auto=True, color=selected, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=60))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

        if num_cols:
            st.markdown(f'<div style="font-weight:700;font-size:1rem;color:#1a1a2e;margin:0.8rem 0 0.3rem 0;">{selected} grouped by numeric columns</div>', unsafe_allow_html=True)
            for num in num_cols[:2]:
                grouped = df.groupby(selected)[num].mean().sort_values(ascending=False).head(10).reset_index()
                fig2 = px.bar(grouped, x=selected, y=num, title=f"Average {num} by {selected}", color=selected)
                fig2.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=60))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
    else:
        st.info("Select a numeric or categorical column above to explore its relationships.")

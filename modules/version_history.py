import streamlit as st
import pandas as pd
import json
import datetime


def _load_history():
    if "_version_history" not in st.session_state:
        st.session_state._version_history = []
    if "_version_snapshots" not in st.session_state:
        st.session_state._version_snapshots = {}

    if not st.session_state._version_history and st.session_state.df is not None:
        st.session_state._version_history.append({
            "version": 1,
            "name": "Original Dataset",
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "rows": len(st.session_state.df),
            "cols": len(st.session_state.df.columns),
            "changes": "Initial upload",
        })
        st.session_state._version_snapshots[1] = st.session_state.df.copy()

    return st.session_state._version_history


def record_change(df, name, changes):
    """Auto-record a cleaning/transformation step to the version audit trail."""
    history = _load_history()
    ver_num = len(history) + 1
    history.append({
        "version": ver_num,
        "name": name or f"Step {ver_num}",
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows": len(df),
        "cols": len(df.columns),
        "changes": changes,
    })
    st.session_state._version_snapshots[ver_num] = df.copy()
    return ver_num


def render():
    _load_history()
    history = st.session_state._version_history
    df = st.session_state.df

    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding:1rem 0 0.3rem 0;"><span style="font-size:1.5rem;">🕒</span><span style="font-weight:800;font-size:1.4rem;color:#1a1a2e;">Dataset Version History & Audit Trail</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">Every cleaning operation and user edit is tracked automatically. Restore any previous dataset state at any time.</div>', unsafe_allow_html=True)

    if not history:
        st.info("No version history recorded yet. Load a dataset to begin tracking changes.")
        return

    st.markdown('<div style="font-weight:700;font-size:1.1rem;color:#1e293b;margin-bottom:0.6rem;">📜 Version History Log</div>', unsafe_allow_html=True)

    for i, v in enumerate(reversed(history)):
        ver_num = v["version"]
        is_latest = i == 0
        with st.container():
            border = "2px solid #6366f1" if is_latest else "1px solid #e2e8f0"
            bg = "#f8fafc" if is_latest else "#ffffff"

            col_info, col_act = st.columns([4, 1])
            with col_info:
                st.markdown(
                    f'<div style="background:{bg};border:{border};border-radius:14px;padding:0.85rem 1.1rem;margin-bottom:0.6rem;">'
                    f'<div style="display:flex;align-items:center;gap:0.6rem;">'
                    f'<span style="background:#6366f1;color:white;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:800;">V{ver_num}</span>'
                    f'<span style="font-weight:700;font-size:1rem;color:#0f172a;">{v["name"]}</span>'
                    f'{"<span style=\"background:#dcfce7;color:#166534;font-size:0.65rem;font-weight:800;padding:0.15rem 0.5rem;border-radius:6px;margin-left:0.5rem;\">ACTIVE</span>" if is_latest else ""}'
                    f'<span style="font-size:0.75rem;color:#94a3b8;margin-left:auto;">{v["date"]}</span></div>'
                    f'<div style="font-size:0.85rem;color:#475569;margin-top:0.35rem;">{v["rows"]:,} rows &times; {v["cols"]} cols &middot; {v.get("changes", "")}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_act:
                if not is_latest and ver_num in st.session_state._version_snapshots:
                    if st.button(f"↩️ Restore V{ver_num}", key=f"restore_ver_{ver_num}", use_container_width=True):
                        restored_df = st.session_state._version_snapshots[ver_num].copy()
                        st.session_state.df = restored_df
                        record_change(restored_df, f"Restored V{ver_num}", f"Reverted back to Version {ver_num}")
                        st.success(f"Restored Version {ver_num} successfully!")
                        st.rerun()

    st.markdown('<div style="font-weight:700;font-size:1.05rem;color:#1e293b;margin:1.4rem 0 0.5rem 0;">💾 Save Manual Snapshot</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        name = st.text_input("Version Name", placeholder="e.g., Pre-scaling Checkpoint, Filtered Dataset", key="manual_ver_name")
    with c2:
        if st.button("💾 Save Checkpoint", type="primary", use_container_width=True, key="btn_save_ver"):
            if df is not None:
                record_change(df, name or "Manual Checkpoint", "User created a manual snapshot")
                st.success("Version snapshot saved!")
                st.rerun()

import streamlit as st
import datetime


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "_prof_name":     "Data Analyst",
        "_prof_email":    "analyst@example.com",
        "_prof_role":     "Senior Data Analyst",
        "_prof_org":      "My Organisation",
        "_prof_bio":      "Passionate about turning raw data into actionable insights.",
        "_prof_joined":   "January 2025",
        "_prof_location": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _avatar(name: str, size: int = 80) -> str:
    initials = "".join(p[0].upper() for p in name.split()[:2]) if name else "U"
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:linear-gradient(135deg,#4f46e5,#6366f1);'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:{size//2.8:.0f}px;font-weight:800;color:white;'
        f'box-shadow:0 4px 18px rgba(99,102,241,0.4);'
        f'border:3px solid white;">{initials}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render():
    _init()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.6rem;padding:0.8rem 0 0.2rem;">'
        '<span style="font-size:1.6rem;">👤</span>'
        '<div><div style="font-size:1.35rem;font-weight:800;color:#111827;">My Profile</div>'
        '<div style="font-size:0.8rem;color:#6b7280;">View and edit your personal details</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Profile hero card ────────────────────────────────────────────────────
    hero_col, info_col = st.columns([1, 3])
    with hero_col:
        st.markdown(_avatar(st.session_state._prof_name, 96), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📷  Change Photo", key="prof_photo", use_container_width=True):
            st.info("Photo upload coming soon.")

    with info_col:
        st.markdown(
            f'<div style="padding:0.5rem 0;">'
            f'<div style="font-size:1.5rem;font-weight:800;color:#111827;">'
            f'{st.session_state._prof_name}</div>'
            f'<div style="font-size:0.9rem;color:#6366f1;font-weight:600;margin-top:0.15rem;">'
            f'{st.session_state._prof_role}</div>'
            f'<div style="font-size:0.8rem;color:#6b7280;margin-top:0.2rem;">'
            f'🏢 {st.session_state._prof_org} &nbsp;·&nbsp; '
            f'📧 {st.session_state._prof_email} &nbsp;·&nbsp; '
            f'📅 Joined {st.session_state._prof_joined}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Status badge
        status = st.session_state.get("_s_status", "Online")
        status_colors = {
            "Online": "#22c55e", "Away": "#f59e0b",
            "Busy": "#ef4444", "Offline": "#6b7280", "Focus Mode": "#8b5cf6",
        }
        status_icons = {"Online": "🟢", "Away": "🟡", "Busy": "🔴", "Offline": "⚫", "Focus Mode": "🟣"}
        sc = status_colors.get(status, "#22c55e")
        si = status_icons.get(status, "🟢")
        st.markdown(
            f'<div style="margin-top:0.6rem;">'
            f'<span style="background:{sc}20;color:{sc};padding:0.25rem 0.7rem;'
            f'border-radius:20px;font-size:0.75rem;font-weight:700;">'
            f'{si} {status}</span>'
            f'<span style="margin-left:0.5rem;font-size:0.72rem;color:#9ca3af;">'
            f'Change in Settings → Status</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_edit, tab_activity, tab_stats = st.tabs(["✏️  Edit Profile", "📋  Activity", "📊  My Stats"])

    # ── TAB: EDIT PROFILE ────────────────────────────────────────────────────
    with tab_edit:
        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;color:#6366f1;'
            'text-transform:uppercase;letter-spacing:0.8px;margin:1rem 0 0.5rem;">PERSONAL INFORMATION</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.session_state._prof_name = st.text_input(
                "Full Name", value=st.session_state._prof_name, key="pedit_name"
            )
            st.session_state._prof_email = st.text_input(
                "Email Address", value=st.session_state._prof_email, key="pedit_email"
            )
        with c2:
            st.session_state._prof_role = st.text_input(
                "Job Title / Role", value=st.session_state._prof_role, key="pedit_role"
            )
            st.session_state._prof_org = st.text_input(
                "Organisation", value=st.session_state._prof_org, key="pedit_org"
            )
        st.session_state._prof_location = st.text_input(
            "Location (optional)", value=st.session_state._prof_location,
            placeholder="e.g. Mumbai, India", key="pedit_loc"
        )
        st.session_state._prof_bio = st.text_area(
            "Bio", value=st.session_state._prof_bio,
            height=90, key="pedit_bio",
            help="A short description that others see on your profile."
        )

        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;color:#6366f1;'
            'text-transform:uppercase;letter-spacing:0.8px;margin:1.2rem 0 0.5rem;">EXPERTISE & SKILLS</div>',
            unsafe_allow_html=True,
        )
        skills_opts = ["Python", "SQL", "Machine Learning", "Data Visualisation", "Statistics",
                       "Excel", "Power BI", "Tableau", "R", "Deep Learning"]
        selected_skills = st.multiselect(
            "Your skills (select all that apply)",
            options=skills_opts,
            default=["Python", "SQL", "Statistics"],
            key="pedit_skills"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_save, col_cancel = st.columns([1, 3])
        with col_save:
            if st.button("💾  Save Profile", key="prof_save", type="primary", use_container_width=True):
                st.success("✅  Profile updated successfully!")
        with col_cancel:
            if st.button("↩️  Discard Changes", key="prof_discard", use_container_width=True):
                st.info("Changes discarded.")

    # ── TAB: ACTIVITY ────────────────────────────────────────────────────────
    with tab_activity:
        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;color:#6366f1;'
            'text-transform:uppercase;letter-spacing:0.8px;margin:1rem 0 0.7rem;">RECENT ACTIVITY</div>',
            unsafe_allow_html=True,
        )

        # Dynamic activity from session
        activities = []
        if st.session_state.get("filename"):
            activities.append(("📂", "Uploaded dataset", st.session_state.filename, "#3b82f6"))
        projects = st.session_state.get("_workspace_projects", [])
        for p in projects[-3:]:
            activities.append(("📁", "Saved project", p.get("name", ""), "#8b5cf6"))
        if st.session_state.get("cleaning_history"):
            activities.append(("🧹", "Ran data cleaning", f"{len(st.session_state.cleaning_history)} steps", "#10b981"))

        if not activities:
            st.markdown(
                '<div style="text-align:center;padding:2rem;color:#9ca3af;font-size:0.85rem;">'
                '<div style="font-size:2rem;margin-bottom:0.5rem;">📋</div>'
                'No recent activity yet. Start by uploading a dataset!</div>',
                unsafe_allow_html=True,
            )
        else:
            for icon, action, detail, color in activities:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.8rem;'
                    f'padding:0.65rem 0.8rem;border-radius:10px;'
                    f'background:#f8faff;border-left:3px solid {color};margin-bottom:0.5rem;">'
                    f'<span style="font-size:1.1rem;">{icon}</span>'
                    f'<div><div style="font-weight:600;font-size:0.84rem;color:#111827;">{action}</div>'
                    f'<div style="font-size:0.72rem;color:#9ca3af;">{detail}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── TAB: STATS ───────────────────────────────────────────────────────────
    with tab_stats:
        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;color:#6366f1;'
            'text-transform:uppercase;letter-spacing:0.8px;margin:1rem 0 0.7rem;">YOUR ANALYTICS OVERVIEW</div>',
            unsafe_allow_html=True,
        )
        n_proj = len(st.session_state.get("_workspace_projects", []))
        n_clean = len(st.session_state.get("cleaning_history", []))
        has_df = st.session_state.get("df") is not None

        stat_items = [
            ("📁", "Saved Projects", str(n_proj),      "#6366f1"),
            ("📂", "Datasets Loaded", "1" if has_df else "0", "#3b82f6"),
            ("🧹", "Cleaning Steps",  str(n_clean),    "#10b981"),
            ("⭐", "Skills Listed",   "6",              "#f59e0b"),
        ]
        s1, s2, s3, s4 = st.columns(4)
        for (icon, label, val, color), col in zip(stat_items, [s1, s2, s3, s4]):
            with col:
                st.markdown(
                    f'<div style="background:#fff;border-radius:14px;padding:1.1rem;'
                    f'border:1px solid #e8ecf4;text-align:center;'
                    f'box-shadow:0 2px 10px rgba(0,0,0,0.04);">'
                    f'<div style="font-size:1.5rem;">{icon}</div>'
                    f'<div style="font-size:1.6rem;font-weight:800;color:{color};margin:0.2rem 0;">{val}</div>'
                    f'<div style="font-size:0.68rem;font-weight:600;color:#9ca3af;'
                    f'text-transform:uppercase;letter-spacing:0.4px;">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        # Quick link to settings
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#eef2ff;border-radius:12px;padding:1rem 1.2rem;">'
            '<div style="font-weight:600;font-size:0.85rem;color:#3730a3;margin-bottom:0.3rem;">⚙️  Manage your preferences</div>'
            '<div style="font-size:0.78rem;color:#6b7280;">Go to Settings to update notifications, security, appearance and more.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("→  Open Settings", key="prof_goto_settings"):
            st.session_state.page = "Settings"
            st.rerun()

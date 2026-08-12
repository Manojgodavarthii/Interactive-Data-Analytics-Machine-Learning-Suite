import streamlit as st


# ── Injected CSS ──────────────────────────────────────────────────────────────
SETTINGS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── SETTINGS NAV BUTTONS — remove ALL default streamlit button background ── */
div[data-testid="stSidebar"] .stButton > button,
.settings-left .stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #4b5563 !important;
}

/* Kill the ugly gray [data-testid] badge highlight on ALL settings nav items */
.settings-nav-btn .stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #4b5563 !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    padding: 0.5rem 0.8rem !important;
    border-radius: 10px !important;
    width: 100% !important;
    text-align: left !important;
    display: flex !important;
    align-items: center !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    margin-bottom: 2px !important;
}
.settings-nav-btn .stButton > button:hover {
    background: #f0f4ff !important;
    color: #4f46e5 !important;
}
.settings-nav-btn .stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Active nav item is rendered as HTML, not a button */
.snav-active {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.5rem 0.8rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white !important;
    font-weight: 700;
    font-size: 0.86rem;
    box-shadow: 0 4px 14px rgba(99,102,241,0.28);
    margin-bottom: 2px;
    cursor: default;
}

/* Settings panel card */
.settings-card {
    background: #ffffff;
    border-radius: 18px;
    border: 1px solid #e8ecf4;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    padding: 2rem 2.2rem;
    min-height: 520px;
}

/* Section label */
.s-section-label {
    font-size: 0.65rem;
    font-weight: 800;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 1.5px solid #eef0f8;
}

/* Info pill */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #eef2ff;
    color: #3730a3;
    border-radius: 8px;
    padding: 0.55rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* Strength bar container */
.strength-wrap { margin: 0.5rem 0 0.8rem; }

/* Status card */
.status-card {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 0.9rem;
    border-radius: 11px;
    margin-bottom: 0.4rem;
    transition: background 0.12s;
}
.status-card:hover { background: #f5f7ff; }
.status-card.active-status {
    background: #f5f7ff;
}
</style>
"""

# ── Nav items ─────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("general",       "⚙️",  "General"),
    ("notifications", "🔔",  "Notifications"),
    ("appearance",    "🎨",  "Appearance"),
    ("privacy",       "🔒",  "Privacy"),
    ("security",      "🛡️",  "Security"),
    ("skills",        "🧠",  "Skills"),
    ("meeting",       "📅",  "Meeting"),
    ("status",        "🟢",  "Status"),
]


# ── State init ────────────────────────────────────────────────────────────────
def _init():
    if "_settings_tab" not in st.session_state:
        st.session_state._settings_tab = "general"
    if "_settings_saved" not in st.session_state:
        st.session_state._settings_saved = False

    defaults = {
        "_s_language":      "English",
        "_s_timezone":      "UTC+05:30 (India)",
        "_s_date_format":   "DD/MM/YYYY",
        "_s_autosave":      True,
        "_s_notif_email":   True,
        "_s_notif_push":    False,
        "_s_notif_inapp":   True,
        "_s_notif_weekly":  True,
        "_s_notif_alerts":  True,
        "_s_theme":         "Light",
        "_s_fontsize":      "Medium",
        "_s_density":       "Comfortable",
        "_s_profile_visible": True,
        "_s_data_sharing":  False,
        "_s_analytics":     True,
        "_s_status":        "Online",
        "_s_meeting_notif": 15,
        "_s_meeting_cal":   "None",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Panel header ──────────────────────────────────────────────────────────────
def _panel_header(title, subtitle, icon):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">'
        f'<div style="width:46px;height:46px;border-radius:12px;'
        f'background:linear-gradient(135deg,#eef2ff,#e0e7ff);'
        f'display:flex;align-items:center;justify-content:center;font-size:1.4rem;">{icon}</div>'
        f'<div><div style="font-size:1.2rem;font-weight:800;color:#111827;line-height:1.2;">{title}</div>'
        f'<div style="font-size:0.78rem;color:#9ca3af;margin-top:0.1rem;">{subtitle}</div></div></div>'
        f'<hr style="margin:0.8rem 0 1.2rem!important;">',
        unsafe_allow_html=True,
    )


def _group(label):
    st.markdown(
        f'<div class="s-section-label">{label}</div>',
        unsafe_allow_html=True,
    )


def _save_btn(tab):
    st.markdown("<br>", unsafe_allow_html=True)
    col_s, col_x = st.columns([1, 4])
    with col_s:
        if st.button("💾  Save Changes", key=f"save_{tab}", type="primary", use_container_width=True):
            st.session_state._settings_saved = True
            st.rerun()
    if st.session_state._settings_saved:
        st.success("✅  Settings saved successfully!")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: GENERAL
# ─────────────────────────────────────────────────────────────────────────────
def _tab_general():
    _panel_header("General Settings", "Language, timezone and application behaviour", "⚙️")

    _group("LANGUAGE & REGION")
    c1, c2 = st.columns(2)
    with c1:
        langs = ["English", "Hindi", "French", "Spanish", "German", "Arabic", "Chinese"]
        idx = langs.index(st.session_state._s_language) if st.session_state._s_language in langs else 0
        st.session_state._s_language = st.selectbox("🌐 Language", langs, index=idx, key="gen_lang")
    with c2:
        tzs = ["UTC+00:00 (London)", "UTC+05:30 (India)", "UTC-05:00 (EST)", "UTC-08:00 (PST)", "UTC+08:00 (SGT)", "UTC+09:00 (JST)"]
        idx2 = tzs.index(st.session_state._s_timezone) if st.session_state._s_timezone in tzs else 1
        st.session_state._s_timezone = st.selectbox("🕐 Timezone", tzs, index=idx2, key="gen_tz")

    _group("DATE & TIME FORMAT")
    c3, c4 = st.columns(2)
    with c3:
        fmts = ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD", "D MMMM YYYY"]
        idx3 = fmts.index(st.session_state._s_date_format) if st.session_state._s_date_format in fmts else 0
        st.session_state._s_date_format = st.selectbox("📅 Date Format", fmts, index=idx3, key="gen_fmt")
    with c4:
        st.selectbox("🕛 Time Format", ["12-hour (AM/PM)", "24-hour"], key="gen_timefmt")

    _group("APPLICATION BEHAVIOUR")
    st.session_state._s_autosave = st.toggle(
        "Auto-save work every 5 minutes",
        value=st.session_state._s_autosave, key="gen_autosave"
    )
    st.toggle("Show onboarding tips for new features", value=True, key="gen_tips")
    st.toggle("Send anonymous usage data to improve the app", value=False, key="gen_anon")
    st.toggle("Show keyboard shortcut hints in tooltips", value=True, key="gen_kbhints")

    _group("STARTUP")
    st.selectbox("Open on startup", ["Dashboard", "Last visited page", "Workspace"], key="gen_startup")

    _save_btn("general")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
def _tab_notifications():
    _panel_header("Notifications", "Control how and when you receive alerts", "🔔")

    _group("NOTIFICATION CHANNELS")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state._s_notif_email = st.toggle("📧 Email", value=st.session_state._s_notif_email, key="notif_em")
    with c2:
        st.session_state._s_notif_push = st.toggle("📱 Push", value=st.session_state._s_notif_push, key="notif_pu")
    with c3:
        st.session_state._s_notif_inapp = st.toggle("🔔 In-App", value=st.session_state._s_notif_inapp, key="notif_ia")

    _group("EVENT TYPES")
    col_a, col_b = st.columns(2)
    with col_a:
        st.toggle("Dataset upload complete", value=True, key="notif_upload")
        st.toggle("Analysis finished", value=True, key="notif_analysis")
        st.toggle("Report generated", value=True, key="notif_report")
    with col_b:
        st.toggle("Scheduled job alerts", value=True, key="notif_job")
        st.toggle("Collaboration updates", value=False, key="notif_collab")
        st.toggle("System maintenance", value=True, key="notif_maint")

    _group("DIGEST & FREQUENCY")
    st.session_state._s_notif_weekly = st.toggle(
        "Weekly summary digest every Monday 9 AM",
        value=st.session_state._s_notif_weekly, key="notif_weekly"
    )
    st.toggle("Critical error alerts (always on)", value=True, key="notif_alerts", disabled=True)
    st.selectbox("Notification quiet hours", ["None", "10 PM – 7 AM", "11 PM – 8 AM", "Custom"], key="notif_quiet")
    st.selectbox("Notification sound", ["Default", "Soft chime", "Silent"], key="notif_sound")

    _save_btn("notifications")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: APPEARANCE
# ─────────────────────────────────────────────────────────────────────────────
def _tab_appearance():
    _panel_header("Appearance", "Customise how the application looks and feels", "🎨")

    _group("THEME SELECTION")
    theme_cols = st.columns(3)
    themes = [
        ("☀️", "Light",  "#ffffff",  "#e8ecf4"),
        ("🌙", "Dark",   "#0f172a",  "#1e293b"),
        ("🖥️", "System", "#f1f5f9",  "#dde4f0"),
    ]
    for i, (icon, name, bg, border) in enumerate(themes):
        with theme_cols[i]:
            active = st.session_state._s_theme == name
            border_style = "2.5px solid #6366f1" if active else f"1px solid {border}"
            badge = '<div style="font-size:0.62rem;color:#6366f1;font-weight:700;margin-top:0.3rem;">✓ Active</div>' if active else ""
            st.markdown(
                f'<div style="background:{bg};border:{border_style};border-radius:14px;'
                f'padding:1rem 0.8rem;text-align:center;transition:all 0.2s;">'
                f'<div style="font-size:1.9rem;">{icon}</div>'
                f'<div style="font-size:0.82rem;font-weight:{"700" if active else "500"};'
                f'color:{"#4f46e5" if active else "#6b7280"};margin-top:0.3rem;">{name}</div>'
                f'{badge}</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Select {name}", key=f"theme_{name}", use_container_width=True):
                st.session_state._s_theme = name
                st.rerun()

    _group("TYPOGRAPHY & DENSITY")
    c1, c2 = st.columns(2)
    with c1:
        sizes = ["Small", "Medium", "Large", "Extra Large"]
        st.session_state._s_fontsize = st.select_slider(
            "Font Size", options=sizes, value=st.session_state._s_fontsize, key="app_font"
        )
    with c2:
        densities = ["Compact", "Comfortable", "Spacious"]
        st.session_state._s_density = st.select_slider(
            "Layout Density", options=densities, value=st.session_state._s_density, key="app_density"
        )

    _group("SIDEBAR PREFERENCES")
    st.toggle("Always show sidebar navigation labels", value=True, key="app_sidebar_labels")
    st.toggle("Collapse sidebar on small screens", value=True, key="app_sidebar_collapse")
    st.selectbox("Sidebar width", ["Narrow (220px)", "Standard (260px)", "Wide (300px)"], key="app_sidebar_width")

    _group("COLOUR ACCENT")
    accent_opts = ["Indigo (Default)", "Blue", "Purple", "Teal", "Rose"]
    st.selectbox("Primary accent colour", accent_opts, key="app_accent")

    _save_btn("appearance")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: PRIVACY
# ─────────────────────────────────────────────────────────────────────────────
def _tab_privacy():
    _panel_header("Privacy", "Control what data is shared and who can see your profile", "🔒")

    _group("PROFILE VISIBILITY")
    vis_opts = ["Public", "Organisation only", "Private"]
    st.selectbox("Who can see your profile?", vis_opts, index=1, key="priv_vis")
    st.toggle("Show your activity status to others", value=True, key="priv_activity")
    st.toggle("Allow others to find you by email", value=False, key="priv_email_search")
    st.toggle("Show your last seen time", value=True, key="priv_lastseen")

    _group("DATA HANDLING")
    st.toggle("Allow app to store analysis history", value=True, key="priv_history")
    st.session_state._s_data_sharing = st.toggle(
        "Share anonymised dataset statistics for research",
        value=st.session_state._s_data_sharing, key="priv_share"
    )
    st.toggle("Enable personalised AI recommendations based on usage", value=True, key="priv_ai_rec")
    st.toggle("Allow cookies for session persistence", value=True, key="priv_cookies")

    _group("DATA DELETION")
    st.warning("⚠️  These actions are permanent and cannot be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️  Clear Analysis History", key="priv_clr_hist", use_container_width=True):
            st.session_state.cleaning_history = []
            st.success("Analysis history cleared.")
    with c2:
        if st.button("🗑️  Clear All Saved Projects", key="priv_clr_proj", use_container_width=True):
            st.session_state._workspace_projects = []
            st.success("All projects cleared.")

    _save_btn("privacy")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: SECURITY  — New Password + Confirm New Password only
# ─────────────────────────────────────────────────────────────────────────────
def _tab_security():
    _panel_header("Security", "Manage your password and account access", "🛡️")

    _group("CHANGE PASSWORD")
    st.markdown(
        '<div class="info-pill">🔐 Set a new password below. Confirm it to save.</div>',
        unsafe_allow_html=True,
    )

    new_pw = st.text_input(
        "New Password", type="password", key="sec_new_pw",
        placeholder="At least 8 characters"
    )
    con_pw = st.text_input(
        "Confirm New Password", type="password", key="sec_con_pw",
        placeholder="Re-enter new password"
    )

    # Live strength indicator
    if new_pw:
        strength = 0
        tips = []
        if len(new_pw) >= 8:                              strength += 1
        else:                                              tips.append("Use at least 8 characters")
        if any(c.isupper() for c in new_pw):              strength += 1
        else:                                              tips.append("Add an uppercase letter")
        if any(c.isdigit() for c in new_pw):              strength += 1
        else:                                              tips.append("Add a number")
        if any(c in "!@#$%^&*()_+-=" for c in new_pw):   strength += 1
        else:                                              tips.append("Add a special character (!@#…)")

        labels = ["Weak", "Fair", "Good", "Strong"]
        colors = ["#ef4444", "#f59e0b", "#3b82f6", "#22c55e"]
        bar_ws = [20, 50, 75, 100]
        s      = max(0, strength - 1)
        lbl, clr, bw = labels[s], colors[s], bar_ws[s]

        st.markdown(
            f'<div class="strength-wrap">'
            f'<div style="font-size:0.72rem;color:#6b7280;margin-bottom:0.3rem;">Password strength</div>'
            f'<div style="background:#e5e7eb;border-radius:4px;height:7px;">'
            f'<div style="background:{clr};width:{bw}%;height:7px;border-radius:4px;transition:width 0.3s;"></div>'
            f'</div>'
            f'<div style="font-size:0.72rem;font-weight:700;color:{clr};margin-top:0.25rem;">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if tips:
            with st.expander("💡 Password tips", expanded=False):
                for t in tips:
                    st.caption(f"• {t}")

    # Match check
    if con_pw and new_pw and con_pw != new_pw:
        st.error("❌  New password and confirm password do not match.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔐  Update Password", key="sec_update_pw", type="primary"):
        if not new_pw:
            st.error("Please enter a new password.")
        elif len(new_pw) < 8:
            st.error("Password must be at least 8 characters.")
        elif new_pw != con_pw:
            st.error("❌  New password and confirm password do not match.")
        else:
            st.success("✅  Password updated successfully!")

    _group("SESSION & ACCESS")
    st.toggle("Stay signed in on this device", value=True, key="sec_stay")
    st.toggle("Enable two-factor authentication (2FA)", value=False, key="sec_2fa")
    st.toggle("Notify me on new sign-in from another device", value=True, key="sec_notify_signin")

    _group("ACTIVE SESSIONS")
    st.markdown(
        '<div style="background:#f8faff;border-radius:12px;padding:0.85rem 1.1rem;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        '<div><div style="font-weight:700;font-size:0.85rem;color:#111827;">💻 This device</div>'
        '<div style="font-size:0.72rem;color:#9ca3af;margin-top:0.1rem;">Windows · Chrome · Current session</div></div>'
        '<span style="font-size:0.7rem;background:#dcfce7;color:#166534;'
        'padding:0.22rem 0.6rem;border-radius:20px;font-weight:700;">Active</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪  Sign out all other devices", key="sec_signout_all"):
        st.success("All other sessions signed out.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: SKILLS
# ─────────────────────────────────────────────────────────────────────────────
def _tab_skills():
    _panel_header("Skills & Shortcuts", "Manage AI analysis shortcuts and skill preferences", "🧠")

    _group("AI ANALYSIS SKILLS")
    skill_list = [
        ("📊", "Auto Statistical Summary",  "Automatically run descriptive statistics on upload",  True),
        ("🔗", "Correlation Detection",      "Detect and highlight strongly correlated columns",    True),
        ("🧹", "Smart Data Cleaning",        "Suggest cleaning actions based on data quality",      True),
        ("🔮", "Forecasting Assistant",      "Enable time-series forecasting for date columns",    False),
        ("🏷️", "Type Auto-Detection",        "Auto-detect column data types on upload",            True),
        ("🤖", "AI Narrative Insights",      "Generate natural-language insights from data",       True),
        ("📉", "Anomaly Detection",          "Flag statistical outliers automatically",            False),
        ("🗂️", "Smart Column Grouping",      "Group related columns for faster analysis",          True),
    ]
    for icon, name, desc, default_on in skill_list:
        c1, c2 = st.columns([6, 1])
        with c1:
            st.markdown(
                f'<div style="display:flex;gap:0.65rem;align-items:flex-start;padding:0.45rem 0;">'
                f'<span style="font-size:1.1rem;margin-top:0.1rem;">{icon}</span>'
                f'<div><div style="font-weight:600;font-size:0.84rem;color:#111827;">{name}</div>'
                f'<div style="font-size:0.72rem;color:#9ca3af;">{desc}</div></div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.toggle("", value=default_on, key=f"sk_{name[:14].replace(' ','_')}", label_visibility="collapsed")

    _group("KEYBOARD SHORTCUTS")
    shortcuts = [
        ("Ctrl + U", "Upload new dataset"),
        ("Ctrl + S", "Save current project"),
        ("Ctrl + R", "Reset data to original"),
        ("Ctrl + E", "Export report"),
        ("Ctrl + /", "Open AI chat"),
    ]
    for keys, action in shortcuts:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:0.45rem 0;border-bottom:1px solid #f0f2f8;">'
            f'<span style="font-size:0.83rem;color:#374151;">{action}</span>'
            f'<code style="background:#f3f4f6;padding:0.18rem 0.5rem;border-radius:6px;'
            f'font-size:0.75rem;color:#4f46e5;font-weight:700;border:1px solid #e5e7eb;">{keys}</code></div>',
            unsafe_allow_html=True,
        )

    _save_btn("skills")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: MEETING
# ─────────────────────────────────────────────────────────────────────────────
def _tab_meeting():
    _panel_header("Meeting & Calendar", "Configure meeting reminders and calendar integrations", "📅")

    _group("CALENDAR INTEGRATION")
    cal_choice = st.selectbox(
        "Connect calendar",
        ["None", "Google Calendar", "Outlook / Microsoft 365", "Apple Calendar"],
        key="mtg_cal"
    )
    if cal_choice != "None":
        st.markdown(
            f'<div class="info-pill">✅ {cal_choice} will sync meeting dates with your project plan automatically.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-pill">💡 Select a calendar above to enable automatic sync.</div>',
            unsafe_allow_html=True,
        )

    _group("MEETING REMINDERS")
    mins = [5, 10, 15, 30, 60]
    st.session_state._s_meeting_notif = st.select_slider(
        "Remind me before meeting",
        options=mins,
        value=st.session_state._s_meeting_notif,
        format_func=lambda x: f"{x} min",
        key="mtg_remind"
    )
    st.toggle("Add meeting notes automatically after call", value=False, key="mtg_notes")
    st.toggle("Block calendar during active analysis sessions", value=False, key="mtg_block")
    st.toggle("Send meeting summary email after each call", value=True, key="mtg_summary_email")

    _group("DEFAULT MEETING SETTINGS")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Default meeting duration", ["15 min", "30 min", "45 min", "60 min", "90 min"], index=1, key="mtg_dur")
    with c2:
        st.selectbox("Default conferencing tool", ["Zoom", "Google Meet", "MS Teams", "Webex", "None"], key="mtg_tool")
    st.toggle("Automatically generate agenda from project plan", value=True, key="mtg_agenda")

    _save_btn("meeting")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: STATUS
# ─────────────────────────────────────────────────────────────────────────────
def _tab_status():
    _panel_header("Status", "Set your availability and working status", "🟢")

    _group("CURRENT STATUS")
    statuses = [
        ("🟢", "Online",      "Available and active",         "#22c55e"),
        ("🟡", "Away",        "Away from keyboard",            "#f59e0b"),
        ("🔴", "Busy",        "In a meeting or focused work", "#ef4444"),
        ("⚫", "Offline",     "Appear offline to others",      "#6b7280"),
        ("🟣", "Focus Mode",  "Do not disturb — deep work",   "#8b5cf6"),
    ]
    for icon, name, desc, color in statuses:
        active = st.session_state._s_status == name
        col_a, col_b = st.columns([5, 1])
        with col_a:
            bg     = f"{color}12" if active else "transparent"
            border = f"1.5px solid {color}" if active else "1.5px solid transparent"
            badge  = (
                f'<span style="margin-left:auto;font-size:0.62rem;background:{color}25;'
                f'color:{color};padding:0.12rem 0.45rem;border-radius:20px;font-weight:700;">Active</span>'
            ) if active else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.75rem;'
                f'padding:0.55rem 0.9rem;border-radius:11px;'
                f'background:{bg};border:{border};transition:all 0.15s;">'
                f'<span style="font-size:1.25rem;">{icon}</span>'
                f'<div style="flex:1;">'
                f'<div style="font-weight:{"700" if active else "500"};font-size:0.85rem;'
                f'color:{"#111827" if active else "#374151"};">{name}</div>'
                f'<div style="font-size:0.72rem;color:#9ca3af;">{desc}</div></div>'
                f'{badge}</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            if not active:
                if st.button("Set", key=f"st_{name}", use_container_width=True):
                    st.session_state._s_status = name
                    st.rerun()
            else:
                st.markdown(
                    f'<div style="height:38px;display:flex;align-items:center;justify-content:center;'
                    f'font-size:0.7rem;color:{color};font-weight:700;">✓</div>',
                    unsafe_allow_html=True,
                )

    _group("AUTO STATUS RULES")
    st.toggle("Automatically set 'Away' after 10 min inactivity", value=True, key="stat_auto_away")
    st.toggle("Automatically restore 'Online' when active again", value=True, key="stat_auto_online")
    st.selectbox(
        "Working hours (show Available only during)",
        ["Always", "9 AM – 6 PM", "8 AM – 5 PM", "10 AM – 7 PM", "Custom"],
        key="stat_hours"
    )

    _group("CUSTOM STATUS MESSAGE")
    st.text_input("Status message", placeholder="e.g. Working on Q4 analysis...", key="stat_msg")
    st.selectbox("Clear message after", ["Never", "30 minutes", "1 hour", "4 hours", "End of day"], key="stat_clear")

    _save_btn("status")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render():
    _init()
    st.markdown(SETTINGS_CSS, unsafe_allow_html=True)

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.7rem;padding:0.8rem 0 0.2rem;">'
        '<div style="width:48px;height:48px;border-radius:14px;'
        'background:linear-gradient(135deg,#4f46e5,#6366f1);'
        'display:flex;align-items:center;justify-content:center;font-size:1.4rem;">'
        '⚙️</div>'
        '<div><div style="font-size:1.4rem;font-weight:800;color:#111827;">Settings</div>'
        '<div style="font-size:0.8rem;color:#9ca3af;">Manage your account preferences and app configuration</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Layout: left nav | right panel ───────────────────────────────────────
    left, right = st.columns([1, 3.2])

    with left:
        st.markdown(
            '<div style="font-size:0.6rem;font-weight:800;color:#9ca3af;'
            'text-transform:uppercase;letter-spacing:1.2px;margin-bottom:0.6rem;">'
            'SETTINGS MENU</div>',
            unsafe_allow_html=True,
        )
        for key, icon, label in NAV_ITEMS:
            active = st.session_state._settings_tab == key
            if active:
                st.markdown(
                    f'<div class="snav-active">'
                    f'<span style="font-size:1rem;">{icon}</span>'
                    f'<span>{label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="settings-nav-btn">', unsafe_allow_html=True)
                if st.button(f"{icon}  {label}", key=f"snav_{key}", use_container_width=True):
                    st.session_state._settings_tab = key
                    st.session_state._settings_saved = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # White card container
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        tab = st.session_state._settings_tab
        if tab == "general":         _tab_general()
        elif tab == "notifications":  _tab_notifications()
        elif tab == "appearance":     _tab_appearance()
        elif tab == "privacy":        _tab_privacy()
        elif tab == "security":       _tab_security()
        elif tab == "skills":         _tab_skills()
        elif tab == "meeting":        _tab_meeting()
        elif tab == "status":         _tab_status()
        st.markdown('</div>', unsafe_allow_html=True)

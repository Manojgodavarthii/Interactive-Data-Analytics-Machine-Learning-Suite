import streamlit as st
import pandas as pd
import datetime
import re
import calendar as cal_lib
import base64
import io


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS — Project Store
# ─────────────────────────────────────────────────────────────────────────────
def _load_projects():
    if "_workspace_projects" not in st.session_state:
        st.session_state._workspace_projects = []
    return st.session_state._workspace_projects


def _get_project(name):
    for p in st.session_state._workspace_projects:
        if p["name"] == name:
            return p
    return None


def _ensure_keys(proj):
    """Ensure all feature keys exist on legacy project dicts."""
    proj.setdefault("calendar_events", {})   # {date_str: [task_str, …]}
    proj.setdefault("attachments", [])        # [{name, size, preview}]
    proj.setdefault("current_work", [])       # [task_str, …]
    proj.setdefault("er_images", [])          # [{name, data_b64, mime}]
    return proj


# ─────────────────────────────────────────────────────────────────────────────
#  DATE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_DATE_PATTERNS = [
    (r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b",
     lambda m: _try_date(int(m[0]), int(m[1]), int(m[2]))),
    (r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b",
     lambda m: _try_date(int(m[2]), int(m[1]), int(m[0]))),
    (r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b",
     lambda m: _parse_named(m[2], m[1], m[0])),
    (r"\b([A-Za-z]+)\s+(\d{1,2})[,\s]+(\d{4})\b",
     lambda m: _parse_named(m[2], m[0], m[1])),
]


def _try_date(y, m, d):
    try:
        return datetime.date(y, m, d)
    except Exception:
        return None


def _parse_named(year_str, month_str, day_str):
    mo = _MONTH_MAP.get(str(month_str).lower().strip())
    if mo:
        return _try_date(int(year_str), mo, int(day_str))
    return None


def extract_dates_from_text(text: str) -> list:
    found = set()
    for pattern, parser in _DATE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            d = parser(m.groups())
            if d:
                found.add(d)
    return sorted(found)


def read_uploaded_text(file) -> str:
    name = file.name.lower()
    raw = file.read()

    if name.endswith(".pdf"):
        try:
            try:
                import pypdf as pypdf_lib
                reader = pypdf_lib.PdfReader(io.BytesIO(raw))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                pass
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(raw))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                pass
        except Exception:
            pass
        return raw.decode("latin-1", errors="ignore")

    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            pass
        return raw.decode("utf-8", errors="ignore")

    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE_CSS = """
<style>
/* ── Calendar Grid ── */
.cal-grid { display:grid; grid-template-columns:repeat(7,1fr); gap:3px; }
.cal-header-cell {
    text-align:center; font-size:0.62rem; font-weight:700;
    color:#9ca3af; text-transform:uppercase; letter-spacing:0.5px;
    padding:0.3rem 0;
}
.cal-day {
    min-height:54px; border-radius:10px; padding:4px 6px;
    background:#f8faff; border:1px solid #eef0f8; position:relative;
    cursor:pointer; transition:all 0.15s;
}
.cal-day:hover { background:#eef2ff; border-color:#a5b4fc; }
.cal-day.today { border:2px solid #6366f1; background:#eef2ff; }
.cal-day.has-events { background:#f0f4ff; }
.cal-day.other-month { opacity:0.35; cursor:default; }
.cal-day-num {
    font-size:0.78rem; font-weight:600; color:#374151; line-height:1;
}
.cal-day.today .cal-day-num { color:#4f46e5; font-weight:800; }
.event-dot {
    display:inline-block; width:6px; height:6px;
    background:#6366f1; border-radius:50%; margin:1px;
}
/* ── Project card ── */
.proj-card {
    background:white; border-radius:16px; border:1px solid #e4e8f0;
    box-shadow:0 2px 12px rgba(0,0,0,0.05); padding:1rem 1.2rem;
    margin-bottom:0.8rem; transition:box-shadow 0.15s;
}
.proj-card:hover { box-shadow:0 4px 20px rgba(99,102,241,0.12); }
/* ── Section label ── */
.ws-section {
    font-size:0.65rem; font-weight:800; color:#6366f1;
    text-transform:uppercase; letter-spacing:1px;
    margin:1.4rem 0 0.6rem; padding-bottom:0.3rem;
    border-bottom:1.5px solid #eef0f8;
}
/* ── Attachment pill ── */
.attach-pill {
    display:flex; align-items:center; gap:0.6rem;
    background:#f8faff; border:1px solid #e8ecf4; border-radius:10px;
    padding:0.5rem 0.8rem; margin-bottom:0.4rem; font-size:0.82rem;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE CALENDAR
# ─────────────────────────────────────────────────────────────────────────────
def _calendar_section(proj):
    events = proj["calendar_events"]   # {date_str → [task, …]}

    st.markdown('<div class="ws-section">📅 PLAN DATES — INTERACTIVE CALENDAR</div>', unsafe_allow_html=True)

    # ── Month navigation ─────────────────────────────────────────────────────
    today = datetime.date.today()
    if "_cal_year" not in st.session_state:
        st.session_state._cal_year  = today.year
        st.session_state._cal_month = today.month

    nav_l, nav_c, nav_r, nav_today = st.columns([1, 4, 1, 1])
    with nav_l:
        if st.button("◀", key="cal_prev", use_container_width=True):
            m = st.session_state._cal_month - 1
            if m < 1:
                m = 12; st.session_state._cal_year -= 1
            st.session_state._cal_month = m
            st.rerun()
    with nav_c:
        mn = cal_lib.month_name[st.session_state._cal_month]
        st.markdown(
            f'<div style="text-align:center;font-size:1rem;font-weight:800;color:#111827;'
            f'padding:0.45rem 0;">{mn} {st.session_state._cal_year}</div>',
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button("▶", key="cal_next", use_container_width=True):
            m = st.session_state._cal_month + 1
            if m > 12:
                m = 1; st.session_state._cal_year += 1
            st.session_state._cal_month = m
            st.rerun()
    with nav_today:
        if st.button("Today", key="cal_today", use_container_width=True):
            st.session_state._cal_year  = today.year
            st.session_state._cal_month = today.month
            st.rerun()

    # ── Build calendar HTML ──────────────────────────────────────────────────
    year  = st.session_state._cal_year
    month = st.session_state._cal_month
    first_wd, days_in_month = cal_lib.monthrange(year, month)
    # first_wd: 0=Mon … 6=Sun  →  shift to start on Sunday
    start_offset = (first_wd + 1) % 7

    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    header_html = "".join(f'<div class="cal-header-cell">{d}</div>' for d in day_names)

    cells_html = ""
    cell_num   = 0
    day_num    = 1
    total_cells = start_offset + days_in_month
    total_cells = total_cells + (7 - total_cells % 7) if total_cells % 7 != 0 else total_cells

    for _ in range(total_cells):
        if cell_num < start_offset or day_num > days_in_month:
            cells_html += '<div class="cal-day other-month"></div>'
        else:
            d    = datetime.date(year, month, day_num)
            dstr = str(d)
            is_today   = d == today
            day_events = events.get(dstr, [])
            has_ev     = len(day_events) > 0

            cls  = "cal-day"
            if is_today:  cls += " today"
            if has_ev:    cls += " has-events"

            dots = "".join('<span class="event-dot"></span>' for _ in day_events[:4])
            if len(day_events) > 4:
                dots += f'<span style="font-size:0.55rem;color:#6366f1;font-weight:700;">+{len(day_events)-4}</span>'

            cells_html += (
                f'<div class="{cls}" title="{len(day_events)} event(s)">'
                f'<div class="cal-day-num">{day_num}</div>'
                f'<div style="margin-top:3px;">{dots}</div>'
                f'</div>'
            )
            day_num += 1
        cell_num += 1

    st.markdown(
        f'<div class="cal-grid">{header_html}{cells_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Quick event count bar ────────────────────────────────────────────────
    month_events = {k: v for k, v in events.items()
                    if k.startswith(f"{year}-{month:02d}")}
    total_ev = sum(len(v) for v in month_events.values())
    if total_ev:
        st.markdown(
            f'<div style="text-align:right;font-size:0.72rem;color:#6366f1;'
            f'font-weight:600;margin-top:0.4rem;">'
            f'📌 {total_ev} task{"s" if total_ev!=1 else ""} this month</div>',
            unsafe_allow_html=True,
        )

    # ── View all dates & tasks ───────────────────────────────────────────────
    if events:
        with st.expander("📋  View All Dates & Tasks", expanded=False):
            sorted_dates = sorted(events.keys())
            for dstr in sorted_dates:
                tasks = events[dstr]
                try:
                    d_obj = datetime.date.fromisoformat(dstr)
                    label = d_obj.strftime("%A, %d %B %Y")
                    is_past  = d_obj < today
                    is_today2 = d_obj == today
                    color = "#22c55e" if is_today2 else ("#9ca3af" if is_past else "#6366f1")
                except Exception:
                    label = dstr
                    color = "#6366f1"
                st.markdown(
                    f'<div style="margin-bottom:0.7rem;">'
                    f'<div style="font-size:0.75rem;font-weight:700;color:{color};'
                    f'text-transform:uppercase;letter-spacing:0.5px;">{label}</div>',
                    unsafe_allow_html=True,
                )
                for t in tasks:
                    st.markdown(
                        f'<div style="font-size:0.83rem;color:#374151;'
                        f'padding:0.25rem 0.6rem;background:#f8faff;'
                        f'border-left:3px solid {color};border-radius:0 6px 6px 0;'
                        f'margin:0.2rem 0;">• {t}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Click-a-day detail ───────────────────────────────────────────────────
    st.markdown('<div class="ws-section">📌 DAY DETAIL — VIEW & ADD TASK</div>', unsafe_allow_html=True)
    sel_date = st.date_input(
        "Select a date to view or add tasks",
        value=today,
        key="cal_sel_date",
        help="Pick any date on the calendar to see or add tasks"
    )
    sel_str  = str(sel_date)
    day_tasks = events.get(sel_str, [])

    if day_tasks:
        st.markdown(
            f'<div style="font-size:0.8rem;font-weight:700;color:#4f46e5;margin-bottom:0.4rem;">'
            f'Tasks on {sel_date.strftime("%d %B %Y")} ({len(day_tasks)} item{"s" if len(day_tasks)!=1 else ""})</div>',
            unsafe_allow_html=True,
        )
        for idx, t in enumerate(day_tasks):
            c1, c2 = st.columns([8, 1])
            with c1:
                st.markdown(
                    f'<div style="background:#f0f4ff;border-radius:8px;padding:0.45rem 0.7rem;'
                    f'font-size:0.84rem;color:#374151;border-left:3px solid #6366f1;'
                    f'margin-bottom:0.25rem;">📌 {t}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"del_ev_{sel_str}_{idx}", help="Remove this task"):
                    events[sel_str].pop(idx)
                    st.rerun()
    else:
        st.markdown(
            f'<div style="font-size:0.8rem;color:#9ca3af;font-style:italic;">'
            f'No tasks for {sel_date.strftime("%d %B %Y")}.</div>',
            unsafe_allow_html=True,
        )

    # ── Add task to selected date ────────────────────────────────────────────
    new_task_c, add_btn_c = st.columns([5, 1])
    with new_task_c:
        new_task = st.text_input(
            "Add task / plan note", placeholder="e.g. Sprint review meeting",
            key="cal_new_task", label_visibility="collapsed"
        )
    with add_btn_c:
        if st.button("➕ Add", key="cal_add_task", use_container_width=True):
            if new_task.strip():
                events.setdefault(sel_str, []).append(new_task.strip())
                st.success(f"Added to {sel_date.strftime('%d %b %Y')}")
                st.rerun()
            else:
                st.warning("Please enter a task description.")


# ─────────────────────────────────────────────────────────────────────────────
#  DOCUMENT UPLOAD → DATE EXTRACTION → CALENDAR + ATTACHMENTS
# ─────────────────────────────────────────────────────────────────────────────
def _document_upload_section(proj):
    events      = proj["calendar_events"]
    attachments = proj["attachments"]

    st.markdown('<div class="ws-section">📎 UPLOAD DOCUMENT WITH DATES</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.7rem;">'
        'Upload a project document — all dates found in it will be extracted and added to your calendar automatically.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload document",
        type=["txt", "csv", "md", "pdf", "docx"],
        key="doc_upload",
        label_visibility="collapsed",
        help="Supported: TXT, CSV, MD, PDF, DOCX",
    )

    if uploaded:
        already = any(a["name"] == uploaded.name for a in attachments)
        if not already:
            text = read_uploaded_text(uploaded)
            dates = extract_dates_from_text(text)

            # ── Store attachment ─────────────────────────────────────────────
            raw_bytes = uploaded.getvalue() if hasattr(uploaded, "getvalue") else b""
            b64 = base64.b64encode(raw_bytes).decode() if raw_bytes else ""
            preview_lines = text[:400] if text else ""
            attachments.append({
                "name":     uploaded.name,
                "size":     f"{len(raw_bytes)/1024:.1f} KB",
                "preview":  preview_lines,
                "b64":      b64,
                "mime":     uploaded.type or "application/octet-stream",
                "uploaded": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

            # ── Add extracted dates to calendar ──────────────────────────────
            if dates:
                added_count = 0
                for d in dates:
                    dstr = str(d)
                    task_entry = f"[From: {uploaded.name}] Date milestone"
                    existing   = events.get(dstr, [])
                    if task_entry not in existing:
                        events.setdefault(dstr, []).append(task_entry)
                        added_count += 1

                st.success(
                    f"✅  **{uploaded.name}** uploaded! "
                    f"Found **{len(dates)}** date(s) → added **{added_count}** new calendar entry(ies)."
                )
                with st.expander(f"📅 Dates extracted from '{uploaded.name}'", expanded=True):
                    for d in dates:
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:0.5rem;'
                            f'padding:0.3rem 0.5rem;background:#eef2ff;border-radius:8px;'
                            f'margin-bottom:0.25rem;font-size:0.83rem;color:#3730a3;">'
                            f'📅 <strong>{d.strftime("%A, %d %B %Y")}</strong></div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info(f"📄 **{uploaded.name}** saved to attachments. No recognisable dates found in the document.")
        else:
            st.info(f"'{uploaded.name}' is already in attachments.")

    # ── Attachments panel ────────────────────────────────────────────────────
    if attachments:
        st.markdown('<div class="ws-section">📂 ATTACHMENTS</div>', unsafe_allow_html=True)
        for i, att in enumerate(attachments):
            ext   = att["name"].rsplit(".", 1)[-1].upper() if "." in att["name"] else "FILE"
            color_map = {"PDF": "#ef4444", "DOCX": "#3b82f6", "TXT": "#6b7280",
                         "CSV": "#22c55e", "MD": "#8b5cf6"}
            ec    = color_map.get(ext, "#6366f1")
            c1, c2 = st.columns([7, 1])
            with c1:
                st.markdown(
                    f'<div class="attach-pill">'
                    f'<span style="background:{ec}20;color:{ec};font-size:0.62rem;font-weight:800;'
                    f'padding:0.15rem 0.35rem;border-radius:5px;">{ext}</span>'
                    f'<div style="flex:1;">'
                    f'<div style="font-weight:600;font-size:0.83rem;color:#111827;">{att["name"]}</div>'
                    f'<div style="font-size:0.7rem;color:#9ca3af;">{att["size"]} · {att["uploaded"]}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🗑️", key=f"del_att_{i}", help="Remove attachment"):
                    attachments.pop(i)
                    st.rerun()
            if att.get("preview"):
                with st.expander(f"👁️ Preview: {att['name']}", expanded=False):
                    st.text(att["preview"][:600] + ("…" if len(att["preview"]) > 600 else ""))


# ─────────────────────────────────────────────────────────────────────────────
#  VIEW MODAL — full project detail
# ─────────────────────────────────────────────────────────────────────────────
def _view_project_modal(proj):
    _ensure_keys(proj)
    status_color = {"In Progress": "#f59e0b", "Completed": "#22c55e", "Review": "#6366f1"}.get(
        proj.get("status", "In Progress"), "#6b7280"
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;">'
        f'<span style="font-size:1.5rem;">📁</span>'
        f'<div><div style="font-size:1.1rem;font-weight:800;color:#111827;">{proj["name"]}</div>'
        f'<div style="font-size:0.75rem;color:#9ca3af;">Dataset: {proj.get("dataset","")} · '
        f'{proj.get("rows",0):,} rows · Created: {proj.get("created","")}</div></div>'
        f'<span style="margin-left:auto;font-size:0.72rem;background:{status_color}20;'
        f'color:{status_color};padding:0.2rem 0.6rem;border-radius:20px;font-weight:700;">'
        f'{proj.get("status","In Progress")}</span></div>',
        unsafe_allow_html=True,
    )

    tab_cal, tab_cw, tab_er, tab_att = st.tabs(
        ["📅 Plan Dates", "🛠️ Current Work", "🗂️ ER Diagrams", "📎 Attachments"]
    )

    # ── Tab: Plan Dates ───────────────────────────────────────────────────────
    with tab_cal:
        _calendar_section(proj)
        _document_upload_section(proj)

    # ── Tab: Current Work ─────────────────────────────────────────────────────
    with tab_cw:
        cw = proj["current_work"]
        st.markdown('<div class="ws-section">🛠️ PROJECT CURRENT WORK</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.7rem;">'
            'Track active tasks and ongoing work items for this project.</div>',
            unsafe_allow_html=True,
        )

        # Add new work item
        inp_c, add_c = st.columns([6, 1])
        with inp_c:
            new_cw = st.text_input(
                "New work item",
                placeholder="e.g. Build data pipeline, Fix null values in column X…",
                key="cw_new",
                label_visibility="collapsed",
            )
        with add_c:
            if st.button("➕ Add", key="cw_add", use_container_width=True):
                if new_cw.strip():
                    cw.append({"task": new_cw.strip(), "done": False,
                               "added": datetime.datetime.now().strftime("%Y-%m-%d")})
                    st.rerun()
                else:
                    st.warning("Enter a task description.")

        if cw:
            done_count = sum(1 for t in cw if t.get("done"))
            pct = int(done_count / len(cw) * 100) if cw else 0
            st.markdown(
                f'<div style="margin-bottom:0.8rem;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.73rem;color:#6b7280;margin-bottom:0.2rem;">'
                f'<span>Progress</span><span>{done_count}/{len(cw)} completed ({pct}%)</span></div>'
                f'<div style="background:#e5e7eb;border-radius:6px;height:8px;">'
                f'<div style="background:linear-gradient(90deg,#4f46e5,#6366f1);'
                f'width:{pct}%;height:8px;border-radius:6px;transition:width 0.3s;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            for idx, item in enumerate(cw):
                done = item.get("done", False)
                ic1, ic2, ic3 = st.columns([0.5, 7, 1])
                with ic1:
                    checked = st.checkbox("", value=done, key=f"cw_chk_{idx}", label_visibility="collapsed")
                    if checked != done:
                        cw[idx]["done"] = checked
                        st.rerun()
                with ic2:
                    style = "text-decoration:line-through;color:#9ca3af;" if done else "color:#111827;"
                    st.markdown(
                        f'<div style="padding:0.4rem 0;font-size:0.85rem;{style}">'
                        f'{item["task"]}'
                        f'<span style="font-size:0.65rem;color:#d1d5db;margin-left:0.5rem;">'
                        f'Added {item.get("added","")}</span></div>',
                        unsafe_allow_html=True,
                    )
                with ic3:
                    if st.button("✕", key=f"cw_del_{idx}", help="Remove"):
                        cw.pop(idx)
                        st.rerun()
        else:
            st.markdown(
                '<div style="text-align:center;padding:2rem;color:#9ca3af;font-size:0.85rem;">'
                '<div style="font-size:2rem;margin-bottom:0.4rem;">🛠️</div>'
                'No work items yet. Add your first task above!</div>',
                unsafe_allow_html=True,
            )

    # ── Tab: ER Diagrams ──────────────────────────────────────────────────────
    with tab_er:
        er_images = proj["er_images"]
        st.markdown('<div class="ws-section">🗂️ PROJECT ER DIAGRAMS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.7rem;">'
            'Upload Entity-Relationship diagrams or any project architecture images.</div>',
            unsafe_allow_html=True,
        )

        er_upload = st.file_uploader(
            "Upload ER diagram",
            type=["png", "jpg", "jpeg", "svg", "pdf", "webp"],
            accept_multiple_files=True,
            key="er_upload",
            label_visibility="collapsed",
        )
        if er_upload:
            for f in er_upload:
                if not any(e["name"] == f.name for e in er_images):
                    raw   = f.read()
                    b64   = base64.b64encode(raw).decode()
                    mime  = f.type or "image/png"
                    er_images.append({
                        "name":     f.name,
                        "data_b64": b64,
                        "mime":     mime,
                        "size":     f"{len(raw)/1024:.1f} KB",
                        "added":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    })
            st.success(f"✅ {len(er_upload)} image(s) uploaded.")
            st.rerun()

        if er_images:
            st.markdown(
                f'<div style="font-size:0.78rem;color:#6b7280;margin-bottom:0.6rem;">'
                f'{len(er_images)} diagram(s) stored</div>',
                unsafe_allow_html=True,
            )
            for idx, er in enumerate(er_images):
                with st.container():
                    er_c1, er_c2 = st.columns([8, 1])
                    with er_c1:
                        st.markdown(
                            f'<div style="font-size:0.83rem;font-weight:600;color:#111827;">'
                            f'🗂️ {er["name"]} <span style="font-size:0.7rem;color:#9ca3af;'
                            f'font-weight:400;">{er["size"]} · {er["added"]}</span></div>',
                            unsafe_allow_html=True,
                        )
                    with er_c2:
                        if st.button("🗑️", key=f"er_del_{idx}", help="Remove"):
                            er_images.pop(idx)
                            st.rerun()
                    # Show image if it is an image type
                    if er["mime"].startswith("image/"):
                        try:
                            img_bytes = base64.b64decode(er["data_b64"])
                            st.image(img_bytes, caption=er["name"], use_container_width=True)
                        except Exception:
                            st.caption("(Cannot render this image)")
                    else:
                        st.markdown(
                            f'<div style="background:#f8faff;border-radius:8px;'
                            f'padding:0.6rem 1rem;font-size:0.78rem;color:#6b7280;">'
                            f'📄 {er["name"]} — {er["size"]}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="text-align:center;padding:2rem;color:#9ca3af;font-size:0.85rem;">'
                '<div style="font-size:2.5rem;margin-bottom:0.5rem;">🗂️</div>'
                'No ER diagrams uploaded yet.</div>',
                unsafe_allow_html=True,
            )

    # ── Tab: Attachments ──────────────────────────────────────────────────────
    with tab_att:
        attachments = proj["attachments"]
        st.markdown('<div class="ws-section">📎 DOCUMENT ATTACHMENTS</div>', unsafe_allow_html=True)
        if attachments:
            for i, att in enumerate(attachments):
                ext = att["name"].rsplit(".", 1)[-1].upper() if "." in att["name"] else "FILE"
                color_map = {"PDF": "#ef4444", "DOCX": "#3b82f6", "TXT": "#6b7280",
                             "CSV": "#22c55e", "MD": "#8b5cf6"}
                ec  = color_map.get(ext, "#6366f1")
                ac1, ac2 = st.columns([8, 1])
                with ac1:
                    st.markdown(
                        f'<div class="attach-pill">'
                        f'<span style="background:{ec}20;color:{ec};font-size:0.62rem;font-weight:800;'
                        f'padding:0.15rem 0.35rem;border-radius:5px;">{ext}</span>'
                        f'<div><div style="font-weight:600;font-size:0.83rem;color:#111827;">{att["name"]}</div>'
                        f'<div style="font-size:0.7rem;color:#9ca3af;">{att["size"]} · {att["uploaded"]}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                with ac2:
                    if st.button("🗑️", key=f"view_att_del_{i}"):
                        attachments.pop(i)
                        st.rerun()
                if att.get("preview"):
                    with st.expander(f"👁️ Preview: {att['name']}"):
                        st.text(att["preview"][:600])
        else:
            st.markdown(
                '<div style="text-align:center;padding:2rem;color:#9ca3af;font-size:0.85rem;">'
                '<div style="font-size:2rem;margin-bottom:0.4rem;">📎</div>'
                'No attachments yet. Upload documents from the Plan Dates tab.</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render():
    st.markdown(WORKSPACE_CSS, unsafe_allow_html=True)
    projects = _load_projects()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.7rem;padding:0.8rem 0 0.2rem;">'
        '<div style="width:48px;height:48px;border-radius:14px;'
        'background:linear-gradient(135deg,#4f46e5,#6366f1);'
        'display:flex;align-items:center;justify-content:center;font-size:1.4rem;">📁</div>'
        '<div><div style="font-size:1.4rem;font-weight:800;color:#111827;">Workspace</div>'
        '<div style="font-size:0.8rem;color:#9ca3af;">Save, manage, and plan your analysis projects</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Save current project ─────────────────────────────────────────────────
    if st.session_state.get("filename"):
        st.markdown('<div class="ws-section">💾 SAVE CURRENT PROJECT</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([3, 1.5, 1])
        with c1:
            proj_name = st.text_input(
                "Project name",
                value=st.session_state.get("filename", "").rsplit(".", 1)[0],
                key="ws_proj_name",
                label_visibility="collapsed",
                placeholder="Project name…",
            )
        with c2:
            proj_status = st.selectbox(
                "Status", ["In Progress", "Completed", "Review"],
                key="ws_proj_status", label_visibility="collapsed"
            )
        with c3:
            if st.button("💾 Save", key="ws_save", type="primary", use_container_width=True):
                if proj_name.strip():
                    exists = _get_project(proj_name.strip())
                    if exists:
                        exists["date"]   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        exists["status"] = proj_status
                        st.success(f"Updated '{proj_name}'")
                    else:
                        new_proj = {
                            "name":        proj_name.strip(),
                            "date":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "dataset":     st.session_state.get("filename", "Unknown"),
                            "rows":        len(st.session_state.df) if st.session_state.get("df") is not None else 0,
                            "cols":        len(st.session_state.df.columns) if st.session_state.get("df") is not None else 0,
                            "status":      proj_status,
                            "created":     datetime.datetime.now().strftime("%Y-%m-%d"),
                            "last_opened": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "calendar_events": {},
                            "attachments":     [],
                            "current_work":    [],
                            "er_images":       [],
                        }
                        projects.append(new_proj)
                        st.success(f"✅ Saved '{proj_name}'")
                        st.rerun()
                else:
                    st.warning("Please enter a project name.")

    # ── Projects list ────────────────────────────────────────────────────────
    if not projects:
        st.markdown(
            '<div style="background:#f9fafb;border:1.5px dashed #d1d5db;border-radius:16px;'
            'padding:3rem 2rem;text-align:center;margin-top:1rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.5rem;">📂</div>'
            '<div style="font-size:0.95rem;font-weight:600;color:#374151;">No saved projects yet</div>'
            '<div style="font-size:0.8rem;color:#9ca3af;margin-top:0.3rem;">'
            'Upload a dataset and save your first project to get started.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="ws-section">📁 SAVED PROJECTS</div>', unsafe_allow_html=True)

    # Check if a project is being viewed
    viewing_key = "_ws_viewing"
    if st.session_state.get(viewing_key):
        vname = st.session_state[viewing_key]
        vproj = _get_project(vname)
        if vproj:
            _ensure_keys(vproj)
            if st.button("← Back to Projects", key="ws_back"):
                st.session_state[viewing_key] = None
                st.rerun()
            st.markdown("---")
            _view_project_modal(vproj)
            return
        else:
            st.session_state[viewing_key] = None

    # ── Project cards ────────────────────────────────────────────────────────
    for i, proj in enumerate(projects):
        _ensure_keys(proj)
        status = proj.get("status", "In Progress")
        status_color = {"In Progress": "#f59e0b", "Completed": "#22c55e", "Review": "#6366f1"}.get(status, "#6b7280")
        n_events = sum(len(v) for v in proj["calendar_events"].values())
        n_tasks  = len(proj["current_work"])
        n_att    = len(proj["attachments"]) + len(proj["er_images"])

        st.markdown('<div class="proj-card">', unsafe_allow_html=True)
        info_col, btn_col = st.columns([5, 3])
        with info_col:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">'
                f'<span style="font-weight:800;font-size:0.95rem;color:#111827;">{proj["name"]}</span>'
                f'<span style="font-size:0.65rem;background:{status_color}20;color:{status_color};'
                f'padding:0.12rem 0.45rem;border-radius:20px;font-weight:700;">{status}</span>'
                f'</div>'
                f'<div style="font-size:0.73rem;color:#9ca3af;margin-top:0.25rem;">'
                f'📂 {proj.get("dataset","")} &nbsp;·&nbsp; '
                f'{proj.get("rows",0):,} rows &nbsp;·&nbsp; '
                f'Created {proj.get("created","")}'
                f'</div>'
                f'<div style="display:flex;gap:0.6rem;margin-top:0.4rem;">'
                f'<span style="font-size:0.68rem;background:#eef2ff;color:#4f46e5;'
                f'padding:0.1rem 0.4rem;border-radius:5px;font-weight:600;">📅 {n_events} event{"s" if n_events!=1 else ""}</span>'
                f'<span style="font-size:0.68rem;background:#f0fdf4;color:#15803d;'
                f'padding:0.1rem 0.4rem;border-radius:5px;font-weight:600;">🛠️ {n_tasks} task{"s" if n_tasks!=1 else ""}</span>'
                f'<span style="font-size:0.68rem;background:#fdf4ff;color:#7e22ce;'
                f'padding:0.1rem 0.4rem;border-radius:5px;font-weight:600;">📎 {n_att} file{"s" if n_att!=1 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with btn_col:
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                if st.button("👁️ View", key=f"ws_view_{i}", use_container_width=True, type="primary"):
                    st.session_state[viewing_key] = proj["name"]
                    st.rerun()
            with bc2:
                st.selectbox(
                    "Status",
                    ["In Progress", "Completed", "Review"],
                    index=["In Progress", "Completed", "Review"].index(status),
                    key=f"ws_st_{i}",
                    label_visibility="collapsed",
                    on_change=lambda i=i: setattr(
                        projects[i], "status",
                        st.session_state.get(f"ws_st_{i}", "In Progress")
                    )
                )
            with bc3:
                if st.button("🗑️ Del", key=f"ws_del_{i}", use_container_width=True):
                    projects.pop(i)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

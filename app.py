import streamlit as st

st.set_page_config(
    page_title="Smart Dataset Analysis Platform",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np

# Categorized sidebar pages structure with Report Generation and Export at the very end
NAV_SECTIONS = [
    {
        "title": "Core Analytics",
        "icon": "📊",
        "pages": ["Dashboard", "Dataset Overview", "Raw Dataset", "Data Cleaning"]
    },
    {
        "title": "Analysis & Visuals",
        "icon": "📈",
        "pages": [
            "Statistical Analysis", "Visualizations", "Correlation Analysis",
            "Feature Analysis", "Custom Analysis", "Advanced Analytics", "Predictive Modeling", "All Charts"
        ]
    },
    {
        "title": "AI & Insights",
        "icon": "🤖",
        "pages": [
            "AI Insights", "Auto Dashboard", "Dataset Health", "Type Detection",
            "Analysis Recs", "Data Storytelling", "AI Recs", "AI Chat", "Root Cause"
        ]
    },
    {
        "title": "Workspace & History",
        "icon": "💼",
        "pages": ["Version History", "Workspace"]
    },
    {
        "title": "Reports & Export",
        "icon": "📑",
        "pages": ["Report Generation", "Export"]
    }
]

# Flattened list of pages for lookup
PAGES = [page for section in NAV_SECTIONS for page in section["pages"]]

PAGE_ICONS = {
    "Dashboard": "🏠", "Dataset Overview": "📋", "Raw Dataset": "📄",
    "Data Cleaning": "🧹", "Statistical Analysis": "📈", "Visualizations": "🎨",
    "Correlation Analysis": "🔗", "Feature Analysis": "🔍", "Custom Analysis": "⚙️",
    "Advanced Analytics": "🚀", "Predictive Modeling": "🔮", "All Charts": "📊", "AI Insights": "🤖",
    "Auto Dashboard": "🤖", "Dataset Health": "❤️", "Type Detection": "🏷️",
    "Analysis Recs": "💡", "Data Storytelling": "📖", "AI Recs": "🎯",
    "AI Chat": "💬", "Root Cause": "🔍", "Version History": "🕒", "Workspace": "💼",
    "Report Generation": "📝", "Export": "💾",
}

ALWAYS_ACCESSIBLE = {"Dashboard"}


@st.cache_resource
def get_cached_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

        /* BASE & SCALING & LAYOUT CONTAINMENT */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            font-display: swap !important;
        }
        html { font-size: 19.5px !important; }

        .stApp {
            background: linear-gradient(165deg, #f4f7fc 0%, #ebf0f9 45%, #e2e8f5 100%) !important;
            color: #0f172a !important;
            contain: layout paint !important;
        }

        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1800px !important;
            min-height: 90vh !important;
        }

        /* CLS LAYOUT SHIFT CONTAINMENT RULES */
        section.stMain {
            contain: layout paint !important;
            min-height: 80vh !important;
        }

        div.stVerticalBlock {
            min-height: 40px !important;
        }

        div.stElementContainer {
            min-height: 24px !important;
        }

        div[data-testid="stSidebar"] {
            contain: layout paint !important;
        }

        /* Fixed Heights for Form & Nav Controls to Eliminate Layout Shift (CLS) */
        div[class*="st-key-nav_"] button,
        div[class*="st-key-pred_"] {
            min-height: 44px !important;
        }

        header[data-testid="stHeader"],
        div[data-testid="stHeader"],
        .stHeader,
        footer,
        .stAppDeployButton,
        .stActionButton,
        #MainMenu,
        div[data-testid="stDecoration"] {
            display: none !important;
            height: 0 !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }

        /* TYPOGRAPHY */
        h1 {
            font-size: 2.3rem !important;
            font-weight: 800 !important;
            color: #0f172a !important;
            letter-spacing: -0.6px !important;
        }
        h2 {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #0f172a !important;
            letter-spacing: -0.4px !important;
        }
        h3 {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #1e293b !important;
            letter-spacing: -0.2px !important;
        }

        /* SIDEBAR STYLING */
        div[data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
            box-shadow: 6px 0 35px rgba(15, 23, 42, 0.06) !important;
            min-width: 320px !important;
            width: 320px !important;
        }
        div[data-testid="stSidebar"] > div:first-child {
            padding-top: 0 !important;
            padding-bottom: 2rem !important;
        }

        /* SIDEBAR BUTTONS */
        div[data-testid="stSidebar"] .stButton > button {
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 12px !important;
            color: #475569 !important;
            font-weight: 600 !important;
            font-size: 1.02rem !important;
            padding: 0.65rem 1rem !important;
            text-align: left !important;
            width: 100% !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: none !important;
            margin: 0.1rem 0;
            display: flex !important;
            align-items: center !important;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
            background: #eef2ff !important;
            color: #4f46e5 !important;
            border-color: #c7d2fe !important;
            transform: translateX(3px) !important;
        }
        div[data-testid="stSidebar"] .stButton > button:focus { outline: none !important; }

        /* CARDS & CONTAINERS */
        .css-card, .stCard, .metric-card {
            background: #ffffff !important;
            border-radius: 16px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05) !important;
            padding: 1.25rem !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        .css-card:hover {
            background: #ffffff !important;
            border-radius: 16px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05) !important;
            padding: 1.1rem 1.3rem !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #cbd5e1 !important;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08) !important;
        }
        div[data-testid="stMetricValue"] {
            font-weight: 800 !important;
            color: #0f172a !important;
            font-size: 2.1rem !important;
            letter-spacing: -0.5px !important;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 700 !important;
            color: #64748b !important;
            font-size: 0.88rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
        }

        /* BUTTONS GENERAL */
        .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 1.02rem !important;
            padding: 0.65rem 1.3rem !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            color: #334155 !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
        }
        .stButton > button:hover {
            border-color: #818cf8 !important;
            color: #4f46e5 !important;
            background: #f8fafc !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.15) !important;
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
            font-weight: 700 !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 10px 28px rgba(99, 102, 241, 0.45) !important;
            transform: translateY(-2px) !important;
        }

        /* FORM CONTROLS */
        .stSelectbox label, .stMultiSelect label,
        .stTextInput label, .stNumberInput label, .stTextArea label {
            font-weight: 700 !important;
            font-size: 0.98rem !important;
            color: #1e293b !important;
            margin-bottom: 0.35rem !important;
        }
        div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
            font-size: 1.05rem !important;
            padding: 0.15rem 0.3rem !important;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: #6366f1 !important;
        }
        .stNumberInput input, .stTextInput input, .stTextArea textarea {
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
            font-size: 1.05rem !important;
            padding: 0.65rem 0.9rem !important;
        }
        .stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.16) !important;
        }

        /* TABS STYLING */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem !important;
            background: #ffffff !important;
            border-radius: 14px !important;
            padding: 0.45rem !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04) !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 1.02rem !important;
            padding: 0.6rem 1.25rem !important;
            transition: all 0.2s ease !important;
            color: #64748b !important;
            background: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* DATAFRAME */
        .stDataFrame {
            border-radius: 16px !important;
            overflow: hidden !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;
            font-size: 1rem !important;
        }

        /* PLOTLY CHARTS */
        .js-plotly-plot { border-radius: 16px !important; }
        .stPlotlyChart {
            border-radius: 16px !important;
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06) !important;
            padding: 0.6rem !important;
            margin-bottom: 1.2rem !important;
        }

        /* FILE UPLOADER */
        section[data-testid="stFileUploadDropzone"] {
            background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%) !important;
            border: 2px dashed #818cf8 !important;
            border-radius: 20px !important;
            padding: 3.5rem 2rem !important;
            transition: all 0.25s ease !important;
        }
        section[data-testid="stFileUploadDropzone"]:hover {
            border-color: #4f46e5 !important;
            background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%) !important;
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.18) !important;
        }

        /* INSIGHT BOX */
        .insight-box {
            background: #f8fafc !important;
            border-left: 5px solid #6366f1 !important;
            border-radius: 0 14px 14px 0 !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.8rem !important;
            font-size: 1.05rem !important;
            color: #334155 !important;
            line-height: 1.7 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
        }

        /* SECTION TITLE */
        .section-title {
            font-size: 1.5rem !important;
            font-weight: 800 !important;
            color: #0f172a !important;
            margin: 1.5rem 0 1rem 0 !important;
            padding-bottom: 0.6rem !important;
            border-bottom: 2px solid #e2e8f0 !important;
            letter-spacing: -0.4px !important;
        }

        /* DOWNLOAD BUTTON */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 1.02rem !important;
            border: none !important;
            box-shadow: 0 6px 18px rgba(99, 102, 241, 0.32) !important;
        }

        /* EXPANDER & ALERT */
        .stAlert { border-radius: 14px !important; font-size: 1.02rem !important; }
        .streamlit-expanderHeader {
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 1.02rem !important;
        }

        /* SCROLLBAR */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 10px; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #a5b4fc, #6366f1);
            border-radius: 10px;
        }

        /* CHAT INPUT */
        .stChatInput textarea { font-size: 1.05rem !important; border-radius: 14px !important; }
        [data-testid="stChatMessage"] { font-size: 1.05rem !important; }
    </style>
    """


def inject_css():
    st.markdown(get_cached_css(), unsafe_allow_html=True)


def init_session():
    defaults = {
        "page": "Dashboard",
        "df": None,
        "original_df": None,
        "filename": None,
        "cleaning_history": [],
        "uploaded": False,
        "_expanded": False,
        "_expanded_fig": None,
        "_expanded_insight": "",
        "_view_all": False,
        "_view_all_figs": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sidebar_nav():
    has_data = st.session_state.df is not None
    with st.sidebar:
        # App Header Banner
        st.markdown(
            '<div style="background:linear-gradient(160deg,#0f172a 0%,#1e1b4b 55%,#312e81 100%);'
            'padding:1.6rem 1.2rem 1.4rem;margin:-1rem -1rem 1rem -1rem;text-align:center;position:relative;overflow:hidden;'
            'box-shadow:0 8px 25px rgba(15,23,42,0.18);">'
            '<div style="position:absolute;top:-20px;right:-20px;width:90px;height:90px;background:rgba(99,102,241,0.25);border-radius:50%;filter:blur(10px);"></div>'
            '<div style="position:absolute;bottom:-30px;left:-30px;width:110px;height:110px;background:rgba(139,92,246,0.18);border-radius:50%;filter:blur(12px);"></div>'
            '<div style="position:relative;z-index:2;">'
            '<div style="font-size:2.4rem;margin-bottom:0.25rem;">📊</div>'
            '<div style="color:#ffffff;font-weight:800;font-size:1.25rem;letter-spacing:-0.4px;">DataAnalyzer</div>'
            '<div style="color:rgba(255,255,255,0.45);font-size:0.68rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-top:0.1rem;">AI Analytics Platform</div>'
            '</div></div>',
            unsafe_allow_html=True
        )

        # Dataset Status Card
        if has_data:
            df = st.session_state.df
            st.markdown(
                '<div style="background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);border:1px solid #86efac;border-radius:14px;'
                'padding:0.75rem 0.95rem;margin:0 0 1rem 0;box-shadow:0 3px 10px rgba(34,197,94,0.08);">'
                '<div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.25rem;">'
                '<span style="width:8px;height:8px;background:#22c55e;border-radius:50%;display:inline-block;box-shadow:0 0 6px rgba(34,197,94,0.6);"></span>'
                '<span style="font-weight:800;color:#166534;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.6px;">Dataset Active</span></div>'
                f'<div style="color:#15803d;font-weight:700;font-size:0.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:240px;">{st.session_state.filename}</div>'
                f'<div style="color:#16a34a;font-size:0.78rem;font-weight:600;margin-top:0.1rem;">{len(df):,} rows / {len(df.columns)} columns</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#f8fafc;border:1.5px dashed #cbd5e1;border-radius:14px;'
                'padding:0.85rem 1rem;margin:0 0 1rem 0;text-align:center;color:#64748b;">'
                '<div style="font-size:1.5rem;margin-bottom:0.2rem;">☁️</div>'
                '<div style="font-weight:700;font-size:0.88rem;color:#334155;">No dataset loaded</div>'
                '<div style="font-size:0.75rem;color:#94a3b8;margin-top:0.1rem;">Upload a CSV/Excel on Dashboard</div></div>',
                unsafe_allow_html=True
            )

        # Categorized Sidebar Items
        for section in NAV_SECTIONS:
            st.markdown(
                f'<div style="font-size:0.72rem;font-weight:800;color:#64748b;text-transform:uppercase;'
                f'letter-spacing:1.3px;padding:0.6rem 0.3rem 0.3rem 0.3rem;margin-top:0.4rem;display:flex;align-items:center;gap:0.35rem;">'
                f'<span>{section["icon"]}</span><span>{section["title"]}</span></div>',
                unsafe_allow_html=True
            )

            for page in section["pages"]:
                icon = PAGE_ICONS.get(page, "📄")
                is_active = st.session_state.page == page
                is_accessible = has_data or page in ALWAYS_ACCESSIBLE

                if is_active:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:0.65rem;padding:0.6rem 0.95rem;border-radius:12px;'
                        f'background:linear-gradient(135deg,#4f46e5,#6366f1);color:#ffffff;font-weight:700;font-size:1.02rem;'
                        f'margin:0.12rem 0;box-shadow:0 4px 16px rgba(99,102,241,0.38);border:1px solid rgba(255,255,255,0.2);">'
                        f'<span style="font-size:1.15rem;">{icon}</span><span>{page}</span></div>',
                        unsafe_allow_html=True
                    )
                elif is_accessible:
                    st.button(f"{icon}  {page}", key=f"nav_{page}", on_click=_on_nav_change, args=(page,), use_container_width=True)
                else:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:0.65rem;padding:0.55rem 0.95rem;border-radius:12px;'
                        f'color:#94a3b8;font-size:0.98rem;margin:0.12rem 0;cursor:not-allowed;background:rgba(241,245,249,0.5);">'
                        f'<span style="font-size:1.1rem;opacity:0.4;">{icon}</span>'
                        f'<span style="font-weight:500;color:#94a3b8;">{page}</span>'
                        f'<span style="margin-left:auto;font-size:0.62rem;background:#e2e8f0;color:#64748b;padding:0.15rem 0.45rem;border-radius:6px;font-weight:800;letter-spacing:0.5px;">LOCKED</span></div>',
                        unsafe_allow_html=True
                    )

        # Sidebar Footer
        st.markdown(
            '<div style="border-top:1px solid #e2e8f0;margin-top:1.2rem;padding-top:0.9rem;text-align:center;">'
            '<div style="font-size:0.75rem;color:#64748b;font-weight:600;">Smart Dataset Analysis Platform</div>'
            '<div style="font-size:0.68rem;color:#94a3b8;margin-top:0.2rem;font-weight:500;">v3.5 Professional Edition</div></div>',
            unsafe_allow_html=True
        )


def _on_nav_change(target_page):
    st.session_state.page = target_page


# Pre-import page modules for instant O(1) page loading speed
import modules.dashboard as mod_dashboard
import modules.dataset_overview as mod_dataset_overview
import modules.raw_dataset as mod_raw_dataset
import modules.data_cleaning as mod_data_cleaning
import modules.statistical_analysis as mod_statistical_analysis
import modules.visualizations as mod_visualizations
import modules.correlation_analysis as mod_correlation_analysis
import modules.feature_analysis as mod_feature_analysis
import modules.custom_analysis as mod_custom_analysis
import modules.advanced_analytics as mod_advanced_analytics
import modules.all_charts as mod_all_charts
import modules.ai_insights as mod_ai_insights
import modules.auto_dashboard as mod_auto_dashboard
import modules.dataset_health as mod_dataset_health
import modules.type_detection as mod_type_detection
import modules.analysis_recs as mod_analysis_recs
import modules.data_storytelling as mod_data_storytelling
import modules.ai_recs as mod_ai_recs
import modules.ai_chat as mod_ai_chat
import modules.root_cause as mod_root_cause
import modules.version_history as mod_version_history
import modules.workspace as mod_workspace
import modules.report_generation as mod_report_generation
import modules.export as mod_export

PAGE_MAP = {
    "Dashboard": mod_dashboard.render,
    "Dataset Overview": mod_dataset_overview.render,
    "Raw Dataset": mod_raw_dataset.render,
    "Data Cleaning": mod_data_cleaning.render,
    "Statistical Analysis": mod_statistical_analysis.render,
    "Visualizations": mod_visualizations.render,
    "Correlation Analysis": mod_correlation_analysis.render,
    "Feature Analysis": mod_feature_analysis.render,
    "Custom Analysis": mod_custom_analysis.render,
    "Advanced Analytics": mod_advanced_analytics.render,
    "Predictive Modeling": mod_advanced_analytics.render_prediction_module,
    "All Charts": mod_all_charts.render,
    "AI Insights": mod_ai_insights.render,
    "Auto Dashboard": mod_auto_dashboard.render,
    "Dataset Health": mod_dataset_health.render,
    "Type Detection": mod_type_detection.render,
    "Analysis Recs": mod_analysis_recs.render,
    "Data Storytelling": mod_data_storytelling.render,
    "AI Recs": mod_ai_recs.render,
    "AI Chat": mod_ai_chat.render,
    "Root Cause": mod_root_cause.render,
    "Version History": mod_version_history.render,
    "Workspace": mod_workspace.render,
    "Report Generation": mod_report_generation.render,
    "Export": mod_export.render,
}


def _load_page(page):
    return PAGE_MAP.get(page)


def main():
    init_session()
    inject_css()

    if st.session_state.get("_expanded"):
        if not st.session_state._expanded_fig:
            st.session_state._expanded = False
            st.rerun()
            return
        import plotly.io as pio
        if st.button("✕ Back to Dashboard", key="exp_back", type="primary", use_container_width=True):
            st.session_state._expanded = False
            st.session_state._expanded_fig = None
            st.session_state._expanded_insight = ""
            st.rerun()
        try:
            fig = pio.from_json(st.session_state._expanded_fig)
            fig.update_layout(font=dict(size=16), title_font=dict(size=22), height=850)
        except Exception:
            st.error("Could not render this chart in full screen.")
            st.button("Return to Dashboard", key="exp_back_err", type="primary", on_click=lambda: st.session_state.update(_expanded=False, _expanded_fig=None, _expanded_insight=""))
            return
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True, "displaylogo": False,
            "modeBarButtonsToRemove": ["sendDataToCloud", "lasso2d", "select2d"],
            "toImageButtonOptions": {"format": "png", "filename": "chart", "height": 1000, "width": 1600},
            "scrollZoom": True,
        })
        if st.session_state._expanded_insight:
            st.markdown('<div style="background:#f8fafc;border-left:5px solid #6366f1;border-radius:0 14px 14px 0;padding:1.1rem 1.4rem;font-size:1.15rem;color:#1e293b;line-height:1.7;box-shadow:0 4px 14px rgba(15,23,42,0.04);">💡 ' + st.session_state._expanded_insight + '</div>', unsafe_allow_html=True)
        return

    if st.session_state.get("_view_all"):
        if not st.session_state._view_all_figs:
            st.session_state._view_all = False
            st.rerun()
            return
        import plotly.io as pio
        st.markdown('<div style="display:flex;align-items:center;gap:0.75rem;padding:0.8rem 0;"><span style="font-size:1.8rem;">🖥️</span><span style="font-weight:900;font-size:1.6rem;color:#0f172a;">All Charts — Full Screen & Zoom View</span></div>', unsafe_allow_html=True)
        if st.button("✕ Back to Dashboard", key="vall_back", type="primary"):
            st.session_state._view_all = False
            st.session_state._view_all_figs = []
            st.rerun()
        for idx, j in enumerate(st.session_state._view_all_figs):
            try:
                fig = pio.from_json(j)
                fig.update_layout(font=dict(size=16), title_font=dict(size=22), height=850)
                st.plotly_chart(fig, use_container_width=True, config={
                    "displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["sendDataToCloud", "lasso2d", "select2d"],
                    "toImageButtonOptions": {"format": "png", "filename": f"chart_{idx}", "height": 1000, "width": 1600},
                    "scrollZoom": True,
                })
            except Exception:
                st.caption(f"Could not render chart {idx + 1}")
        return

    sidebar_nav()
    with st.spinner(f"Loading {st.session_state.page}..."):
        render_fn = _load_page(st.session_state.page)
        if render_fn:
            render_fn()


if __name__ == "__main__":
    main()

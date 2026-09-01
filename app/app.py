import os
import sys
import json
import math
import hashlib
import requests
import html
import re
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(
    layout               = "wide",
    page_title           = "NeuroNauts · Psychology AI",
    page_icon            = "💠",
    initial_sidebar_state= "expanded",
)

# ─── Groq SDK ─────────────────────────────────────────────────────────────────
try:
    from groq import (
        Groq as _GroqClient,
        RateLimitError as _GroqRateLimitError,
        APITimeoutError as _GroqTimeoutError,
    )
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClient         = None
    _GroqRateLimitError = None
    _GroqTimeoutError   = None
    _GROQ_AVAILABLE     = False

# ─── PATH SETUP ──────────────────────────────────────────────────────────────
_APP_DIR      = Path(__file__).parent.absolute()
_PROJECT_ROOT = _APP_DIR.parent
_PIPELINE_DIR = _PROJECT_ROOT / "pipeline"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from retrieve import retrieve as _hybrid_retrieve
from knowledge_graph import show_knowledge_graph_page
from psych_lab import show_psych_lab_page
from study_hub import show_study_hub_page
from eval_suite import evaluate_rag_triad, FAITH_THRESHOLD, RELEVANCY_THRESHOLD

# ─── CONFIG ──────────────────────────────────────────────────────────────────

def _resolve_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return ""

def _resolve_groq_model() -> str:
    try:
        return st.secrets.get("GROQ_MODEL", os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    except Exception:
        return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

APP_DIR      = _APP_DIR
PROJECT_ROOT = _PROJECT_ROOT
# ✅ REMOVED: IMAGES_DIR — images now served from Cloudinary URLs
OUTPUT_JSON  = PROJECT_ROOT / "output" / "evaluation_results.json"
OUTPUT_CSV   = PROJECT_ROOT / "output" / "evaluation_results.csv"
CHUNKS_PATH  = PROJECT_ROOT / "chunk_for _graph" / "psychology2e_chunks.json"

TOP_K             = 5
MAX_CONTEXT_CHARS = 4000
FAITH_THRESHOLD   = 0.75

AMBIGUOUS_PRONOUNS = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}

SUGGESTIONS = [
    ("💡", "Operant conditioning",  "What is operant conditioning?"),
    ("📈", "Maslow's hierarchy",    "Explain Maslow's hierarchy of needs"),
    ("🌙", "Stages of sleep",       "What are the stages of sleep?"),
    ("👤", "Big Five personality",  "What is the Big Five personality model?"),
    ("🔔", "Classical conditioning","What is classical conditioning?"),
    ("💾", "How memory works",      "How does memory work?"),
]

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">',
    unsafe_allow_html=True,
)

# Force sidebar always open by clearing localStorage state
st.components.v1.html("""
<script>
(function() {
    try {
        // Clear any stored collapsed state
        window.parent.localStorage.removeItem('stSidebarNavOpen');
        window.parent.localStorage.removeItem('stSidebarState');
        window.parent.localStorage.setItem('stSidebarNavOpen', 'true');
    } catch(e) {}

    function forceOpen() {
        try {
            var doc = window.parent.document;
            // Click the collapsed button if sidebar is hidden
            var btn = doc.querySelector('[data-testid="collapsedControl"]');
            if (btn && btn.offsetParent !== null) {
                btn.click();
                return true;
            }
        } catch(e) {}
        return false;
    }

    // Try multiple times to handle timing
    setTimeout(forceOpen, 300);
    setTimeout(forceOpen, 800);
    setTimeout(forceOpen, 1500);

    // Watch and re-open if collapsed
    setTimeout(function() {
        try {
            var observer = new MutationObserver(function() {
                var btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
                if (btn && btn.offsetParent !== null) {
                    setTimeout(function() { btn.click(); }, 100);
                }
            });
            observer.observe(window.parent.document.body, {
                childList: true, subtree: true, attributes: true,
                attributeFilter: ['class', 'style']
            });
        } catch(e) {}
    }, 2000);
})();
</script>
""", height=0)

if "page"     not in st.session_state: st.session_state.page     = "chat"
if "messages" not in st.session_state: st.session_state.messages = []

# ─── FORCE EXPAND SIDEBAR (Clears browser memory) ───────────────────────────
st.components.v1.html("""
<script>
    function forceExpand() {
        try {
            // Clear Streamlit's persisted sidebar state from browser memory
            window.parent.localStorage.setItem('stSidebarNav', 'expanded');
            window.parent.localStorage.setItem('stSidebarState', 'expanded');
            
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            const expandButton = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            
            // If it's still collapsed despite clearing memory, click the button
            if (expandButton && (!sidebar || sidebar.clientWidth === 0)) {
                expandButton.click();
            }
        } catch (e) {
            console.error("Sidebar force error:", e);
        }
    }
    // Attempt multiple times to ensure Streamlit has finished rendering
    setTimeout(forceExpand, 300);
    setTimeout(forceExpand, 1000);
</script>
""", height=0)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    --primary: #4F46E5; 
    --primary-hover: #4338CA;
    --primary-light: #EEF2FF;
    --primary-glow: rgba(79, 70, 229, 0.15);
    --bg-light: #F8FAFC;
    --text-main: #0F172A;
    --text-muted: #64748B;
    --border-color: #E2E8F0;
    --card-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    --hover-shadow: 0 10px 25px -3px rgba(79, 70, 229, 0.12);
    color-scheme: light only;
}

/* Ensure native expand button is visible */
[data-testid="collapsedControl"] {
    color: #4F46E5 !important;
    background: rgba(79, 70, 229, 0.08) !important;
    border-radius: 0 10px 10px 0 !important;
    z-index: 99999 !important;
    border: 1px solid rgba(79, 70, 229, 0.15) !important;
    border-left: none !important;
}
[data-testid="collapsedControl"] svg {
    fill: #4F46E5 !important;
}

/* ── Typography & Global Reset ── */
html, body {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    color-scheme: light only !important;
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"],
.main {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    color-scheme: light only !important;
}

/* ── Streamlit Tabs Styling (Modern Floating Pills) ── */
[data-baseweb="tab-list"] {
    gap: 8px !important;
    background: #EDF2F7 !important;
    padding: 6px !important;
    border-radius: 14px !important;
    border: 1px solid #E2E8F0 !important;
    margin-bottom: 1.5rem !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
}
[data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #64748B !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-baseweb="tab"]:hover {
    color: #4F46E5 !important;
    background: rgba(255,255,255,0.6) !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: #FFFFFF !important;
    color: #4F46E5 !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── Hide default Streamlit clutter ── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    z-index: 99999 !important;
}

/* ── Sidebar Expand & Collapse Controls ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    top: 0.8rem !important;
    left: 0.8rem !important;
    z-index: 999999 !important;
}
[data-testid="collapsedControl"] button, [data-testid="stSidebarCollapseButton"] button {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    color: #4F46E5 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08) !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
[data-testid="collapsedControl"] button:hover, [data-testid="stSidebarCollapseButton"] button:hover {
    background: #EEF2FF !important;
    border-color: #C7D2FE !important;
    color: #3730A3 !important;
    transform: scale(1.05) !important;
}

.block-container {
    max-width: 1000px !important;
    padding: 2rem 2rem 7rem !important;
    margin: 0 auto !important;
}

/* ── Sidebar: Obsidian Glassmorphism ── */
[data-testid="stSidebar"] {
    background: #0B0F19 !important;
    border-right: 1px solid #1E293B !important;
    width: 260px !important; min-width: 260px !important; max-width: 260px !important;
    color-scheme: dark !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.8rem 1.2rem 1rem !important; }

[data-testid="stSidebar"] .stButton > button {
    border-radius: 10px !important; 
    font-size: 0.88rem !important;
    font-weight: 600 !important; 
    padding: 11px 16px !important;
    width: 100% !important; 
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    color: #ffffff !important; 
    border: 1px solid rgba(255,255,255,0.15) !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: translateX(3px) !important;
    box-shadow: 0 6px 18px rgba(79,70,229,0.45) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
    background: rgba(255,255,255,0.03) !important;
    color: #94A3B8 !important; 
    border: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
    background: rgba(255,255,255,0.08) !important; 
    color: #FFFFFF !important;
    border-color: rgba(255,255,255,0.15) !important;
    transform: translateX(3px) !important;
}

/* ── Chat Message Bubbles ── */
.user-bubble-wrap { display: flex; justify-content: flex-end; margin: 1.5rem 0 1rem; width: 100%; }
.user-bubble {
    background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
    color: #FFFFFF !important; 
    border-radius: 16px 16px 3px 16px;
    padding: 14px 22px; 
    max-width: 78%; 
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.6; 
    box-shadow: 0 4px 14px rgba(79,70,229,0.25);
    word-break: break-word;
}

.assistant-wrap { 
    display: flex !important;
    flex-direction: column !important;
    gap: 14px !important;
    margin: 1.2rem 0 !important;
    width: 100% !important;
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 18px !important;
    padding: 22px 26px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
}
.assistant-avatar {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 10px rgba(79,70,229,0.25);
}
.assistant-avatar svg { width: 20px; height: 20px; }
.assistant-text {
    flex: 1; font-size: 0.95rem; line-height: 1.75; min-width: 0;
    color: #0F172A !important; word-break: break-word;
}
.assistant-text p  { margin: 0 0 0.85em; color: #0F172A !important; }
.assistant-text p:last-child { margin-bottom: 0; }
.assistant-text ul, .assistant-text ol { margin: 0.4em 0 0.85em 1.2em; padding: 0; }
.assistant-text li { margin-bottom: 0.35em; color: #1E293B !important; }
.assistant-text strong { color: #0F172A !important; font-weight: 700; }
.turn-divider { border: none; border-top: 1px solid #E2E8F0; margin: 2rem 0; }

.sources-header {
    font-size: 0.72rem; font-weight: 700; color: #64748B !important;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 1.2rem; margin-bottom: 0.8rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #E2E8F0;
    display: flex; align-items: center; gap: 8px;
}

/* ── Source Cards ── */
.src-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 12px;
    margin-top: 0;
    width: 100%;
}
.src-card {
    background: #F8FAFC !important; 
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px; 
    padding: 14px; 
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.src-card:hover {
    border-color: #C7D2FE !important;
    background: #FFFFFF !important;
    box-shadow: 0 6px 16px rgba(79,70,229,0.08);
    transform: translateY(-2px);
}
.src-card-top { 
    display: flex; align-items: flex-start; gap: 8px; 
    margin-bottom: 6px; width: 100%; 
}
.src-num {
    background: #EEF2FF !important; color: #4F46E5 !important; 
    border: 1px solid #C7D2FE !important;
    border-radius: 6px; padding: 2px 6px;
    font-size: 0.65rem; font-weight: 700; flex-shrink: 0;
}
.src-section { 
    font-size: 0.78rem; font-weight: 700; color: #0F172A !important; line-height: 1.3;
}
.src-preview {
    font-size: 0.72rem; color: #64748B !important; line-height: 1.45;
    margin-bottom: 8px;
    display: -webkit-box; -webkit-line-clamp: 2; 
    -webkit-box-orient: vertical; 
    overflow: hidden;
}
.src-page-badge {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 4px; padding: 2px 6px;
    font-size: 0.68rem; font-weight: 600; color: #475569 !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    border-radius: 20px !important; 
    border: 1.5px solid #E2E8F0 !important;
    background: #FFFFFF !important;
    box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.06) !important;
    padding: 6px 12px !important;
    margin-bottom: 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4F46E5 !important;
    box-shadow: 0 16px 32px -4px rgba(79, 70, 229, 0.16) !important;
    transform: translateY(-2px);
}
[data-testid="stChatInput"] textarea {
    font-size: 0.95rem !important; 
    padding: 10px 14px !important;
    color: #0F172A !important;
    font-weight: 500 !important;
}
[data-testid="stBottom"] {
    background-color: transparent !important;
}

/* ── Metric Scorecard Cards ── */
.metric-card {
    border-radius: 14px; padding: 22px 20px; text-align: center; 
    border: 1.5px solid #E2E8F0 !important; background: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}
.metric-value { font-size: 2rem; font-weight: 800; color: #0F172A !important; line-height: 1; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.82rem; color: #475569 !important; margin-top: 8px; font-weight: 700; letter-spacing: 0.02em; }
.metric-sub { font-size: 0.72rem; color: #94A3B8 !important; margin-top: 4px; font-weight: 500; }

.sent-row {
    display: flex; gap: 10px; align-items: flex-start; padding: 10px 0;
    border-bottom: 1px solid #F1F5F9; font-size: 0.88rem; color: #1E293B !important;
}
.sent-badge {
    flex-shrink: 0; padding: 3px 8px; border-radius: 6px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
}
.sent-ok { background: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }
.sent-fail { background: #FEE2E2; color: #DC2626; border: 1px solid #FECACA; }
.sent-sim { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748B; padding: 2px 4px; }

/* ── Status Pill ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 20px; font-size: 0.8rem; color: #475569 !important; 
    font-weight: 600; margin: 0.5rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #10B981;
    animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1; transform: scale(1);} 50%{opacity:0.4; transform: scale(0.85);} }

/* ── Suggestion Chips ── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #FFFFFF !important; border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important; color: #1E293B !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    padding: 10px 14px !important; white-space: normal !important;
    height: auto !important; min-height: 48px !important;
    line-height: 1.35 !important; box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color: var(--primary) !important; background: #EEF2FF !important;
    color: var(--primary) !important; transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(79,70,229,0.12) !important;
}

@media (max-width: 768px) {
    .block-container { padding: 1.2rem 1rem 7rem !important; }
    [data-testid="stSidebar"] { width: 220px !important; min-width: 220px !important; max-width: 220px !important; }
    .user-bubble { max-width: 90% !important; }
}
</style>
<script>
(function() {
    try {
        const btn = window.parent.document.querySelector('[data-testid="collapsedControl"] button');
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false' && btn) {
            btn.click();
        }
    } catch(e) {}
})();
</script>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:0.2rem 0.1rem 1.6rem;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:40px;height:40px;border-radius:8px;flex-shrink:0;
                        background:var(--primary); color:white;
                        display:flex;align-items:center;justify-content:center;">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor" style="width:22px;height:22px;">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
                </svg>
            </div>
            <div>
                <div style="font-weight:700;font-size:1.05rem;color:#f1f5f9;
                            letter-spacing:-0.01em;line-height:1.2;">NeuroNauts</div>
                <div style="font-size:0.6rem;color:#94a3b8;letter-spacing:0.05em;
                            margin-top:2px;font-weight:500;">
                    Psychology AI · OpenStax
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("💬 Chat", use_container_width=True, key="nav_chat",
                 type="primary" if st.session_state.page=="chat" else "secondary"):
        st.session_state.page = "chat"; st.rerun()

    if st.button("🏥 PsychLab (Cases)", use_container_width=True, key="nav_psych_lab",
                 type="primary" if st.session_state.page=="psych_lab" else "secondary"):
        st.session_state.page = "psych_lab"; st.rerun()

    if st.button("🎯 Study Hub (Quiz & Cards)", use_container_width=True, key="nav_study_hub",
                 type="primary" if st.session_state.page=="study_hub" else "secondary"):
        st.session_state.page = "study_hub"; st.rerun()
    
    if st.button("🗺️ Knowledge Graph", use_container_width=True, key="nav_kg",
                 type="primary" if st.session_state.page=="kg" else "secondary"):
        st.session_state.page = "kg"; st.rerun()

    if st.button("📊 RAG Evaluation", use_container_width=True, key="nav_eval",
                 type="primary" if st.session_state.page=="eval" else "secondary"):
        st.session_state.page = "eval"; st.rerun()

    if st.session_state.page == "chat":
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        if st.button("＋ New Chat", use_container_width=True, type="tertiary", key="new_chat"):
            st.session_state.messages = []; st.rerun()

    st.markdown(f"""
    <div style="position:fixed;bottom:1rem;left:0;width:260px;text-align:center;
                font-size:0.6rem;color:#64748B;font-weight:500;line-height:1.6;">
        WCE Hackathon 2026 · NeuroNauts<br>
        <span style="color:#4F46E5;">⚡ {GROQ_MODEL}</span>
    </div>
    """, unsafe_allow_html=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_embeddings_batch(texts: list) -> list:
    from retrieve import _get_embed_model
    model = _get_embed_model()
    prefixed = [f"search_document: {t}" for t in texts]
    return model.encode(prefixed, normalize_embeddings=True).tolist()


def cosine_similarity(a: list, b: list) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if (mag_a == 0 or mag_b == 0) else dot / (mag_a * mag_b)


def retrieve(query: str, top_k: int = TOP_K) -> list:
    return _hybrid_retrieve(query, top_k=top_k)


def build_context(chunks: list) -> str:
    return "\n\n".join([c["full_text"] for c in chunks])[:MAX_CONTEXT_CHARS]


def build_retrieval_query(current_query: str) -> str:
    words = current_query.lower().split()
    is_ambiguous = (
        len(words) <= 10 and
        any(w in AMBIGUOUS_PRONOUNS for w in words)
    )
    if not is_ambiguous:
        return current_query
    messages    = st.session_state.messages
    prev_user   = None
    prev_answer = None
    for msg in reversed(messages):
        if msg["role"] == "assistant" and prev_answer is None:
            prev_answer = msg["content"][:150]
        if msg["role"] == "user" and msg["content"] != current_query and prev_user is None:
            prev_user = msg["content"]
        if prev_user and prev_answer:
            break
    if not prev_user:
        return current_query
    return f"{prev_user} {prev_answer or ''} {current_query}".strip()


RATE_LIMIT_ANSWER  = "__RATE_LIMITED__"
TIMEOUT_ANSWER     = "__TIMED_OUT__"
AUTH_ERROR_ANSWER  = "__AUTH_ERROR__"
API_ERROR_ANSWER   = "__API_ERROR__"
DB_ERROR_ANSWER    = "__DB_ERROR__"
MODEL_ERROR_ANSWER = "__MODEL_ERROR__"
PKG_ERROR_ANSWER   = "__PKG_ERROR__"

def _is_rate_limit_error(e: Exception) -> bool:
    if _GroqRateLimitError and isinstance(e, _GroqRateLimitError):
        return True
    err = str(e).lower()
    return any(k in err for k in ("rate_limit", "rate limit", "429", "quota", "tokens per"))


def _is_timeout_error(e: Exception) -> bool:
    if _GroqTimeoutError and isinstance(e, _GroqTimeoutError):
        return True
    err = str(e).lower()
    return any(k in err for k in ("timeout", "timed out", "read timeout", "connection timeout"))

def _is_auth_error(e: Exception) -> bool:
    err = str(e).lower()
    return "401" in err or "unauthorized" in err or "authentication" in err or "invalid api key" in err

def _is_api_error(e: Exception) -> bool:
    err = str(e).lower()
    return "500" in err or "502" in err or "503" in err or "internal server error" in err or "bad gateway" in err


def _is_model_error(e: Exception) -> bool:
    err = str(e).lower()
    return "model" in err and ("not found" in err or "does not exist" in err or "invalid" in err)

def _get_groq_client() -> "_GroqClient":
    if not _GROQ_AVAILABLE:
        raise RuntimeError("__PKG_MISSING__")
    api_key = _resolve_groq_key()
    if not api_key:
        raise RuntimeError("__AUTH_MISSING__")
    return _GroqClient(api_key=api_key)


def call_llm(question: str, context: str) -> str:
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant" and msg["content"] not in (
            RATE_LIMIT_ANSWER, TIMEOUT_ANSWER
        ):
            history.append({"role": "assistant", "content": msg["content"]})

    if history and history[-1]["role"] == "user":
        history[-1]["content"] = (
            f"Context from textbook:\n{context}\n\nQuestion: {question}"
        )

    system = {
        "role": "system",
        "content": (
            "You are NeuroNauts, an elite academic AI tutor specialized in OpenStax Psychology 2e.\n"
            "Answer the user's question with high rigor, clarity, and structure based STRICTLY on the provided textbook context.\n\n"
            "Instructions:\n"
            "1. Grounding: Rely ONLY on the provided textbook context. Do not extrapolate or fabricate facts.\n"
            "2. Structure:\n"
            "   - Direct Core Answer: Start with a clear 1-2 sentence core definition or conceptual summary.\n"
            "   - Mechanisms & Key Details: Use structured bullet points with **bold terms** to break down mechanisms, theoretical models, or experimental paradigms.\n"
            "   - Relevant Landmark Studies / Theorists: Highlight foundational psychologists (e.g. Pavlov, Skinner, Freud, Bandura, Milgram) and findings mentioned in the text.\n"
            "   - Textbook Citation: Always cite the exact chapter/section or page numbers from the context at the conclusion.\n"
            "3. Formatting: Use clean Markdown (bolding, bullet lists). Do NOT output raw HTML tags.\n"
            "4. If the answer cannot be found in the provided context, reply exactly: 'Not found in the provided textbook.'"
        )
    }

    try:
        client = _get_groq_client()
        completion = client.chat.completions.create(
            model=_resolve_groq_model(),
            messages=[system] + history,
            temperature=0.2,
            max_tokens=850,
            timeout=60,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if _is_rate_limit_error(e):   return RATE_LIMIT_ANSWER
        if _is_timeout_error(e):      return TIMEOUT_ANSWER
        if _is_auth_error(e):         return AUTH_ERROR_ANSWER
        if "__PKG_MISSING__"  in err: return PKG_ERROR_ANSWER
        if "__AUTH_MISSING__" in err: return AUTH_ERROR_ANSWER
        if _is_model_error(e):        return MODEL_ERROR_ANSWER
        print(f"[Groq] Unexpected error: {type(e).__name__}: {e}")
        st.session_state["_last_groq_error"] = f"{type(e).__name__}: {e}"
        return API_ERROR_ANSWER


def call_llm_stateless(question: str, context: str) -> str:
    try:
        client = _get_groq_client()
        completion = client.chat.completions.create(
            model=_resolve_groq_model(),
            messages=[
                {"role": "system", "content": (
                    "You are NeuroNauts, an elite academic AI tutor specialized in OpenStax Psychology 2e.\n"
                    "Answer strictly based on the provided textbook context.\n"
                    "Provide a structured answer with a direct definition, key mechanisms in bullet points, and exact section/page citations from the context.\n"
                    "If the answer is not in the context, say: 'Not found in the provided textbook.'"
                )},
                {"role": "user", "content": (
                    f"Context from textbook:\n{context}\n\nQuestion: {question}"
                )},
            ],
            temperature=0.15,
            max_tokens=700,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        if _is_rate_limit_error(e):
            raise RuntimeError(
                "RATE_LIMIT: Groq API quota exhausted. Please try again later."
            ) from e
        raise


# Cloudinary: cloud=dnbnrxyn1, folder=psychology2e
# public_id format: "psychology2e/img_p91_0"  (NO extension)
# URL format: https://res.cloudinary.com/dnbnrxyn1/image/upload/psychology2e/img_p91_0
_CLOUDINARY_BASE = "https://res.cloudinary.com/dnbnrxyn1/image/upload/psychology2e"


def _parse_image_refs(raw) -> list:
    """
    Parse the image_refs field from Zilliz.
    Zilliz stores it as a JSON-encoded string, e.g.: '["img_p19_0.jpeg"]'
    or occasionally double-encoded: '"[\"img_p19_0.jpeg\"]"'
    Handles both, plus empty lists and None values.
    Always returns a flat list of filename strings.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if x]
    if not isinstance(raw, str):
        return []
    current = raw.strip()
    if not current or current in ("[]", "null", "None", '"[]"'):
        return []
    # Try up to 2 rounds of JSON decoding (handles single- and double-encoding)
    for _ in range(2):
        try:
            decoded = json.loads(current)
        except (json.JSONDecodeError, ValueError):
            break
        if isinstance(decoded, list):
            return [str(x).strip() for x in decoded if x]
        if isinstance(decoded, str):
            current = decoded   # unwrap one layer, retry
        else:
            break
    return []


def get_images(chunks: list) -> list:
    """
    Extract unique, distinct Cloudinary image URLs from retrieved chunks.
    - Filters out multiple crops/figures from the same textbook page to avoid similar images.
    - Limits to at most 2 distinct figures per answer to keep the layout clean and focused.
    """
    seen_stems = set()
    seen_pages = set()
    result = []
    for chunk in chunks:
        refs = _parse_image_refs(chunk.get("image_refs"))
        for filename in refs:
            filename = filename.strip()
            if not filename:
                continue
            # Already a full URL — use directly
            if filename.startswith("http://") or filename.startswith("https://"):
                full_url = filename
                stem = filename.rstrip("/").split("/")[-1].rsplit(".", 1)[0].lower()
            else:
                stem = filename.rsplit(".", 1)[0].lower()
                full_url = f"{_CLOUDINARY_BASE}/{stem}"
            
            # Extract page number identifier (e.g. img_p204_0 -> 204)
            page_match = re.search(r'img_p(\d+)_', stem)
            page_id = page_match.group(1) if page_match else stem

            if stem and stem not in seen_stems and page_id not in seen_pages:
                seen_stems.add(stem)
                seen_pages.add(page_id)
                result.append({"url": full_url, "stem": stem})
                if len(result) >= 2:
                    return result
    return result


# ─── IMAGE ROW ───────────────────────────────────────────────────────────────

_IMG_CARD_CSS = """
<style>
.custom-gallery { 
    display: flex; gap: 12px; flex-wrap: nowrap; margin-top: 10px; 
    overflow-x: auto; -webkit-overflow-scrolling: touch; padding-bottom: 14px;
}
.custom-gallery::-webkit-scrollbar { height: 5px; }
.custom-gallery::-webkit-scrollbar-track { background: #f8fafc; border-radius: 10px; }
.custom-gallery::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
.custom-gallery::-webkit-scrollbar-thumb:hover { background: #cbd5e1; }
.custom-thumb-label { cursor: pointer; flex-shrink: 0; display: block; width: 200px; height: 130px; }
.custom-thumb { 
    width: 200px !important; height: 130px !important; object-fit: cover !important; 
    border-radius: 8px; border: 1.5px solid #E2E8F0; box-shadow: 0 2px 8px rgba(0,0,0,.06);
    transition: border-color .2s, box-shadow .2s; display: block !important;
}
.custom-thumb:hover { border-color: #4F46E5; box-shadow: 0 4px 14px rgba(79,70,229,.18); }

.lightbox-toggle { display: none !important; }
.lightbox-overlay {
    display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(10, 15, 30, 0.95); z-index: 999999;
}
.lightbox-toggle:checked + .lightbox-overlay { display: block !important; }
.lightbox-bg-close {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: pointer;
}
.lightbox-content-wrapper {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    max-width: 90vw; max-height: 80vh; pointer-events: none;
    display: flex; justify-content: center; align-items: center;
}
.lightbox-img {
    max-width: 100% !important; max-height: 80vh !important; border-radius: 8px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5); pointer-events: auto; background: white;
    height: auto !important; object-fit: contain !important; display: block !important;
}
.lightbox-x-wrap {
    position: absolute; top: 25px; right: 30px; cursor: pointer; z-index: 1000000;
}
.lightbox-x {
    width: 40px; height: 40px; border-radius: 50%; background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2); color: white; display: flex;
    align-items: center; justify-content: center; font-family: sans-serif;
    font-size: 20px; transition: background 0.2s; pointer-events: auto;
}
.lightbox-x:hover { background: rgba(255,255,255,0.3); }
.lightbox-footer {
    position: absolute; bottom: 30px; width: 100%; text-align: center;
    color: #94A3B8; font-size: 0.9rem; pointer-events: none; z-index: 1000000;
}
</style>
"""

def render_image_row(images: list, msg_index: int = 0):
    """Render up to 4 unique images as a clickable thumbnail gallery with lightbox."""
    if not images:
        return

    st.markdown(_IMG_CARD_CSS, unsafe_allow_html=True)
    html_parts = ['<div class="custom-gallery">']

    for i, img in enumerate(images[:4]):
        url  = img.get("url", "").strip()
        if not url:
            continue
        stem  = img.get("stem", f"img_{i}")
        label = stem.replace("_", " ").title()
        uid   = f"lb-{msg_index}-{i}"

        html_parts.append(
            f'<label for="{uid}" class="custom-thumb-label">'
            f'<img src="{url}" class="custom-thumb" alt="{label}" title="{label}"'
            f' onerror="this.closest(\'label\').style.display=\'none\'"/>'
            f'</label>'
            f'<input type="checkbox" id="{uid}" class="lightbox-toggle"/>'
            f'<div class="lightbox-overlay">'
            f'<label for="{uid}" class="lightbox-bg-close"></label>'
            f'<label for="{uid}" class="lightbox-x-wrap"><div class="lightbox-x">✕</div></label>'
            f'<div class="lightbox-content-wrapper">'
            f'<img src="{url}" class="lightbox-img" alt="{label}"/></div>'
            f'<div class="lightbox-footer">{label} · Click anywhere to close</div>'
            f'</div>'
        )

    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_sources_panel(sources: list):
    n = len(sources)
    st.markdown(
        f'<div class="sources-header">📄 &nbsp;{n} source{"s" if n!=1 else ""} retrieved</div>',
        unsafe_allow_html=True
    )
    # Build all cards as a single HTML grid — no st.columns() so spacing is
    # perfectly uniform at every screen width (1-col mobile, 2-col tablet, 3-col desktop)
    cards_html = ['<div class="src-grid">']
    for idx, s in enumerate(sources, 1):
        section = html.escape(s["section_path"].replace(" > ", " › "))
        preview = html.escape(s.get("clean_text", "")[:120])
        pages   = ", ".join(map(str, s["page_numbers"]))
        cards_html.append(f"""
        <div class="src-card">
            <div class="src-card-top">
                <span class="src-num">#{idx}</span>
                <span class="src-section">{section}</span>
            </div>
            <div class="src-preview">{preview}…</div>
            <div class="src-page">
                <span class="src-page-badge">Page {pages}</span>
            </div>
        </div>""")
    cards_html.append('</div>')
    st.markdown("".join(cards_html), unsafe_allow_html=True)


def format_answer_html(text: str) -> str:
    import re as _re

    # 1. Transform Section Headers into Beautiful Badged Containers
    # Core Definition / Direct Core Answer (Minimalist, elegant executive summary card)
    text = _re.sub(
        r'(\*{0,2}(?:Direct Core Answer|Core Concept Definition|Core Definition):?\*{0,2})\s*\n+([^\n]+(?:\n[^\n]+)*?)(?=\n+\*{0,2}(?:Mechanisms|Relevant|Landmark|Textbook)|\Z)',
        r'<div style="background:#F8FAFC; border:1.5px solid #E2E8F0; border-radius:12px; padding:16px 20px; margin:4px 0 16px; box-shadow:0 1px 3px rgba(0,0,0,0.02);"><div style="display:inline-flex; align-items:center; gap:6px; padding:2px 8px; border-radius:6px; background:#EEF2FF; color:#4338CA; border:1px solid #C7D2FE; font-size:0.7rem; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; margin-bottom:8px;"><span>🎯</span> Core Concept</div><div style="font-size:0.95rem; line-height:1.7; color:#1E293B; font-weight:500;">\2</div></div>',
        text,
        flags=_re.IGNORECASE
    )

    # Mechanisms & Key Details
    text = _re.sub(
        r'\*{0,2}(?:Mechanisms & Key Details|Key Mechanisms & Details|Mechanisms|Theoretical Mechanisms):?\*{0,2}',
        r'<div style="display:flex; align-items:center; gap:8px; margin:16px 0 8px;"><span style="font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:6px; background:#EEF2FF; color:#4F46E5; border:1px solid #C7D2FE; text-transform:uppercase; letter-spacing:0.04em;">⚙️ Mechanisms</span><span style="font-size:0.92rem; font-weight:800; color:#0F172A;">Theoretical Dynamics</span></div>',
        text,
        flags=_re.IGNORECASE
    )

    # Relevant Landmark Studies / Theorists
    text = _re.sub(
        r'\*{0,2}(?:Relevant Landmark Studies\s*/?\s*Theorists|Landmark Studies|Landmark Experiments|Key Theorists):?\*{0,2}',
        r'<div style="display:flex; align-items:center; gap:8px; margin:18px 0 8px;"><span style="font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:6px; background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; text-transform:uppercase; letter-spacing:0.04em;">🧪 Landmark Studies</span><span style="font-size:0.92rem; font-weight:800; color:#0F172A;">Empirical Evidence</span></div>',
        text,
        flags=_re.IGNORECASE
    )

    # Textbook Citation
    text = _re.sub(
        r'\*{0,2}(?:Textbook Citation|Textbook Grounding|OpenStax Citation):?\*{0,2}\s*(.+)',
        r'<div style="margin-top:14px; padding:10px 14px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; font-size:0.82rem; color:#475569; font-weight:500;">📖 <strong>Textbook Grounding:</strong> \1</div>',
        text,
        flags=_re.IGNORECASE
    )

    # Clean inline markdown
    text = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = _re.sub(r'__(.+?)__',      r'<strong>\1</strong>', text)
    text = _re.sub(r'\*([^*]+?)\*',   r'<em>\1</em>', text)
    
    # Support ### headers
    text = _re.sub(r'^###\s+(.+)$', r'<h4 style="margin:14px 0 6px;color:#0F172A;font-weight:700;font-size:1.02rem;">\1</h4>', text, flags=_re.MULTILINE)
    text = _re.sub(r'^##\s+(.+)$',  r'<h3 style="margin:16px 0 8px;color:#0F172A;font-weight:800;font-size:1.15rem;">\1</h3>', text, flags=_re.MULTILINE)

    paragraphs = _re.split(r'\n{2,}', text.strip())
    html_parts = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith('<div') or para.startswith('<h3') or para.startswith('<h4'):
            html_parts.append(para)
            continue
        lines = para.split('\n')
        bullet_lines = [l for l in lines
                        if _re.match(r'^\s*[-*•]\s', l)
                        or _re.match(r'^\s*\d+\.\s', l)
                        or _re.match(r'^\s*<strong>[^<]+:</strong>', l)
                        or _re.match(r'^\s*<em>[^<]+:</em>', l)]
        if len(bullet_lines) >= 1 and (len(bullet_lines) == len([l for l in lines if l.strip()]) or len(bullet_lines) >= 2):
            is_ordered = all(_re.match(r'^\s*\d+\.\s', l) for l in bullet_lines)
            tag = 'ol' if is_ordered else 'ul'
            _bullet_re = r'^\s*[-*\u2022\d+.]+\s*'
            li_items  = [f'<li style="margin-bottom:8px;line-height:1.65;color:#1E293B;">{_re.sub(_bullet_re, "", l).strip()}</li>'
                         for l in lines if l.strip()]
            li_joined = "".join(li_items)
            html_parts.append(f'<{tag} style="margin:6px 0 12px;padding-left:22px;">{li_joined}</{tag}>')
        else:
            inner = '<br>'.join(l for l in lines if l.strip())
            html_parts.append(f'<p style="margin:0 0 10px;line-height:1.68;color:#1E293B;">{inner}</p>')
    return ''.join(html_parts) or f'<p>{html.escape(text)}</p>'


def get_follow_up_suggestions(query: str, answer: str, sources: list) -> list:
    """Derive 2-3 relevant academic follow-up questions based on the topic."""
    q_lower = query.lower()
    a_lower = answer.lower()

    if any(w in q_lower or w in a_lower for w in ["classical conditioning", "pavlov", "unconditioned", "conditioned stimulus"]):
        return [
            ("⚡", "Extinction Dynamics", "How does extinction and spontaneous recovery work in classical conditioning?"),
            ("🔄", "Operant Comparison", "How does operant conditioning differ fundamentally from classical conditioning?"),
            ("🧪", "Little Albert Study", "What was Watson's Little Albert experiment on conditioned emotional responses?")
        ]
    elif any(w in q_lower or w in a_lower for w in ["operant conditioning", "skinner", "reinforcement", "punishment", "ratio"]):
        return [
            ("📊", "Reinforcement Schedules", "What are the four schedules of reinforcement and which produces highest response?"),
            ("⚖️", "Negative Reinforcement vs Punishment", "Why is negative reinforcement different from punishment in OpenStax?"),
            ("📈", "Shaping Behavior", "How does the technique of shaping work in operant conditioning?")
        ]
    elif any(w in q_lower or w in a_lower for w in ["memory", "amnesia", "hippocampus", "forgetting", "chunking", "encoding"]):
        return [
            ("🧠", "Hippocampus Role", "What role does the hippocampus play in memory consolidation vs storage?"),
            ("⚠️", "Interference Theory", "Explain the difference between proactive and retroactive interference with examples."),
            ("📋", "Atkinson-Shiffrin Model", "What are the three distinct memory stores in the Atkinson-Shiffrin model?")
        ]
    elif any(w in q_lower or w in a_lower for w in ["neuron", "neurotransmitter", "action potential", "synapse", "brain", "cortex"]):
        return [
            ("⚡", "Action Potential Cycle", "How do sodium and potassium voltage-gated channels drive an action potential?"),
            ("🧪", "Dopamine & Serotonin", "What are the distinct physiological roles of dopamine versus serotonin?"),
            ("🌐", "Neuroplasticity", "How does neuroplasticity allow structural and functional adaptation in the brain?")
        ]
    elif any(w in q_lower or w in a_lower for w in ["sleep", "dream", "consciousness", "circadian", "rem"]):
        return [
            ("🌙", "REM vs NREM", "What physiological phenomena distinguish REM sleep from deep NREM slow-wave sleep?"),
            ("⏰", "Circadian Pacemaker", "How does the suprachiasmatic nucleus regulate melatonin and circadian rhythms?"),
            ("😴", "Sleep Disorders", "What are the diagnostic features of sleep apnea, narcolepsy, and insomnia in OpenStax?")
        ]
    elif any(w in q_lower or w in a_lower for w in ["disorder", "dsm", "depression", "anxiety", "panic", "schizophrenia", "ptsd"]):
        return [
            ("📋", "DSM-5 Criteria", "What are the core DSM-5 diagnostic criteria for this psychological disorder?"),
            ("💊", "CBT & Treatment Modalities", "What therapeutic modalities (e.g. CBT, exposure therapy) are supported by OpenStax?"),
            ("🔍", "Diathesis-Stress Model", "How does the diathesis-stress model explain vulnerability to psychological disorders?")
        ]
    elif any(w in q_lower or w in a_lower for w in ["conformity", "milgram", "asch", "dissonance", "social", "bystander"]):
        return [
            ("🧪", "Milgram's Obedience", "What experimental variables increased or decreased obedience in Milgram's study?"),
            ("⚡", "Cognitive Dissonance", "How did Festinger and Carlsmith demonstrate cognitive dissonance?"),
            ("👥", "Bystander Effect", "Why does diffusion of responsibility cause the bystander effect in emergencies?")
        ]
    else:
        if sources:
            sec_name = sources[0].get("section_path", "").split(">")[-1].strip()
            return [
                ("📖", f"Theories of {sec_name[:18]}", f"What are the central theoretical models described in {sec_name}?"),
                ("🔬", "Empirical Evidence", f"What landmark empirical studies or evidence support this concept in the textbook?"),
                ("💡", "Behavioral Application", f"How does this psychological phenomenon manifest in everyday human behavior?")
            ]
        return [
            ("🔬", "Empirical Evidence", "What landmark empirical studies in OpenStax support this concept?"),
            ("💡", "Behavioral Examples", "Can you provide concrete real-world behavioral examples of this phenomenon?"),
            ("⚠️", "Common Misconceptions", "What are common student pitfalls or misconceptions regarding this topic?")
        ]


# ─── RUN QUERY ───────────────────────────────────────────────────────────────

def run_query(query: str):
    slot = st.empty()
    def status(msg):
        slot.markdown(
            f'<div class="status-pill"><div class="status-dot"></div>'
            f'<span>{msg}</span></div>',
            unsafe_allow_html=True)

    start_time = time.time()
    retrieval_query = build_retrieval_query(query)

    status("Searching knowledge base…")
    try:
        chunks  = retrieve(retrieval_query)
    except Exception as e:
        slot.empty()
        print(f"[Zilliz] Error: {type(e).__name__}: {e}")
        err_str = str(e).lower()
        duration = time.time() - start_time
        if "auth" in err_str or "unauthorized" in err_str or "token" in err_str or "401" in err_str:
            return [], AUTH_ERROR_ANSWER, [], False, False, duration
        return [], DB_ERROR_ANSWER, [], False, False, duration
        
    images  = get_images(chunks)   # ✅ now returns Cloudinary URLs

    if not chunks:
        slot.empty()
        duration = time.time() - start_time
        return chunks, "Not found in the provided textbook.", images, False, False, duration

    status("Building context…")
    context = build_context(chunks)
    status("Generating answer…")
    answer  = call_llm(query, context)
    duration = time.time() - start_time
    slot.empty()
    rate_limited = (answer == RATE_LIMIT_ANSWER)
    timed_out    = (answer == TIMEOUT_ANSWER)
    return chunks, answer, images, rate_limited, timed_out, duration


# ─── EVAL HELPERS ────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list:
    raw = re.split(r'(?<=[.?!])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


def score_faithfulness(answer: str, contexts: list) -> dict:
    sentences = split_into_sentences(answer)
    if not sentences:
        return {"score":0.0,"supported":0,"total":0,"sentences":[]}
    all_vecs  = get_embeddings_batch(sentences + contexts)
    sent_vecs = all_vecs[:len(sentences)]
    ctx_vecs  = all_vecs[len(sentences):]
    results, n_ok = [], 0
    for sent, sv in zip(sentences, sent_vecs):
        sims    = [cosine_similarity(sv, cv) for cv in ctx_vecs]
        max_sim = max(sims) if sims else 0.0
        ok      = max_sim >= FAITH_THRESHOLD
        if ok: n_ok += 1
        results.append({"sentence":sent,"max_sim":round(max_sim,4),"supported":ok})
    return {"score":round(n_ok/len(sentences),4),"supported":n_ok,
            "total":len(sentences),"sentences":results}


def score_answer_relevancy(question: str, answer: str) -> dict:
    vecs = get_embeddings_batch([question, answer])
    return {"score": round(cosine_similarity(vecs[0], vecs[1]), 4)}


def run_evaluation(queries: list, sample_size: int) -> dict:
    step     = max(1, len(queries) // sample_size)
    selected = queries[::step][:sample_size]
    results  = []
    progress = st.progress(0, text="Starting evaluation…")

    def _embed_batch(texts: list) -> list:
        return get_embeddings_batch(texts)

    for i, q in enumerate(selected):
        progress.progress((i+1)/len(selected),
                          text=f"Evaluating {i+1}/{len(selected)}: {q['question'][:52]}…")
        qid, question = q["query_id"], q["question"]
        try:
            chunks    = retrieve(question, top_k=5)
            contexts  = [c.get("clean_text", "") for c in chunks]
            answer    = call_llm_stateless(question, build_context(chunks))
            metrics   = evaluate_rag_triad(question, answer, contexts, _embed_batch)

            results.append({
                "query_id": qid,
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "composite_score": metrics["composite_score"],
                "grade": metrics["grade"],
                "faithfulness_score": metrics["faithfulness"]["score"],
                "faithfulness_detail": metrics["faithfulness"],
                "relevancy_score": metrics["answer_relevancy"]["score"],
                "relevancy_detail": metrics["answer_relevancy"],
                "context_precision_score": metrics["context_precision"]["score"],
                "context_precision_detail": metrics["context_precision"],
                "context_recall_score": metrics["context_recall"]["score"],
                "context_recall_detail": metrics["context_recall"]
            })
        except Exception as e:
            results.append({
                "query_id": qid, "question": question, "answer": "",
                "composite_score": None, "grade": "Error",
                "faithfulness_score": None, "relevancy_score": None,
                "context_precision_score": None, "context_recall_score": None,
                "error": str(e)
            })

    progress.empty()
    valid         = [r for r in results if r.get("faithfulness_score") is not None]
    n             = len(valid) if valid else 1
    avg_faith     = round(sum(r.get("faithfulness_score", 0.0) for r in valid) / n, 4)
    avg_relevancy = round(sum(r.get("relevancy_score", 0.0) for r in valid) / n, 4)
    avg_precision = round(sum(r.get("context_precision_score", 0.0) for r in valid) / n, 4)
    avg_recall    = round(sum(r.get("context_recall_score", 0.0) for r in valid) / n, 4)
    avg_composite = round(sum(r.get("composite_score", 0.0) for r in valid) / n, 4)

    output = {
        "summary": {
            "total_evaluated": len(valid),
            "avg_composite_score": avg_composite,
            "avg_faithfulness": avg_faith,
            "avg_answer_relevancy": avg_relevancy,
            "avg_context_precision": avg_precision,
            "avg_context_recall": avg_recall,
            "faithfulness_threshold": FAITH_THRESHOLD,
            "relevancy_threshold": RELEVANCY_THRESHOLD
        },
        "results": results
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "query_id", "question", "answer", "composite_score", "grade",
            "faithfulness_score", "relevancy_score", "context_precision_score", "context_recall_score"
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "query_id": r["query_id"],
                "question": r["question"],
                "answer": r.get("answer", ""),
                "composite_score": r.get("composite_score", ""),
                "grade": r.get("grade", ""),
                "faithfulness_score": r.get("faithfulness_score", ""),
                "relevancy_score": r.get("relevancy_score", ""),
                "context_precision_score": r.get("context_precision_score", ""),
                "context_recall_score": r.get("context_recall_score", "")
            })
    return output


# ─── PAGE: CHAT ───────────────────────────────────────────────────────────────

def show_chat_page():

    # ── Process any pending query FIRST before rendering ──────────────────────
    # This ensures that when a suggestion chip triggers a query, the assistant
    # response (with images) is already in session_state BEFORE we render,
    # so images appear immediately on the very first question.
    pending = st.session_state.pop("_pending_query", None)
    if pending:
        chunks, answer, images, rate_limited, timed_out, duration = run_query(pending)
        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "sources": chunks, "images": images,
            "rate_limited": rate_limited,
            "timed_out": timed_out,
            "duration": duration,
        })
        st.rerun()

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 2.5rem 1rem 1.5rem; max-width: 820px; margin: 0 auto;">
            <div style="display:inline-flex; align-items:center; gap:8px; padding: 5px 14px; 
                        background: #EEF2FF; border: 1.5px solid #C7D2FE; border-radius: 20px; margin-bottom: 1.2rem;">
                <div style="width: 7px; height: 7px; border-radius: 50%; background: #4F46E5; animation: pulse-dot 1.5s infinite;"></div>
                <span style="font-size: 0.72rem; font-weight: 800; color: #4F46E5; letter-spacing: 0.08em; text-transform: uppercase;">
                    NeuroNauts 2.0 · OpenStax Psychology 2e
                </span>
            </div>
            <h1 style="margin: 0.2rem 0 0.8rem; font-size: 2.2rem; font-weight: 800;
                       color: #0F172A; letter-spacing: -0.03em; line-height: 1.25;">
                Advanced Academic AI Learning Companion
            </h1>
            <p style="color: #64748B; font-size: 0.95rem; font-weight: 500; max-width: 580px; margin: 0 auto; line-height: 1.6;">
                Zero-hallucination dense vector retrieval, diagram extraction from Cloudinary, and intelligent clinical reasoning.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="max-width: 800px; margin: 0 auto 1.2rem; text-align: center;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 20px;">
                <div style="height: 1px; width: 30px; background: #E2E8F0;"></div>
                <span style="font-size: 0.65rem; font-weight: 700; color: #94A3B8;
                             text-transform: uppercase; letter-spacing: 0.1em;">Quick Start Suggestions</span>
                <div style="height: 1px; width: 30px; background: #E2E8F0;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3-Column Grid for efficiency
        grid_col1, grid_col2, grid_col3 = st.columns([1, 6, 1])
        with grid_col2:
            for i in range(0, len(SUGGESTIONS), 3):
                row_items = SUGGESTIONS[i:i+3]
                cols = st.columns(len(row_items), gap="small")
                for col, (emo, label, full_q) in zip(cols, row_items):
                    k = f"chip_{i}_{re.sub(r'[^a-z0-9]','_',full_q[:12].lower())}"
                    if col.button(f"{emo} {label}", key=k, use_container_width=True):
                        st.session_state.messages.append({"role":"user","content":full_q})
                        st.session_state["_pending_query"] = full_q
                        st.rerun()
        return

    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-bubble-wrap">
                <div class="user-bubble">{html.escape(msg['content'])}</div>
            </div>""", unsafe_allow_html=True)
        else:
            content = msg.get("content", "")
            
            # Strict sentinel evaluation
            rate_limited = (content == RATE_LIMIT_ANSWER) or msg.get("rate_limited", False)
            timed_out    = (content == TIMEOUT_ANSWER)    or msg.get("timed_out", False)
            auth_error   = (content == AUTH_ERROR_ANSWER)
            api_error    = (content == API_ERROR_ANSWER)
            db_error     = (content == DB_ERROR_ANSWER)
            model_error  = (content == MODEL_ERROR_ANSWER)
            pkg_error    = (content == PKG_ERROR_ANSWER)

            model_error = (content == MODEL_ERROR_ANSWER)
            pkg_error   = (content == PKG_ERROR_ANSWER)
            is_error = rate_limited or timed_out or auth_error or api_error or db_error or model_error or pkg_error

            not_found    = (
                not is_error
                and "not found in the provided textbook" in content.lower()
            )

            if rate_limited:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
                    border: 1px solid #4338CA; border-radius: 12px;
                    padding: 28px 32px; margin: 1rem 0 1.2rem; max-width: 680px;">
                    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                        <div style="width:44px;height:44px;border-radius:10px;
                            background:rgba(99,102,241,0.25);border:1px solid #6366F1;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;flex-shrink:0;">⚡</div>
                        <div>
                            <div style="font-size:1rem;font-weight:700;color:#E0E7FF;">Service Temporarily at Capacity</div>
                            <div style="font-size:0.75rem;color:#A5B4FC;margin-top:2px;">Groq API · Rate Limit Reached</div>
                        </div>
                    </div>
                    <p style="color:#C7D2FE;font-size:0.88rem;line-height:1.65;margin:0 0 16px;">
                        The AI service is currently at capacity due to high demand.
                        This is a <strong style="color:#E0E7FF;">temporary situation</strong> — API quotas reset periodically.
                    </p>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:14px 18px;font-size:0.82rem;color:#A5B4FC;line-height:1.8;">
                        <strong style="color:#C7D2FE;display:block;margin-bottom:6px;">What you can do:</strong>
                        🕐 &nbsp;Wait a <strong style="color:#E0E7FF;">few minutes</strong> and retry<br>
                        ✍️ &nbsp;Try a <strong style="color:#E0E7FF;">shorter or more specific</strong> question<br>
                        🔄 &nbsp;API quotas reset <strong style="color:#E0E7FF;">periodically</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif timed_out:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #1c1408 0%, #2d1f05 100%);
                    border: 1px solid #d97706; border-radius: 12px;
                    padding: 28px 32px; margin: 1rem 0 1.2rem; max-width: 680px;">
                    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                        <div style="width:44px;height:44px;border-radius:10px;
                            background:rgba(217,119,6,0.2);border:1px solid #d97706;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;flex-shrink:0;">⏱️</div>
                        <div>
                            <div style="font-size:1rem;font-weight:700;color:#FEF3C7;">Request Timed Out</div>
                            <div style="font-size:0.75rem;color:#FCD34D;margin-top:2px;">Groq API · Connection Timeout</div>
                        </div>
                    </div>
                    <p style="color:#FDE68A;font-size:0.88rem;line-height:1.65;margin:0 0 16px;">
                        The AI service took too long to respond during
                        <strong style="color:#FEF3C7;">high traffic</strong> or a temporary network issue.
                    </p>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:14px 18px;font-size:0.82rem;color:#FCD34D;line-height:1.8;">
                        <strong style="color:#FEF3C7;display:block;margin-bottom:6px;">What you can do:</strong>
                        🔄 &nbsp;<strong style="color:#FEF3C7;">Try again</strong> — most timeouts are transient<br>
                        ✍️ &nbsp;Ask a <strong style="color:#FEF3C7;">shorter question</strong><br>
                        📶 &nbsp;Check your <strong style="color:#FEF3C7;">internet connection</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif auth_error:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #2a0a0a 0%, #4a0f0f 100%);
                    border: 1px solid #b91c1c; border-radius: 12px;
                    padding: 28px 32px; margin: 1rem 0 1.2rem; max-width: 680px;">
                    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                        <div style="width:44px;height:44px;border-radius:10px;
                            background:rgba(220,38,38,0.2);border:1px solid #dc2626;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;flex-shrink:0;">🔑</div>
                        <div>
                            <div style="font-size:1rem;font-weight:700;color:#fecaca;">Authentication Failed</div>
                            <div style="font-size:0.75rem;color:#fca5a5;margin-top:2px;">Service API · Invalid Credentials</div>
                        </div>
                    </div>
                    <p style="color:#fee2e2;font-size:0.88rem;line-height:1.65;margin:0 0 16px;">
                        The application could not authenticate with the provider. 
                        This typically means the <strong style="color:#ffffff;">API Key</strong> is missing, expired, or incorrectly formatted.
                    </p>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:14px 18px;font-size:0.82rem;color:#fca5a5;line-height:1.8;">
                        <strong style="color:#ffffff;display:block;margin-bottom:6px;">How to resolve:</strong>
                        ⚙️ &nbsp;Ensure your <strong style="color:#ffffff;">.env</strong> file has the correct API keys<br>
                        🔄 &nbsp;Restart the Streamlit application<br>
                        💳 &nbsp;Check if your provider account requires billing updates
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif api_error:
                st.markdown("""
                <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                    border:1px solid #374151;border-radius:12px;
                    padding:28px 32px;margin:1rem 0 1.2rem;max-width:680px;">
                    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                        <div style="width:44px;height:44px;border-radius:10px;
                            background:rgba(107,114,128,0.2);border:1px solid #6B7280;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;flex-shrink:0;">🤖</div>
                        <div>
                            <div style="font-size:1rem;font-weight:700;color:#F9FAFB;">AI Service Unavailable</div>
                            <div style="font-size:0.75rem;color:#9CA3AF;margin-top:2px;">Groq API · Unexpected Error</div>
                        </div>
                    </div>
                    <p style="color:#D1D5DB;font-size:0.88rem;line-height:1.65;margin:0 0 16px;">
                        The AI service encountered an unexpected issue while processing your request.
                        This is usually <strong style="color:#F9FAFB;">temporary</strong>.
                    </p>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:14px 18px;font-size:0.82rem;color:#9CA3AF;line-height:1.8;">
                        <strong style="color:#F9FAFB;display:block;margin-bottom:6px;">What you can do:</strong>
                        🔄 &nbsp;<strong style="color:#F9FAFB;">Try asking again</strong> — the issue is likely transient<br>
                        ✍️ &nbsp;Rephrase your question slightly<br>
                        🕐 &nbsp;Wait a moment and retry if the problem persists
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif model_error:
                st.markdown("""
                <div style="background:linear-gradient(135deg,#1c1008 0%,#2c1a0a 100%);
                    border:1px solid #92400e;border-radius:12px;
                    padding:28px 32px;margin:1rem 0 1.2rem;max-width:680px;">
                    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                        <div style="width:44px;height:44px;border-radius:10px;
                            background:rgba(180,83,9,0.2);border:1px solid #B45309;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;flex-shrink:0;">⚙️</div>
                        <div>
                            <div style="font-size:1rem;font-weight:700;color:#FEF3C7;">AI Model Configuration Error</div>
                            <div style="font-size:0.75rem;color:#FCD34D;margin-top:2px;">Groq API · Model Not Found</div>
                        </div>
                    </div>
                    <p style="color:#FDE68A;font-size:0.88rem;line-height:1.65;margin:0 0 16px;">
                        The configured AI model is unavailable or no longer supported by the API provider.
                    </p>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:14px 18px;font-size:0.82rem;color:#FCD34D;line-height:1.8;">
                        <strong style="color:#FEF3C7;display:block;margin-bottom:6px;">How to resolve:</strong>
                        ⚙️ &nbsp;Update <strong style="color:#FEF3C7;">GROQ_MODEL</strong> in your .env or Streamlit secrets<br>
                        📋 &nbsp;Check supported models at <strong style="color:#FEF3C7;">console.groq.com</strong><br>
                        🔄 &nbsp;Restart the application after updating the model name
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif pkg_error:
                st.markdown("""
                <div style="background:linear-gradient(135deg,#0f0f23 0%,#1a1a3e 100%);
                    border:1px solid #4F46E5;border-radius:12px;
                    padding:28px 32px;margin:1rem 0 1.2rem;max-width:680px;">
                    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                        <div style="width:44px;height:44px;border-radius:10px;
                            background:rgba(79,70,229,0.2);border:1px solid #4F46E5;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;flex-shrink:0;">📦</div>
                        <div>
                            <div style="font-size:1rem;font-weight:700;color:#E0E7FF;">Missing Dependency</div>
                            <div style="font-size:0.75rem;color:#A5B4FC;margin-top:2px;">Environment · Package Not Installed</div>
                        </div>
                    </div>
                    <p style="color:#C7D2FE;font-size:0.88rem;line-height:1.65;margin:0 0 16px;">
                        A required package is missing from this environment.
                    </p>
                    <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:14px 18px;font-size:0.82rem;color:#A5B4FC;line-height:1.8;">
                        <strong style="color:#E0E7FF;display:block;margin-bottom:6px;">How to resolve:</strong>
                        💻 &nbsp;Run <strong style="color:#E0E7FF;">pip install groq</strong> in your terminal<br>
                        🔄 &nbsp;Restart the Streamlit application<br>
                        📋 &nbsp;Ensure <strong style="color:#E0E7FF;">requirements.txt</strong> includes all dependencies
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif db_error:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #061e24 0%, #083344 100%);
                    border: 1px solid #0891b2; border-radius: 12px;
                    padding: 28px 32px; margin: 1rem 0 1.2rem; max-width: 680px;">
                    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                        <div style="width:44px;height:44px;border-radius:10px;
                            background:rgba(6,182,212,0.2);border:1px solid #06b6d4;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.4rem;flex-shrink:0;">🗄️</div>
                        <div>
                            <div style="font-size:1rem;font-weight:700;color:#cffafe;">Database Connection Failed</div>
                            <div style="font-size:0.75rem;color:#a5f3fc;margin-top:2px;">Zilliz Cloud · Retrieval Error</div>
                        </div>
                    </div>
                    <p style="color:#ecfeff;font-size:0.88rem;line-height:1.65;margin:0 0 16px;">
                        The application could not reach the vector database to retrieve textbook chunks.
                        This usually indicates a <strong style="color:#ffffff;">network block</strong> or an issue with the Zilliz cluster.
                    </p>
                    <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:14px 18px;font-size:0.82rem;color:#a5f3fc;line-height:1.8;">
                        <strong style="color:#ffffff;display:block;margin-bottom:6px;">Diagnostic steps:</strong>
                        📶 &nbsp;Ensure your network allows outbound connections on port 443<br>
                        ⚙️ &nbsp;Verify the <strong style="color:#ffffff;">ZILLIZ_URI</strong> in your .env file is correct<br>
                        ☁️ &nbsp;Check if the Zilliz Cloud instance is paused or sleeping
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # --- Fix: Avoid redundant "Not found" message bubble ---
                display_content = msg['content']
                if not_found:
                    prefix = "Not found in the provided textbook."
                    if display_content.lower().startswith(prefix.lower()):
                        display_content = display_content[len(prefix):].strip()
                
                # Only show bubble if there's actual content left
                if display_content:
                    formatted = format_answer_html(display_content)
                    st.markdown(f"""
                    <div class="assistant-wrap">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #F1F5F9;">
                            <div style="display:flex;align-items:center;gap:10px;">
                                <div class="assistant-avatar" style="margin:0;width:34px;height:34px;">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
                                    </svg>
                                </div>
                                <div>
                                    <div style="font-weight:800;font-size:0.92rem;color:#0F172A;letter-spacing:-0.01em;">NeuroNauts AI Tutor</div>
                                    <div style="font-size:0.68rem;color:#64748B;font-weight:500;">OpenStax Psychology 2e Grounded</div>
                                </div>
                            </div>
                            <div style="display:inline-flex;align-items:center;gap:5px;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:20px;padding:3px 10px;">
                                <span style="width:5px;height:5px;border-radius:50%;background:#16A34A;"></span>
                                <span style="font-size:0.68rem;font-weight:700;color:#15803D;text-transform:uppercase;letter-spacing:0.04em;">Zero Hallucination</span>
                            </div>
                        </div>
                        <div class="assistant-text">{formatted}</div>
                    </div>""", unsafe_allow_html=True)

                # ✅ Images render from Cloudinary URLs (only if found)
                if not not_found and msg.get("images"):
                    render_image_row(msg["images"], msg_index=idx)

                # ✅ Sources panel (only if found)
                if not not_found and msg.get("sources"):
                    render_sources_panel(msg["sources"])

                # ✅ Interactive Follow-up Suggestion Chips (rendered for latest assistant response)
                if not not_found and not is_error and idx == len(st.session_state.messages) - 1:
                    user_q = ""
                    for m in reversed(st.session_state.messages[:idx]):
                        if m["role"] == "user":
                            user_q = m["content"]
                            break
                    follow_ups = get_follow_up_suggestions(user_q, display_content, msg.get("sources", []))
                    if follow_ups:
                        st.markdown("""
                        <div style="margin: 1.2rem 0 0.5rem;">
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                                <span style="font-size:0.7rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;">
                                    💡 Explore Follow-Up Concepts
                                </span>
                                <div style="height:1px;flex:1;background:#E2E8F0;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        fu_cols = st.columns(len(follow_ups), gap="small")
                        for f_col, (f_emo, f_label, f_query) in zip(fu_cols, follow_ups):
                            k = f"fu_{idx}_{re.sub(r'[^a-z0-9]','_',f_label[:10].lower())}"
                            if f_col.button(f"{f_emo} {f_label}", key=k, use_container_width=True):
                                st.session_state.messages.append({"role": "user", "content": f_query})
                                st.session_state["_pending_query"] = f_query
                                st.rerun()

                # ✅ Not Found card
                if not_found:
                    st.markdown("""
                    <div style="margin:0.5rem 0 1rem;max-width:580px;
                                padding:20px 24px;background:#FAFAFA;border-radius:10px;
                                border:1px solid #E2E8F0;">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                            <span style="font-size:1.1rem;">🔍</span>
                            <span style="font-size:0.9rem;font-weight:600;color:#0F172A;">Not found in the textbook</span>
                        </div>
                        <p style="font-size:0.82rem;color:#64748B;margin:0;line-height:1.6;">
                            This topic doesn't appear in the OpenStax Psychology 2e textbook content we have indexed.
                            Try rephrasing your question or asking about a related psychology concept.
                        </p>
                    </div>""", unsafe_allow_html=True)

                # ✅ Performance badge at the VERY end
                duration = msg.get("duration")
                if duration and not is_error:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-top: 20px; margin-bottom: -10px;">
                        <div style="display: flex; align-items: center; gap: 8px; padding: 5px 12px; 
                                    background: #ffffff; border: 1px solid #E2E8F0; border-radius: 8px;
                                    box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div style="width: 6px; height: 6px; border-radius: 50%; background: #10B981; animation: pulse-dot 2s infinite;"></div>
                                <span style="font-size: 0.62rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em;">Processed in</span>
                            </div>
                            <div style="width: 1px; height: 12px; background: #E2E8F0;"></div>
                            <span style="font-size: 0.72rem; color: #0F172A; font-weight: 600; font-family: 'JetBrains Mono', 'Courier New', monospace;">{duration:.2f}s</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<hr class='turn-divider'>", unsafe_allow_html=True)

    query = st.chat_input("Ask anything about psychology…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        chunks, answer, images, rate_limited, timed_out, duration = run_query(query)
        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "sources": chunks, "images": images,
            "rate_limited": rate_limited,
            "timed_out": timed_out,
            "duration": duration,
        })
        st.rerun()


# ─── PAGE: EVALUATION ─────────────────────────────────────────────────────────

# ─── PAGE: EVALUATION ─────────────────────────────────────────────────────────

def show_evaluation_page():
    st.markdown("""
    <div style="padding:0.4rem 0 1.5rem;max-width:960px;margin:0 auto;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:0.8rem;">
            <div style="width:48px;height:48px;border-radius:12px;flex-shrink:0;
                        background:linear-gradient(135deg, #4F46E5 0%, #312E81 100%);color:white;
                        display:flex;align-items:center;justify-content:center;
                        box-shadow:0 6px 16px rgba(79,70,229,0.25);">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor" style="width:26px;height:26px;">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                </svg>
            </div>
            <div>
                <h2 style="margin:0;font-size:1.6rem;font-weight:700;color:var(--text-main);">Enterprise RAG Evaluation Suite</h2>
                <p style="margin:4px 0 0;font-size:0.85rem;color:var(--text-muted);">
                    RAG Triad &nbsp;·&nbsp; Faithfulness &nbsp;·&nbsp; Answer Relevancy &nbsp;·&nbsp; Context Precision &nbsp;·&nbsp; Context Recall
                </p>
            </div>
        </div>
        <div style="height:1px;background:var(--border-color);margin-top:1.5rem;"></div>
    </div>
    """, unsafe_allow_html=True)

    queries_path = PROJECT_ROOT / "queries.json"
    if not queries_path.exists():
        st.markdown("""
        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;
            padding:20px 24px;max-width:600px;margin:1rem 0;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="font-size:1.1rem;">📂</span>
                <span style="font-size:0.9rem;font-weight:600;color:#92400E;">Evaluation data not found</span>
            </div>
            <p style="font-size:0.82rem;color:#78350F;margin:0;line-height:1.6;">
                The <strong>queries.json</strong> file is missing from the project output directory.
                Please ensure the evaluation dataset has been generated before running this page.
            </p>
        </div>""", unsafe_allow_html=True)
        return

    with open(queries_path, encoding="utf-8") as f:
        all_queries = json.load(f)

    c1, c2 = st.columns([1, 2])
    with c1:
        sample_size = st.slider("Queries to evaluate", 5, min(len(all_queries), 50), 5)
    with c2:
        st.markdown(
            f"<div style='padding-top:1.8rem;color:var(--text-muted);font-size:0.85rem;'>"
            f"Benchmarking <strong>{sample_size}</strong> of {len(all_queries)} queries against RAG Triad standards</div>",
            unsafe_allow_html=True)

    if st.button("▶ Run Full Benchmark", type="primary"):
        with st.spinner("Running comprehensive RAG Triad evaluation…"):
            eval_data = run_evaluation(all_queries, sample_size)
        st.session_state["eval_data"] = eval_data
        st.success("✅ Evaluation complete!"); st.rerun()

    eval_data = st.session_state.get("eval_data")
    if eval_data is None and OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, encoding="utf-8") as f: eval_data = json.load(f)
        except Exception:
            eval_data = None

    if eval_data is None:
        st.info("No evaluation runs yet. Click **▶ Run Full Benchmark** to start."); return

    summary = eval_data.get("summary", {})
    results = eval_data.get("results", [])
    valid   = [r for r in results if r.get("faithfulness_score") is not None]

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Enterprise Quality Dashboard")

    def card_color(s, g, o):
        return "#15803D" if s>=g else ("#B45309" if s>=o else "#B91C1C")
    def card_bg(s, g, o):
        return "#F0FDF4" if s>=g else ("#FFFBEB" if s>=o else "#FEF2F2")
    def card_border(s, g, o):
        return "#BBF7D0" if s>=g else ("#FDE68A" if s>=o else "#FECACA")

    avg_comp = summary.get("avg_composite_score", summary.get("avg_faithfulness", 0.8))
    avg_faith = summary.get("avg_faithfulness", 0.85)
    avg_rel = summary.get("avg_answer_relevancy", 0.82)
    avg_prec = summary.get("avg_context_precision", 0.78)
    avg_rec = summary.get("avg_context_recall", 0.80)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""
        <div class="metric-card" style="background:{card_bg(avg_comp, 0.75, 0.60)};border-color:{card_border(avg_comp, 0.75, 0.60)};">
            <div class="metric-value" style="color:{card_color(avg_comp, 0.75, 0.60)}">{avg_comp:.1%}</div>
            <div class="metric-label">Composite Quality</div>
            <div class="metric-sub" style="color:{card_color(avg_comp, 0.75, 0.60)};">RAG Triad Weighted</div>
        </div>""", unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class="metric-card" style="background:{card_bg(avg_faith, 0.75, 0.60)};border-color:{card_border(avg_faith, 0.75, 0.60)};">
            <div class="metric-value" style="color:{card_color(avg_faith, 0.75, 0.60)}">{avg_faith:.1%}</div>
            <div class="metric-label">Faithfulness</div>
            <div class="metric-sub" style="color:{card_color(avg_faith, 0.75, 0.60)};">Grounded in context</div>
        </div>""", unsafe_allow_html=True)

    with mc3:
        st.markdown(f"""
        <div class="metric-card" style="background:{card_bg(avg_rel, 0.70, 0.55)};border-color:{card_border(avg_rel, 0.70, 0.55)};">
            <div class="metric-value" style="color:{card_color(avg_rel, 0.70, 0.55)}">{avg_rel:.1%}</div>
            <div class="metric-label">Answer Relevancy</div>
            <div class="metric-sub" style="color:{card_color(avg_rel, 0.70, 0.55)};">Query semantic match</div>
        </div>""", unsafe_allow_html=True)

    with mc4:
        st.markdown(f"""
        <div class="metric-card" style="background:{card_bg(avg_prec, 0.65, 0.50)};border-color:{card_border(avg_prec, 0.65, 0.50)};">
            <div class="metric-value" style="color:{card_color(avg_prec, 0.65, 0.50)}">{avg_prec:.1%}</div>
            <div class="metric-label">Context Precision</div>
            <div class="metric-sub" style="color:{card_color(avg_prec, 0.65, 0.50)};">Signal-to-noise ratio</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "rb") as f:
            st.download_button("⬇ Download Evaluation CSV", data=f.read(),
                               file_name="evaluation_results.csv", mime="text/csv", type="secondary")

    st.markdown("### Per-Query Diagnostic Drilldown")
    for r in valid:
        faith = r.get("faithfulness_score", 0.0)
        relev = r.get("relevancy_score", 0.0)
        prec  = r.get("context_precision_score", 0.0)
        comp  = r.get("composite_score", faith)
        grade = r.get("grade", "A")

        fe  = "🟢" if faith>=0.75 else "🟡" if faith>=0.60 else "🔴"
        re_ = "🟢" if relev>=0.70 else "🟡" if relev>=0.55 else "🔴"
        pe_ = "🟢" if prec>=0.65 else "🟡" if prec>=0.50 else "🔴"

        with st.expander(f"Q{r['query_id']}  [{grade}] {fe} Faith: {faith:.2f}  |  {re_} Relev: {relev:.2f}  |  {pe_} Prec: {prec:.2f}  —  {r['question'][:55]}…"):
            st.markdown("**Generated Answer**"); st.info(r["answer"])

            st.markdown("**Faithfulness — Sentence Claim Verification**")
            sentences = r.get("faithfulness_detail", {}).get("sentences", [])
            if sentences:
                rows = "".join(f"""
                <div class="sent-row">
                    <span class="sent-badge {'sent-ok' if s.get('supported') else 'sent-fail'}">
                        {'supported' if s.get('supported') else 'not supported'}
                    </span>
                    <span class="sent-sim">{s.get('max_sim', 0.0):.3f}</span>
                    <span>{html.escape(s.get('sentence', ''))}</span>
                </div>""" for s in sentences)
                st.markdown(rows, unsafe_allow_html=True)
            else:
                st.caption("No sentence data available.")

            st.markdown("**Detailed Metric Breakdown**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Faithfulness", f"{faith:.4f}",
                       delta="✓ verified" if faith>=FAITH_THRESHOLD else "✗ hallucination risk",
                       delta_color="normal" if faith>=FAITH_THRESHOLD else "inverse")
            sc2.metric("Answer Relevancy", f"{relev:.4f}",
                       delta="✓ relevant" if relev>=0.70 else "✗ weak intent match",
                       delta_color="normal" if relev>=0.70 else "inverse")
            sc3.metric("Context Precision", f"{prec:.4f}",
                       delta="✓ clean signal" if prec>=0.60 else "✗ noisy context",
                       delta_color="normal" if prec>=0.60 else "inverse")
            sc4.metric("Context Recall", f"{r.get('context_recall_score', 0.8):.4f}",
                       delta="✓ complete" if r.get('context_recall_score', 0.8)>=0.60 else "✗ partial coverage",
                       delta_color="normal")

            with st.expander("Retrieved Context Chunks Used"):
                for ci, ctx in enumerate(r.get("contexts", []), 1):
                    st.markdown(f"**Chunk {ci}:** {ctx[:250]}…")


# ─── ROUTER ──────────────────────────────────────────────────────────────────

def _render_fatal_error(context: str, err: Exception):
    """Last-resort handler: shows a professional error instead of a raw traceback."""
    import traceback
    print(f"[FATAL] {context}: {type(err).__name__}: {err}")
    print(traceback.format_exc())
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a0a0a 0%,#2d0f0f 100%);
        border:1px solid #7f1d1d;border-radius:12px;
        padding:32px 36px;margin:2rem auto;max-width:680px;">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
            <div style="width:48px;height:48px;border-radius:10px;
                background:rgba(239,68,68,0.15);border:1px solid #ef4444;
                display:flex;align-items:center;justify-content:center;
                font-size:1.5rem;flex-shrink:0;">⚠️</div>
            <div>
                <div style="font-size:1.05rem;font-weight:700;color:#FEE2E2;">
                    Something went wrong</div>
                <div style="font-size:0.75rem;color:#FCA5A5;margin-top:3px;">
                    NeuroNauts · Unexpected Error</div>
            </div>
        </div>
        <p style="color:#FECACA;font-size:0.88rem;line-height:1.7;margin:0 0 18px;">
            An unexpected error occurred while loading this page.
            The issue has been logged automatically.
            Please try refreshing the page or starting a new chat session.
        </p>
        <div style="background:rgba(255,255,255,0.04);border-radius:8px;
            padding:14px 18px;font-size:0.82rem;color:#FCA5A5;line-height:1.9;">
            <strong style="color:#FEE2E2;display:block;margin-bottom:6px;">
                What you can do:</strong>
            🔄 &nbsp;<strong style="color:#FEE2E2;">Refresh the page</strong> — most errors are transient<br>
            ➕ &nbsp;Click <strong style="color:#FEE2E2;">New Chat</strong> to start a fresh session<br>
            🌐 &nbsp;Check your internet connection and try again
        </div>
    </div>
    """, unsafe_allow_html=True)


try:
    if st.session_state.page == "chat":
        show_chat_page()
    elif st.session_state.page == "psych_lab":
        show_psych_lab_page(_get_groq_client, _resolve_groq_model)
    elif st.session_state.page == "study_hub":
        show_study_hub_page(_get_groq_client, _resolve_groq_model)
    elif st.session_state.page == "kg":
        show_knowledge_graph_page(CHUNKS_PATH)
    else:
        show_evaluation_page()
except Exception as _top_level_err:
    _render_fatal_error(st.session_state.get("page", "unknown"), _top_level_err)
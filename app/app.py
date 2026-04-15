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
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from retrieve import retrieve as _hybrid_retrieve
from knowledge_graph import show_knowledge_graph_page

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #4F46E5; 
    --primary-hover: #4338CA;
    --bg-light: #F8FAFC;
    --text-main: #0F172A;
    --text-muted: #64748B;
    --border-color: #E2E8F0;
    color-scheme: light only;
}

/* Ensure native expand button is visible and colored red */
[data-testid="collapsedControl"] {
    color: #EF4444 !important;
    background: rgba(239, 68, 68, 0.1) !important;
    border-radius: 0 8px 8px 0 !important;
    z-index: 99999 !important;
}
[data-testid="collapsedControl"] svg {
    fill: #EF4444 !important;
}

/* ── Force light mode on main content only (NOT sidebar) ── */
html, body {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    color-scheme: light only !important;
    background-color: #ffffff !important;
    color: #0F172A !important;
    -webkit-text-size-adjust: 100% !important;
}

/* Main wrappers: light mode */
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"],
.main {
    background-color: #ffffff !important;
    color: #0F172A !important;
    color-scheme: light only !important;
}

/* ── Override mobile dark-mode — scoped to main content, not sidebar ── */
@media (prefers-color-scheme: dark) {
    html, body {
        background-color: #ffffff !important;
        color: #0F172A !important;
        color-scheme: light !important;
    }
    /* Main area only */
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section[data-testid="stMain"],
    .main {
        background-color: #ffffff !important;
        color: #0F172A !important;
    }
    /* Block container background */
    .block-container {
        background-color: #ffffff !important;
    }
    /* Assistant & user text */
    .assistant-text, .assistant-text p,
    .assistant-text li, .assistant-text strong,
    .user-bubble {
        color: #0F172A !important;
    }
    /* Chat input */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #0F172A !important;
        border-color: #E2E8F0 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important;
        opacity: 1 !important;
    }
    [data-testid="stBottom"] {
        background-color: #ffffff !important;
        border-top: 1px solid #E2E8F0 !important;
    }
    /* Source cards */
    .src-card { background: #ffffff !important; }
    .src-section { color: #0F172A !important; }
    .src-num { background: #F8FAFC !important; color: #64748B !important; }
    .src-preview, .src-page, .src-page-badge { color: #64748B !important; }
    .sources-header { color: #64748B !important; }
    /* Quick-start suggestion chips */
    .block-container div[data-testid="stHorizontalBlock"] .stButton > button {
        background: #ffffff !important;
        color: #0F172A !important;
        border-color: #E2E8F0 !important;
    }
    /* Expanders, metrics */
    [data-testid="stExpander"] > div,
    [data-testid="metric-container"] {
        background-color: #ffffff !important;
        color: #0F172A !important;
    }
    /* Input fields in main area */
    section[data-testid="stMain"] input,
    section[data-testid="stMain"] textarea,
    section[data-testid="stMain"] select {
        background-color: #ffffff !important;
        color: #0F172A !important;
    }
}
/* Hide Streamlit chrome */
#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
/* Keep header visible so Streamlit's native sidebar toggle is accessible */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}
.block-container {
    max-width: 100% !important;
    padding: 2.5rem 2.5rem 7rem !important;
}

[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B !important;
    width: 260px !important; min-width: 260px !important; max-width: 260px !important;
    color-scheme: dark !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.2rem 1rem !important; }

[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px !important; 
    font-size: 0.85rem !important;
    font-weight: 500 !important; 
    padding: 10px 14px !important;
    width: 100% !important; 
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #ffffff !important; 
    border: 1px solid var(--primary) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: var(--primary-hover) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
    background: transparent !important;
    color: #CBD5E1 !important; 
    border: 1px solid #334155 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
    background: #1E293B !important; 
    color: #F8FAFC !important;
}

/* ── Main content area: always white background ── */
[data-testid="stAppViewContainer"] > .main {
    background-color: #ffffff !important;
}
section[data-testid="stMain"] {
    background-color: #ffffff !important;
}

.user-bubble-wrap { display:flex; justify-content:flex-end; margin:1.5rem 0 1rem; width: 100%; }
.user-bubble {
    background: #F1F5F9 !important;
    color: #0F172A !important; 
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px 12px 2px 12px;
    padding: 14px 20px; 
    max-width: 75%; 
    font-size: 0.95rem;
    line-height: 1.6; 
    word-break: break-word;
}

.assistant-wrap { display:flex; gap:16px; margin:1rem 0; align-items:flex-start; width: 100%; }
.assistant-avatar {
    width: 32px; height: 32px; border-radius: 6px;
    background: var(--primary);
    color: white;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 4px;
}
.assistant-avatar svg { width: 18px; height: 18px; }
.assistant-text {
    flex: 1; font-size: 0.95rem; line-height: 1.75; min-width: 0;
    color: #0F172A !important; word-break: break-word; padding-top: 2px;
}
.assistant-text p  { margin: 0 0 0.85em; color: #0F172A !important; }
.assistant-text p:last-child { margin-bottom: 0; }
.assistant-text ul, .assistant-text ol { margin: 0.4em 0 0.85em 1.2em; padding: 0; }
.assistant-text li { margin-bottom: 0.3em; color: #0F172A !important; }
.assistant-text strong { color: #1E293B !important; font-weight: 600; }
.turn-divider { border:none; border-top:1px solid #E2E8F0; margin:2rem 0; }

.sources-header {
    font-size: 0.7rem; font-weight: 600; color: #64748B !important;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-top: 1rem; margin-bottom: 0.75rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #E2E8F0;
    display: flex; align-items: center; gap: 8px;
}

/* ── Source card grid: auto-fill responsive columns ── */
.src-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
    margin-top: 0;
    width: 100%;
}
@media (max-width: 520px) {
    .src-grid {
        grid-template-columns: 1fr;
        gap: 8px;
    }
}
@media (min-width: 521px) and (max-width: 800px) {
    .src-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 9px;
    }
}
.src-card {
    background: #ffffff !important; 
    border: 1px solid #E2E8F0 !important;
    border-radius: 6px; 
    padding: 12px; 
    margin: 0;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    width: 100%; 
    min-width: 0; 
    box-sizing: border-box;
    overflow: hidden; 
    display: flex;
    flex-direction: column;
}
.src-card:hover {
    border-color: #CBD5E1 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.src-card-top { 
    display: flex; align-items: flex-start; gap: 8px; 
    margin-bottom: 6px; width: 100%; 
}
.src-num {
    background: #F8FAFC !important; color: #64748B !important; 
    border: 1px solid #E2E8F0 !important;
    border-radius: 4px; padding: 1px 5px;
    font-size: 0.6rem; font-weight: 600; flex-shrink: 0;
}
.src-section { 
    font-size: 0.75rem; font-weight: 600; color: #0F172A !important; line-height: 1.3;
    word-break: break-word; overflow-wrap: break-word;
}
.src-preview {
    font-size: 0.7rem; color: #64748B !important; line-height: 1.4;
    margin-bottom: 8px;
    display: -webkit-box; -webkit-line-clamp: 2; 
    -webkit-box-orient: vertical; 
    overflow: hidden;
    word-break: break-word; overflow-wrap: break-word;
    flex-grow: 1; 
}
.src-page {
    display:flex; align-items:center; gap:6px;
    font-size:0.6rem; color:#64748B !important;
    margin-top: auto; 
}
.src-page-badge {
    background: #ffffff !important; border: 1px solid #E2E8F0 !important;
    border-radius: 3px; padding: 1px 5px;
    font-size: 0.65rem; font-weight: 500; color: #64748B !important;
}

[data-testid="stChatInput"] {
    border-radius: 28px !important; 
    border: 1px solid #E2E8F0 !important;
    background: #ffffff !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -2px rgba(0, 0, 0, 0.01) !important;
    padding: 4px 10px !important;
    margin-bottom: 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #4F46E5 !important;
    box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.02) !important;
    transform: translateY(-2px);
}

[data-testid="stChatInput"] textarea {
    font-size: 0.95rem !important; 
    padding: 12px 18px !important;
    color: #0F172A !important;
    background: transparent !important;
    font-weight: 500 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

[data-testid="stBottom"] {
    background-color: #ffffff !important;
    border-top: 1px solid #E2E8F0 !important;
}

/* Style the send button icon specifically if possible */
[data-testid="stChatInput"] button {
    background-color: transparent !important;
    color: #4F46E5 !important;
    transition: transform 0.2s ease !important;
}
[data-testid="stChatInput"] button:hover {
    transform: scale(1.1) rotate(-5deg);
}

/* Quick-start suggestion buttons */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #ffffff !important; border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important; color: #1E293B !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    padding: 10px 12px !important; white-space: normal !important;
    height: auto !important; min-height: 48px !important;
    line-height: 1.35 !important; box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important; text-align: center !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color:var(--primary) !important; background:#F8FAFC !important;
    color:var(--primary-hover) !important; 
}

.status-pill {
    display:inline-flex; align-items:center; gap:8px;
    padding:8px 16px; background: #ffffff !important; border: 1px solid #E2E8F0 !important;
    border-radius: 20px; font-size: 0.85rem; color: #64748B !important; 
    font-weight: 500; margin: 0.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--primary);
    animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

.metric-card {
    border-radius: 8px; padding: 24px 20px; text-align: center; 
    border: 1px solid #E2E8F0 !important; background: #ffffff !important;
}
.metric-value{font-size:2rem; font-weight:700; color: #0F172A !important; line-height:1;}
.metric-label{font-size:0.8rem; color: #64748B !important; margin-top:8px; font-weight:600;}
.metric-sub{font-size:0.7rem; color: #94A3B8 !important; margin-top:4px;}

.sent-row{display:flex;gap:8px;align-items:flex-start;padding:8px 0;
          border-bottom:1px solid #E2E8F0;font-size:0.85rem; color:#0F172A !important;}
.sent-badge{flex-shrink:0;padding:2px 7px;border-radius:4px;
            font-size:0.65rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;}
.sent-ok{background:#dcfce7;color:#15803d;border:1px solid #bbf7d0;}
.sent-fail{background:#fee2e2;color:#dc2626;border:1px solid #fecaca;}

@media (max-width: 768px) {
    .block-container { padding: 1.2rem 1rem 7rem !important; }
    [data-testid="stSidebar"] { width: 220px !important; min-width: 220px !important; max-width: 220px !important; }
    .user-bubble   { max-width: 88% !important; font-size: 0.9rem !important; }
    .assistant-text { font-size: 0.9rem !important; line-height: 1.65 !important; }
}

@media (max-width: 480px) {
    .block-container { padding: 0.75rem 0.6rem 6rem !important; }
    .user-bubble-wrap { margin: 1rem 0 0.75rem !important; }
    .user-bubble { max-width: 95% !important; font-size: 0.88rem !important; padding: 12px 15px !important; border-radius: 10px 10px 2px 10px !important; }
    .assistant-wrap { gap: 10px !important; margin: 0.75rem 0 !important; }
    .assistant-avatar { width: 28px !important; height: 28px !important; border-radius: 6px !important; }
    .assistant-avatar svg { width: 15px !important; height: 15px !important; }
    .assistant-text { font-size: 0.87rem !important; line-height: 1.65 !important; }
    .turn-divider { margin: 1.2rem 0 !important; }
    .src-card { padding: 10px !important; }
    .src-section { font-size: 0.72rem !important; }
    .src-preview { font-size: 0.67rem !important; }
    [data-testid="stChatInput"] { border-radius: 28px !important; padding: 2px 4px !important; }
    [data-testid="stChatInput"] textarea { font-size: 0.9rem !important; padding: 10px 14px !important; min-height: 45px !important; font-weight: 500 !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button { font-size: 0.8rem !important; padding: 11px 12px !important; min-height: 54px !important; }
    .status-pill { font-size: 0.78rem !important; padding: 7px 13px !important; }
    .metric-value { font-size: 1.5rem !important; }
    .metric-label { font-size: 0.72rem !important; }
    [data-testid="stSidebar"] { width: 80vw !important; max-width: 280px !important; min-width: unset !important; }
}

@media (pointer: coarse) {
    [data-testid="stSidebar"] .stButton > button { min-height: 48px !important; padding: 12px 14px !important; }
    .src-card:hover { border-color: var(--border-color) !important; box-shadow: none !important; }
    [data-testid="stChatInput"] textarea { font-size: max(16px, 0.94rem) !important; }
}

img { max-width: 100% !important; height: auto !important; }
* { box-sizing: border-box; }
html, body { overflow-x: hidden !important; }
</style>
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
    
    if st.button("🗺️ Knowledge", use_container_width=True, key="nav_kg",
                 type="primary" if st.session_state.page=="kg" else "secondary"):
        st.session_state.page = "kg"; st.rerun()

    if st.button("📊 Eval", use_container_width=True, key="nav_eval",
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
            "You are a psychology textbook assistant with memory of this conversation. "
            "Answer using ONLY the provided context from the textbook. "
            "DO NOT use outside knowledge. "
            "Give a concise, well-structured answer of at least 3-5 sentences. "
            "Include key definitions, examples, and relevant details from the context. "
            "ALWAYS include section and page references in your answer based on the context. "
            "Use the conversation history to understand follow-up questions. "
            "Do NOT generate HTML. "
            "If the answer is not in the context, say: "
            "'Not found in the provided textbook'."
        )
    }

    try:
        client = _get_groq_client()
        completion = client.chat.completions.create(
            model=_resolve_groq_model(),
            messages=[system] + history,
            temperature=0.2,
            max_tokens=800,
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
                    "You are a psychology textbook assistant. "
                    "Answer using ONLY the provided context from the textbook. "
                    "DO NOT use outside knowledge. "
                    "Give a concise, well-structured answer of at least 3-5 sentences. "
                    "ALWAYS include section and page references in your answer based on the context. "
                    "Include key definitions, examples, and relevant details from the context. "
                    "Do NOT generate HTML. "
                    "If the answer is not in the context, say: "
                    "'Not found in the provided textbook'."
                )},
                {"role": "user", "content": (
                    f"Context from textbook:\n{context}\n\nQuestion: {question}"
                )},
            ],
            temperature=0.1,
            max_tokens=600,
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
    Extract unique Cloudinary image URLs from retrieved chunks.
    - Parses image_refs (handles Zilliz JSON encoding).
    - Converts bare filenames → full Cloudinary URLs.
    - Cloudinary public_id = filename WITHOUT extension, so we strip it.
      e.g. "img_p91_0.jpeg" → public_id "img_p91_0"
      URL: https://res.cloudinary.com/dnbnrxyn1/image/upload/img_p91_0
      (Cloudinary serves the original format automatically without the extension)
    - Deduplicates by stem so same image never appears twice.
    """
    seen_stems = set()
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
                # Strip extension to get Cloudinary public_id
                # e.g. "img_p91_0.jpeg" → "img_p91_0"
                stem = filename.rsplit(".", 1)[0].lower()
                full_url = f"{_CLOUDINARY_BASE}/{stem}"
            if stem and stem not in seen_stems:
                seen_stems.add(stem)
                result.append({"url": full_url, "stem": stem})
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
    text = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = _re.sub(r'__(.+?)__',      r'<strong>\1</strong>', text)
    paragraphs = _re.split(r'\n{2,}', text.strip())
    html_parts = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        lines = para.split('\n')
        bullet_lines = [l for l in lines
                        if _re.match(r'^\s*[-*•]\s', l)
                        or _re.match(r'^\s*\d+\.\s', l)]
        if len(bullet_lines) >= 2:
            is_ordered = all(_re.match(r'^\s*\d+\.\s', l) for l in bullet_lines)
            tag = 'ol' if is_ordered else 'ul'
            _bullet_re = r'^\s*[-*\u2022\d+.]+\s'
            li_items  = [f'<li>{_re.sub(_bullet_re, "", l).strip()}</li>'
                         for l in lines if l.strip()]
            li_joined = "".join(li_items)
            html_parts.append(f'<{tag}>{li_joined}</{tag}>')
        else:
            inner = '<br>'.join(l for l in lines if l.strip())
            html_parts.append(f'<p>{inner}</p>')
    return ''.join(html_parts) or f'<p>{html.escape(text)}</p>'


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
    for i, q in enumerate(selected):
        progress.progress((i+1)/len(selected),
                          text=f"Query {i+1}/{len(selected)}: {q['question'][:52]}…")
        qid, question = q["query_id"], q["question"]
        try:
            chunks    = retrieve(question)
            contexts  = [c["clean_text"] for c in chunks]
            answer    = call_llm_stateless(question, build_context(chunks))
            faith     = score_faithfulness(answer, contexts)
            relevancy = score_answer_relevancy(question, answer)
            results.append({
                "query_id":qid,"question":question,"answer":answer,"contexts":contexts,
                "faithfulness_score":faith["score"],"faithfulness_detail":faith,
                "relevancy_score":relevancy["score"],"relevancy_detail":relevancy,
            })
        except Exception as e:
            results.append({"query_id":qid,"question":question,"answer":"",
                            "faithfulness_score":None,"relevancy_score":None,"error":str(e)})
    progress.empty()
    valid         = [r for r in results if r.get("faithfulness_score") is not None]
    avg_faith     = round(sum(r["faithfulness_score"] for r in valid)/len(valid),4) if valid else 0.0
    avg_relevancy = round(sum(r["relevancy_score"]    for r in valid)/len(valid),4) if valid else 0.0
    output = {"summary":{"total_evaluated":len(valid),"avg_faithfulness":avg_faith,
                          "avg_answer_relevancy":avg_relevancy,
                          "faithfulness_threshold":FAITH_THRESHOLD},"results":results}
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON,"w",encoding="utf-8") as f: json.dump(output,f,indent=2,ensure_ascii=False)
    with open(OUTPUT_CSV,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=["query_id","question","answer",
                                          "faithfulness_score","relevancy_score"])
        w.writeheader()
        for r in results:
            w.writerow({"query_id":r["query_id"],"question":r["question"],
                        "answer":r.get("answer",""),"faithfulness_score":r.get("faithfulness_score",""),
                        "relevancy_score":r.get("relevancy_score","")})
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
        <div style="text-align:center; padding: 3rem 0 1.5rem;">
            <div style="display:inline-flex; align-items:center; justify-content:center;
                        width:72px; height:72px; border-radius:20px; 
                        background: linear-gradient(135deg, #4F46E5 0%, #312E81 100%);
                        color:white; box-shadow:0 10px 30px rgba(79,70,229,0.3);">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor" style="width:34px; height:34px;">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
                </svg>
            </div>
            <h1 style="margin: 1.2rem 0 0.5rem; font-size: 1.8rem; font-weight: 800;
                       color: #0F172A; letter-spacing: -0.03em;">
                Ask anything about Psychology
            </h1>
            <p style="color: #64748B; font-size: 0.88rem; font-weight: 500;">
                Powered by OpenStax Psychology 2e · Hybrid Retrieval System
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
                        <div class="assistant-avatar">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
                            </svg>
                        </div>
                        <div class="assistant-text">{formatted}</div>
                    </div>""", unsafe_allow_html=True)

                # ✅ Images render from Cloudinary URLs (only if found)
                if not not_found and msg.get("images"):
                    render_image_row(msg["images"], msg_index=idx)

                # ✅ Sources panel (only if found)
                if not not_found and msg.get("sources"):
                    render_sources_panel(msg["sources"])

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

def show_evaluation_page():
    st.markdown("""
    <div style="padding:0.4rem 0 1.5rem;max-width:860px;margin:0 auto;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:0.8rem;">
            <div style="width:48px;height:48px;border-radius:8px;flex-shrink:0;
                        background:var(--primary);color:white;
                        display:flex;align-items:center;justify-content:center;
                        box-shadow:0 4px 12px rgba(79,70,229,0.15);">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width:26px;height:26px;">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                </svg>
            </div>
            <div>
                <h2 style="margin:0;font-size:1.6rem;font-weight:700;color:var(--text-main);">RAG Evaluation</h2>
                <p style="margin:4px 0 0;font-size:0.85rem;color:var(--text-muted);">
                    Faithfulness &nbsp;·&nbsp; Answer Relevancy &nbsp;·&nbsp; No ground truth needed
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
        sample_size = st.slider("Queries to evaluate", 5, len(all_queries), 5)
    with c2:
        st.markdown(
            f"<div style='padding-top:1.8rem;color:var(--text-muted);font-size:0.85rem;'>"
            f"Evaluating <strong>{sample_size}</strong> of {len(all_queries)} queries</div>",
            unsafe_allow_html=True)

    if st.button("▶ Run Evaluation", type="primary"):
        with st.spinner("Running evaluation…"):
            eval_data = run_evaluation(all_queries, sample_size)
        st.session_state["eval_data"] = eval_data
        st.success("✅ Evaluation complete!"); st.rerun()

    eval_data = st.session_state.get("eval_data")
    if eval_data is None and OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, encoding="utf-8") as f: eval_data = json.load(f)
    if eval_data is None:
        st.info("No results yet. Click **▶ Run Evaluation** to start."); return

    summary = eval_data["summary"]
    results = eval_data["results"]
    valid   = [r for r in results if r.get("faithfulness_score") is not None]

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Overall Scores")

    def card_color(s, g, o):
        return "#15803D" if s>=g else ("#B45309" if s>=o else "#B91C1C")
    def card_bg(s, g, o):
        return "#F0FDF4" if s>=g else ("#FFFBEB" if s>=o else "#FEF2F2")
    def card_border(s, g, o):
        return "#BBF7D0" if s>=g else ("#FDE68A" if s>=o else "#FECACA")

    fc_text   = card_color(summary["avg_faithfulness"], 0.75, 0.60)
    fc_bg     = card_bg(summary["avg_faithfulness"], 0.75, 0.60)
    fc_border = card_border(summary["avg_faithfulness"], 0.75, 0.60)
    rc_text   = card_color(summary["avg_answer_relevancy"], 0.70, 0.55)
    rc_bg     = card_bg(summary["avg_answer_relevancy"], 0.70, 0.55)
    rc_border = card_border(summary["avg_answer_relevancy"], 0.70, 0.55)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""
        <div class="metric-card" style="background:{fc_bg};border-color:{fc_border};">
            <div class="metric-value" style="color:{fc_text}">{summary['avg_faithfulness']:.1%}</div>
            <div class="metric-label">Faithfulness</div>
            <div class="metric-sub" style="color:{fc_text};opacity:0.8;">answer backed by context</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="metric-card" style="background:{rc_bg};border-color:{rc_border};">
            <div class="metric-value" style="color:{rc_text}">{summary['avg_answer_relevancy']:.1%}</div>
            <div class="metric-label">Answer Relevancy</div>
            <div class="metric-sub" style="color:{rc_text};opacity:0.8;">answer addressed question</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary["total_evaluated"]}</div>
            <div class="metric-label">Queries Evaluated</div>
            <div class="metric-sub">threshold: {summary['faithfulness_threshold']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV,"rb") as f:
            st.download_button("⬇ Download CSV", data=f.read(),
                               file_name="evaluation_results.csv", mime="text/csv")

    st.markdown("### Per Query Breakdown")
    for r in valid:
        faith, relev = r["faithfulness_score"], r["relevancy_score"]
        fe  = "🟢" if faith>=0.75 else "🟡" if faith>=0.60 else "🔴"
        re_ = "🟢" if relev>=0.70 else "🟡" if relev>=0.55 else "🔴"
        with st.expander(f"Q{r['query_id']}  {fe} Faith {faith:.2f}  "
                         f"{re_} Relev {relev:.2f}  —  {r['question'][:55]}…"):
            st.markdown("**Generated Answer**"); st.info(r["answer"])
            st.markdown("**Faithfulness — sentence-level verdicts**")
            sentences = r.get("faithfulness_detail",{}).get("sentences",[])
            if sentences:
                rows = "".join(f"""
                <div class="sent-row">
                    <span class="sent-badge {'sent-ok' if s['supported'] else 'sent-fail'}">
                        {'supported' if s['supported'] else 'not supported'}
                    </span>
                    <span class="sent-sim">{s['max_sim']:.3f}</span>
                    <span>{html.escape(s['sentence'])}</span>
                </div>""" for s in sentences)
                st.markdown(rows, unsafe_allow_html=True)
            else:
                st.caption("No sentence data available.")
            st.markdown("**Scores**")
            sc1, sc2 = st.columns(2)
            sc1.metric("Faithfulness", f"{faith:.4f}",
                       delta="✓ above threshold" if faith>=FAITH_THRESHOLD else "✗ below threshold",
                       delta_color="normal" if faith>=FAITH_THRESHOLD else "inverse")
            sc2.metric("Answer Relevancy", f"{relev:.4f}",
                       delta="✓ good" if relev>=0.70 else "✗ needs improvement",
                       delta_color="normal" if relev>=0.70 else "inverse")
            with st.expander("Retrieved contexts used"):
                for ci, ctx in enumerate(r.get("contexts",[]), 1):
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
    elif st.session_state.page == "kg":
        show_knowledge_graph_page(CHUNKS_PATH)
    else:
        show_evaluation_page()
except Exception as _top_level_err:
    _render_fatal_error(st.session_state.get("page", "unknown"), _top_level_err)
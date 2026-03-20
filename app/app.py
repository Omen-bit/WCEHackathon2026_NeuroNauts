import os
import json
import base64
import requests
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, Collection

load_dotenv()

# ─── CONFIG ─────────────────────────────────────────────

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234")
EMBED_URL = f"{LM_STUDIO_BASE_URL}/v1/embeddings"
CHAT_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"

EMBED_MODEL = os.environ.get("LM_STUDIO_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
CHAT_MODEL = os.environ.get("LM_STUDIO_LLM_MODEL", "gemma-3-4b-it-Q4_K_M.gguf")

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "psychology2e_chunks"

APP_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = APP_DIR.parent
IMAGES_DIR = PROJECT_ROOT / "extracted_images"

TOP_K = 5
MAX_CONTEXT_CHARS = 4000

# ─── PAGE CONFIG ────────────────────────────────────────

st.set_page_config(layout="wide", page_title="AI Search")

# ─── SIDEBAR ────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 AI Search")
    if st.button("New Chat"):
        st.session_state.messages = []
        st.rerun()

# ─── CSS (CRITICAL FIX FOR IMAGE ROW) ───────────────────

st.markdown("""
<style>

/* Layout */
.block-container {
    max-width: 1200px;
    margin: auto;
}

/* Question */
.question {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 1rem;
}

/* Answer */
.answer {
    font-size: 1.05rem;
    line-height: 1.8;
    color: #1f2937;
}

/* IMAGE ROW (FORCED SINGLE LINE 🔥) */
.image-row {
    display: flex;
    flex-wrap: nowrap;   /* IMPORTANT */
    overflow-x: auto;
    gap: 12px;
    padding: 10px 0 20px 0;
}

.image-card {
    min-width: 220px;
    max-width: 220px;
    height: 150px;
    flex-shrink: 0;
    border-radius: 12px;
    overflow: hidden;
}

.image-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* SOURCES RIGHT PANEL */
.source-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.source-title {
    font-weight: 600;
    font-size: 0.9rem;
}

.source-preview {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 4px;
}

.source-page {
    font-size: 0.7rem;
    color: #9ca3af;
    margin-top: 6px;
}

</style>
""", unsafe_allow_html=True)

# ─── LOAD MILVUS ────────────────────────────────────────

@st.cache_resource
def load_milvus():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)
    col.load()
    return col

# ─── FUNCTIONS ──────────────────────────────────────────

def embed_query(q):
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": [q]})
    return r.json()["data"][0]["embedding"]

def retrieve(query):
    col = load_milvus()
    vec = embed_query(query)

    results = col.search(
        data=[vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=TOP_K,
        output_fields=["text","section_path","page_numbers","image_refs"],
    )

    chunks = []
    for hit in results[0]:
        e = hit.entity
        chunks.append({
            "text": e.get("text"),
            "section_path": e.get("section_path"),
            "page_numbers": json.loads(e.get("page_numbers","[]")),
            "image_refs": json.loads(e.get("image_refs","[]"))
        })
    return chunks

def call_llm(q, context):
    r = requests.post(CHAT_URL, json={
        "model": CHAT_MODEL,
        "messages": [
            {"role":"system","content":"Answer only from context."},
            {"role":"user","content":f"{context}\n\nQ:{q}"}
        ]
    })
    return r.json()["choices"][0]["message"]["content"]

def build_context(chunks):
    return "\n\n".join([c["text"] for c in chunks])[:MAX_CONTEXT_CHARS]

def get_images(chunks):
    imgs = []
    for c in chunks:
        for ref in c["image_refs"]:
            path = IMAGES_DIR / ref
            if path.exists():
                imgs.append({
                    "path": str(path),
                    "section": c["section_path"]
                })
    return imgs

# ─── STATE ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── LAYOUT ─────────────────────────────────────────────

main_col, right_col = st.columns([3, 1])

# ─── DISPLAY ────────────────────────────────────────────

for msg in st.session_state.messages:

    if msg["role"] == "user":
        with main_col:
            st.markdown(f"<div class='question'>{msg['content']}</div>", unsafe_allow_html=True)

    else:
        with main_col:

            # 🔥 FIXED IMAGE ROW
            if msg.get("images"):
                st.markdown('<div class="image-row">', unsafe_allow_html=True)

                for img in msg["images"]:
                    with open(img["path"], "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()

                    st.markdown(f"""
                    <div class="image-card">
                        <img src="data:image/png;base64,{b64}">
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Answer
            st.markdown(f"<div class='answer'>{msg['content']}</div>", unsafe_allow_html=True)

        # RIGHT PANEL (WITH PAGE NUMBER ✅)
        with right_col:
            if msg.get("sources"):
                st.markdown("### Sources")

                for i, s in enumerate(msg["sources"], 1):
                    title = s["section_path"].split(">")[-1]
                    preview = s["text"][:80] + "..."
                    page = s["page_numbers"][0] if s["page_numbers"] else "?"

                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-title">[{i}] {title}</div>
                        <div class="source-preview">{preview}</div>
                        <div class="source-page">Page {page}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ─── INPUT ──────────────────────────────────────────────

query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({"role":"user","content":query})

    with st.spinner("Thinking..."):
        chunks = retrieve(query)
        context = build_context(chunks)
        answer = call_llm(query, context)
        images = get_images(chunks)

        st.session_state.messages.append({
            "role":"assistant",
            "content":answer,
            "sources":chunks,
            "images":images
        })

    st.rerun()
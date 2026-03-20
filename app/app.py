import os
import json
import base64
import requests
import streamlit as st
import html
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, Collection
from mindmap_logic import fetch_hierarchy, render_mindmap

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

st.set_page_config(layout="wide", page_title="AI Search")

# ─── SIDEBAR ────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 AI Search")
    if st.button("New Chat"):
        st.session_state.messages = []
        st.rerun()

# ─── CSS + JS ───────────────────────────────────────────

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    margin: auto;
}

/* Question */
.question {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #f3f4f6;
    padding-bottom: 0.75rem;
}

/* Answer */
.answer {
    font-size: 1.05rem;
    line-height: 1.8;
}

/* Image Row */
.img-row {
    display: flex;
    overflow-x: auto;
    gap: 12px;
    margin: 16px 0 24px 0;
}

.img-card {
    flex: 0 0 200px;
    height: 130px;
    border-radius: 12px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.2s ease;
}

.img-card:hover {
    transform: scale(1.05);
}

.img-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* Modal */
.modal {
    display: none;
    position: fixed;
    z-index: 99999;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0,0,0,0.9);
    justify-content: center;
    align-items: center;
}

.modal-img {
    max-width: 90%;
    max-height: 90%;
    border-radius: 12px;
}

.close-btn {
    position: absolute;
    top: 20px;
    right: 30px;
    font-size: 30px;
    color: white;
    cursor: pointer;
}

/* Sources */
.source-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
}

.source-header {
    display: flex;
    gap: 8px;
}

.source-num {
    background: #6366f1;
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
}

.source-title {
    font-weight: 600;
    font-size: 0.82rem;
}

.source-preview {
    font-size: 0.75rem;
    color: #6b7280;
}

.source-page {
    font-size: 0.7rem;
    color: #9ca3af;
}
</style>

<script>
function openModal(id) {
    document.getElementById(id).style.display = "flex";
}

function closeModal(id) {
    document.getElementById(id).style.display = "none";
}
</script>
""", unsafe_allow_html=True)

# ─── MILVUS ─────────────────────────────────────────────

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

    # ── Robust field discovery ──────────────────────────────
    available_fields = [f.name for f in col.schema.fields]
    
    # Try to find the best fields to fetch
    text_f = "clean_text" if "clean_text" in available_fields else ("text" if "text" in available_fields else None)
    full_f = "full_text" if "full_text" in available_fields else ("text" if "text" in available_fields else None)
    
    output_fields = ["section_path", "page_numbers", "image_refs"]
    if text_f: output_fields.append(text_f)
    if full_f and full_f != text_f: output_fields.append(full_f)

    results = col.search(
        data=[vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=TOP_K,
        output_fields=output_fields,
    )

    chunks = []
    for hit in results[0]:
        e = hit.entity
        chunks.append({
            "clean_text": e.get(text_f or "text", "No text found"),
            "full_text": e.get(full_f or "text", "No text found"),
            "section_path": e.get("section_path"),
            "page_numbers": json.loads(e.get("page_numbers","[]")),
            "image_refs": json.loads(e.get("image_refs","[]"))
        })
    return chunks

def build_context(chunks):
    return "\n\n".join([c["full_text"] for c in chunks])[:MAX_CONTEXT_CHARS]

import requests
import os

def call_llm(query, context):
    api_key = os.getenv("NVIDIA_API_KEY")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 🔥 Combine query + context properly
    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

    payload = {
        "model": "meta/llama3-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    else:
        return f"Error: {data}"


def get_images(chunks):
    imgs = []
    for c in chunks:
        for ref in c["image_refs"]:
            path = IMAGES_DIR / ref
            if path.exists():
                imgs.append({"path": str(path)})
    return imgs

def render_image_row(images):
    cards = []

    for idx, img in enumerate(images):
        with open(img["path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        cards.append(f"""
        <div class="img-card" onclick="openModal('modal_{idx}')">
            <img src="data:image/jpeg;base64,{b64}" />
        </div>

        <div id="modal_{idx}" class="modal">
            <span class="close-btn" onclick="closeModal('modal_{idx}')">&times;</span>
            <img src="data:image/jpeg;base64,{b64}" class="modal-img"/>
        </div>
        """)

    html_block = '<div class="img-row">' + "".join(cards) + '</div>'
    st.markdown(html_block, unsafe_allow_html=True)

# ─── STATE ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── UI ─────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🔍 AI Search", "🧠 Mind Map"])

with tab1:
    if not st.session_state.messages:
        st.markdown("## 🧠 Ask anything about Psychology")

    for idx, msg in enumerate(st.session_state.messages):

        if msg["role"] == "user":
            safe_q = html.escape(msg["content"])
            st.markdown(f"<div class='question'>{safe_q}</div>", unsafe_allow_html=True)

        else:
            safe_answer = html.escape(msg["content"])
            st.markdown(f"<div class='answer'>{safe_answer}</div>", unsafe_allow_html=True)

            if msg.get("images"):
                render_image_row(msg["images"])

            if msg.get("sources"):
                with st.expander("Show Sources"):
                    for i, s in enumerate(msg["sources"], 1):
                        preview = html.escape(s.get("clean_text","")[:120]) + "..."
                        pages = ", ".join(map(str, s["page_numbers"]))

                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-header">
                                <span class="source-num">{i}</span>
                                <span class="source-title">{s["section_path"]}</span>
                            </div>
                            <div class="source-preview">{preview}</div>
                            <div class="source-page">📄 Page {pages}</div>
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

with tab2:
    st.markdown("### Interactive Learning Visualization")
    st.info("💡 You can pan, zoom, and collapse branches of the mind map to explore the textbook structure.")
    
    if st.button("Refresh Mind Map"):
        st.session_state.hierarchy_paths = fetch_hierarchy(COLLECTION_NAME)
        st.rerun()

    if "hierarchy_paths" not in st.session_state:
        with st.spinner("Building mind map..."):
            st.session_state.hierarchy_paths = fetch_hierarchy(COLLECTION_NAME)
            
    render_mindmap(st.session_state.hierarchy_paths)

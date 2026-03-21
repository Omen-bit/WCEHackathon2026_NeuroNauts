# NeuroNauts 🧠🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Milvus](https://img.shields.io/badge/Milvus-0DAB76?logo=milvus&logoColor=white)](https://milvus.io/)
[![D3.js](https://img.shields.io/badge/D3.js-F9A03C?logo=d3.js&logoColor=white)](https://d3js.org/)
[![LM Studio](https://img.shields.io/badge/LM_Studio-Gemma_4B-purple)](#)
[![Hybrid RAG](https://img.shields.io/badge/Architecture-Hybrid_RAG-orange)](#)

NeuroNauts is an advanced, AI-powered interactive learning companion focused on Psychology. Designed specifically around the **OpenStax Psychology 2e** textbook, this application utilizes an advanced Hybrid Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers natively from the core material, without hallucinations.

---

## ✨ Features

- **Hybrid RAG Semantic Search**: Combines Dense Vector Search (cosine similarity) and Sparse BM25 Keyword Matching via Milvus for high-precision retrieval of paragraphs, images, and pages.
- **Conversation Aware Context**: Seamlessly handles pronoun-heavy follow up questions (e.g., "what are parts of it?") utilizing memory-enriched AI embeddings.
- **Local Native LLM Stack**: Completely local and offline inference utilizing LM Studio (Gemma models), effectively cutting down API costs while preserving high-speed outputs.
- **Interactive Knowledge Graph (D3.js)**: Visualize textbook architectures dynamically! Explore chapter hierarchies, size-weighted nodes based on volume, and instantly jump to topic chats.
- **Automated Text Validation (Eval Dashboard)**: Evaluates response outputs locally in real-time scoring *faithfulness* and *relevancy* via dense embeddings avoiding Ground Truth demands. 
- **Image Referencing in Chat**: Intelligently grabs figures and charts associated with sections to display natively as Lightbox modals within the chat timeline.

## 🛠️ Key Technologies

- **Frontend & UI**: [Streamlit](https://streamlit.io/) + Custom CSS, HTLM5 Dialogs
- **Vector Database**: [Milvus](https://milvus.io/) (for dense embedding storage and hybrid ranking)
- **Local Server Engine**: [LM Studio](https://lmstudio.ai/)
- **Embeddings**: `nomic-embed-text-v1.5-GGUF`
- **Generative AI Model**: `gemma-3-4b-it-Q4_K_M.gguf`
- **Data Ingestion Frameworks**: [Docling](https://github.com/DS4SD/docling), [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Graph Visualization**: [D3.js](https://d3js.org/)

## 🚀 Quick Start & Installation

### 1. Prerequisites
* Python 3.10+
* [Milvus Standalone](https://milvus.io/docs/install_standalone-docker.md) running locally (`localhost:19530`)
* LM Studio running on `http://localhost:1234` with developer server active. Ensure you have loaded the specified embedding and chat models.

### 2. Clone the Repository
```bash
git clone https://github.com/YourUsername/WCEHackathon2026_NeuroNauts.git
cd WCEHackathon2026_NeuroNauts
```

### 3. Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Application Configuration
Copy your environment variables and place your models.
```bash
# Rename the sample env
# Or create a .env with:
LM_STUDIO_BASE_URL="http://localhost:1234"
LM_STUDIO_MODEL="nomic-ai/nomic-embed-text-v1.5-GGUF"
LM_STUDIO_LLM_MODEL="gemma-3-4b-it-Q4_K_M.gguf"
```

### 5. Launch the Application
Start up the Streamlit frontend.
```bash
streamlit run app/app.py
```

## 🧠 Application Architecture

1. **Ingestion & Chunking**: PyMuPDF handles image extraction. Docling handles parsing the raw textbook PDF. Content is semantically chunked and merged preserving chapters, pages, headings, and related images.
2. **Retrieval Module (`pipeline/retrieve.py`)**: Marries standard keyword retrieval (BM25) with vector semantic search (Milvus embeddings) via reciprocal rank fusion (RRF). 
3. **Generation (`app/generate.py` & `app/app.py`)**: Chunks are assembled and constrained within context budgets and shipped to Gemma via standard OpenAI-compatible requests protocols.
4. **Evaluation (`pipeline/` and Eval Page)**: Checks responses recursively mapping generated sentence vectors directly against retrieved context vectors measuring cosine thresholds.

## 🤝 Project Submissions

Developed for **WCE Hackathon 2026**. Designed to radically overhaul textbook traversal and data retention among modern students.

--- 
*Note: Due to the high payload context of large textbooks, a system with at least 16GB of RAM is highly recommended when evaluating local offline models.*

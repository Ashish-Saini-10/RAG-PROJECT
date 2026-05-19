# 🔬 ShowYourWork — Academic & Research RAG Assistant

> **Strict Verification RAG**: Every answer is grounded exclusively in your uploaded documents, with mandatory inline citations and full source traceability.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🚫 Zero Hallucination | LLM is hard-blocked from using pre-trained knowledge |
| 📎 Inline Citations | Every claim cites `[Filename, Page X]` |
| 📊 Dual-Pane UI | Answer left · Source evidence right |
| 📈 Similarity Scores | Cosine similarity % for every retrieved chunk |
| ⚡ Groq Streaming | Near-instant responses via LLaMA 3.1-8B |
| 🧠 Local Embeddings | `all-MiniLM-L6-v2` — 100% free, runs on CPU |
| 🗄️ Local Vector DB | ChromaDB persisted to disk, no cloud needed |

---

## 🏗️ Architecture

```
[PDF Files] → [PyPDFLoader] → [RecursiveCharacterTextSplitter]
                                          ↓
                           [all-MiniLM-L6-v2 Embeddings]
                                          ↓
                              [ChromaDB (local disk)]
                                          ↓
[User Query] → [Cosine Similarity Search] → [Top-K Chunks]
                                          ↓
                    [Zero-Knowledge System Prompt + Context]
                                          ↓
                         [Groq Cloud · LLaMA 3.1-8B-Instant]
                                          ↓
                    [Streamed Answer + Source Evidence Cards]
```

---

## 🚀 Quick Start (Local)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/show-your-work-rag.git
cd show-your-work-rag
pip install -r requirements.txt
```

### 2. Set Your Groq API Key

Get a free key at [console.groq.com/keys](https://console.groq.com/keys)

```bash
# Copy the example file
cp .env.example .env

# Edit .env and paste your key
GROQ_API_KEY=gsk_your_key_here
```

### 3. Add Your PDFs

```bash
# Drop your PDF research papers into the data/ folder
cp your_paper.pdf data/
```

### 4. Ingest Documents (One-time)

```bash
python ingest.py
```

### 5. Launch the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

> **Tip:** You can also upload PDFs and ingest directly from the sidebar inside the app — no CLI needed!

---

## 📁 Project Structure

```
show-your-work-rag/
│
├── app.py              # 🖥️  Streamlit UI (Phase 7)
├── ingest.py           # 📥  PDF ingestion pipeline (Phases 2–4)
├── rag_engine.py       # 🔍  Retrieval + Groq LLM engine (Phases 5–6)
│
├── data/               # 📂  Place your PDF files here
├── chroma_db/          # 🗄️  Auto-created vector index (after ingest)
│
├── .streamlit/
│   ├── config.toml     # 🎨  Dark theme config
│   └── secrets.toml    # 🔑  Local secrets (git-ignored)
│
├── requirements.txt    # 📦  Dependencies
├── .env.example        # 🔧  Environment variable template
└── .gitignore
```

---

## ☁️ Deploy to Streamlit Cloud (Phase 8)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "feat: initial ShowYourWork RAG app"
git remote add origin https://github.com/YOUR_USERNAME/show-your-work-rag.git
git push -u origin main
```

> ⚠️ **Important**: The `chroma_db/` folder is NOT ignored by default so your pre-built index travels with the repo. If your index is large (>100MB), consider rebuilding it on first run or use `git-lfs`.

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New App** → select your GitHub repo
3. Set **Main file path**: `app.py`
4. Click **Deploy**

### 3. Add Secret in Cloud Dashboard

In Streamlit Cloud → **App Settings** → **Secrets**, add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

The app reads from `st.secrets` automatically — your key is **never exposed** in code or repo.

---

## 🔧 Configuration

| Variable | Location | Description |
|---|---|---|
| `GROQ_API_KEY` | `.env` / Streamlit Secrets | Groq Cloud API key |
| `CHUNK_SIZE` | `ingest.py` | Token chunk size (default: 512) |
| `CHUNK_OVERLAP` | `ingest.py` | Overlap between chunks (default: 50) |
| `DEFAULT_TOP_K` | `rag_engine.py` | Retrieved chunks per query (default: 3) |
| `GROQ_MODEL` | `rag_engine.py` | LLM model ID |
| `EMBEDDING_MODEL` | `rag_engine.py` | HuggingFace sentence-transformer |

---

## 🧪 Tech Stack

- **UI**: [Streamlit](https://streamlit.io)
- **LLM**: [Groq](https://groq.com) · `llama-3.1-8b-instant`
- **Embeddings**: [sentence-transformers](https://sbert.net) · `all-MiniLM-L6-v2`
- **Vector DB**: [ChromaDB](https://www.trychroma.com)
- **PDF Parsing**: [PyPDF](https://pypdf.readthedocs.io) via LangChain
- **Orchestration**: [LangChain](https://langchain.com)

---

## 📜 License

MIT — use freely, cite responsibly.

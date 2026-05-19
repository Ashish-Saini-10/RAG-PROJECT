"""
app.py — CiteMind Academic RAG Assistant
=============================================
Phase 7: Split-screen Streamlit UI — Clean & Aligned
"""

import os
# Force protobuf to use pure-Python implementation to avoid descriptor conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
try:
    import google.protobuf
    pb_ver = google.protobuf.__version__
except Exception as e:
    pb_ver = f"Error: {e}"

with open("debug_env.txt", "w") as f:
    f.write(f"sys.executable: {sys.executable}\n")
    f.write(f"sys.version: {sys.version}\n")
    f.write(f"protobuf version: {pb_ver}\n")
    f.write(f"sys.path: {sys.path}\n")

from pathlib import Path
import shutil

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Clean up any leftover persistent directories on disk to free up space
if "disk_cleaned" not in st.session_state:
    st.session_state["disk_cleaned"] = True
    # Delete PDF files inside ./data
    data_path = Path("./data")
    if data_path.exists() and data_path.is_dir():
        for f in data_path.glob("*.pdf"):
            try:
                f.unlink()
            except Exception:
                pass
    # Delete ./chroma_db folder completely
    chroma_path = Path("./chroma_db")
    if chroma_path.exists() and chroma_path.is_dir():
        try:
            shutil.rmtree(chroma_path)
        except Exception:
            pass

if "chroma_client" not in st.session_state:
    import chromadb
    st.session_state["chroma_client"] = chromadb.EphemeralClient()

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CiteMind — RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ───────────────────────────────────────────────── */
:root {
  --bg:        #080c18;
  --bg1:       #0d1120;
  --bg2:       #111827;
  --card:      #141d2e;
  --card-h:    #1a2540;
  --blue:      #3b82f6;
  --violet:    #7c3aed;
  --cyan:      #06b6d4;
  --green:     #10b981;
  --amber:     #f59e0b;
  --red:       #ef4444;
  --t1:        #f1f5f9;
  --t2:        #94a3b8;
  --t3:        #475569;
  --border:    rgba(59,130,246,0.12);
  --border-a:  rgba(59,130,246,0.40);
}

/* ── Reset ────────────────────────────────────────────────── */
html, body, .stApp { background: var(--bg) !important; font-family: 'Inter', sans-serif !important; }
.block-container   { padding: 0 2rem 2rem !important; max-width: 1400px !important; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; }

/* ── Mesh background ──────────────────────────────────────── */
.stApp::before {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(ellipse 70% 50% at 15% 5%,  rgba(59,130,246,0.08) 0%, transparent 65%),
    radial-gradient(ellipse 55% 45% at 85% 8%,  rgba(124,58,237,0.07) 0%, transparent 65%),
    radial-gradient(ellipse 45% 35% at 50% 95%, rgba(6,182,212,0.05)  0%, transparent 65%);
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
  padding-top: 0 !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem !important; }
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* ── Header ───────────────────────────────────────────────── */
.app-header {
  text-align: center;
  padding: 2rem 1rem 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.app-logo {
  font-size: 2.8rem;
  display: block;
  margin-bottom: 0.4rem;
  filter: drop-shadow(0 0 18px rgba(59,130,246,0.55));
}
.app-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  background: linear-gradient(135deg, #60a5fa, #a78bfa, #22d3ee);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.3rem;
  line-height: 1.1;
}
.app-sub {
  font-size: 0.78rem;
  color: var(--t3);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* ── Section labels ───────────────────────────────────────── */
.sec-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--t3);
  margin: 1.2rem 0 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.sec-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Pills ────────────────────────────────────────────────── */
.pill {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.22rem 0.65rem;
  border-radius: 999px;
  font-size: 0.7rem; font-weight: 600;
}
.pill-ok   { background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.25); }
.pill-warn { background:rgba(245,158,11,0.12); color:#fbbf24; border:1px solid rgba(245,158,11,0.25); }
.pill-err  { background:rgba(239,68,68,0.12);  color:#f87171; border:1px solid rgba(239,68,68,0.25);  }

/* ── Streamlit input overrides ────────────────────────────── */
.stTextArea textarea, .stTextInput input {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--t1) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.88rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
  outline: none !important;
}

/* ── All buttons ──────────────────────────────────────────── */
.stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 9px !important;
  font-weight: 600 !important;
  font-size: 0.83rem !important;
  letter-spacing: 0.01em !important;
  padding: 0.6rem 1rem !important;
  transition: opacity 0.18s, transform 0.15s, box-shadow 0.2s !important;
  font-family: 'Inter', sans-serif !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 5px 20px rgba(59,130,246,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; opacity: 1 !important; }

/* ── Slider ───────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important;
  border: 1px dashed var(--border-a) !important;
  border-radius: 10px !important;
  min-height: 90px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  padding: 1rem !important;
}
/* Hide all native child labels and buttons to eliminate translation overlapping entirely */
[data-testid="stFileUploaderDropzone"] > * {
  display: none !important;
}
/* Keep the native file input active and stretched across the dropzone so clicking still works */
[data-testid="stFileUploaderDropzone"] input[type="file"] {
  display: block !important;
  position: absolute !important;
  inset: 0 !important;
  opacity: 0 !important;
  z-index: 2 !important;
  cursor: pointer !important;
}
/* Render our own clean, premium, single-label placeholder */
[data-testid="stFileUploaderDropzone"]::after {
  content: "📁 Drag & Drop or Click to Upload" !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  color: var(--t2) !important;
  pointer-events: none !important;
  z-index: 1 !important;
}

/* ── Checkbox ─────────────────────────────────────────────── */
.stCheckbox label { font-size: 0.82rem !important; color: var(--t2) !important; }

/* ── Divider ──────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

/* ── Query bar container ──────────────────────────────────── */
.query-wrap {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem 1.2rem 0.8rem;
  margin-bottom: 1.2rem;
  box-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.query-label {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--t3);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}

/* ── Panel headers ────────────────────────────────────────── */
.panel-hdr {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.6rem 0.9rem;
  border-radius: 9px;
  margin-bottom: 1rem;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}
.phdr-blue {
  background: rgba(59,130,246,0.08);
  border: 1px solid rgba(59,130,246,0.2);
  color: #93c5fd;
}
.phdr-green {
  background: rgba(16,185,129,0.08);
  border: 1px solid rgba(16,185,129,0.2);
  color: #6ee7b7;
}
.blink {
  width: 7px; height: 7px; border-radius: 50%;
  animation: blink-anim 2s ease-in-out infinite;
  flex-shrink: 0;
}
.blink-blue  { background: var(--blue);  box-shadow: 0 0 6px var(--blue); }
.blink-green { background: var(--green); box-shadow: 0 0 6px var(--green); }
@keyframes blink-anim {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.5; transform:scale(0.8); }
}

/* ── Answer box ───────────────────────────────────────────── */
.ans-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem 1.6rem;
  line-height: 1.85;
  color: var(--t1);
  font-size: 0.9rem;
  min-height: 220px;
}

/* ── Source card ──────────────────────────────────────────── */
.src-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 11px;
  padding: 0.85rem 1rem;
  margin-bottom: 0.75rem;
  transition: border-color 0.2s, transform 0.15s;
  position: relative;
  overflow: hidden;
}
.src-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  border-radius: 11px 0 0 11px;
}
.src-card:hover { border-color: rgba(59,130,246,0.35); transform: translateX(2px); }
.card-high::before { background: linear-gradient(180deg, #10b981, #06b6d4); }
.card-mid::before  { background: linear-gradient(180deg, #f59e0b, #f97316); }
.card-low::before  { background: linear-gradient(180deg, #ef4444, #ec4899); }

.card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.6rem;
}
.card-left { display: flex; align-items: center; gap: 0.5rem; min-width: 0; flex: 1; }
.rank-badge {
  width: 20px; height: 20px;
  border-radius: 50%;
  font-size: 0.6rem; font-weight: 800;
  background: linear-gradient(135deg, var(--blue), var(--violet));
  color: white;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.card-fname {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--t1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.card-page { font-size: 0.68rem; color: var(--t3); margin-top: 1px; }

.score-pill {
  font-size: 0.68rem; font-weight: 700;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  flex-shrink: 0;
}
.sp-high { background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); }
.sp-mid  { background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3); }
.sp-low  { background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(239,68,68,0.3);  }

.score-bar { height: 3px; background: rgba(255,255,255,0.06); border-radius: 2px; margin: 0.4rem 0 0.5rem; overflow: hidden; }
.score-fill { height: 100%; border-radius: 2px; }

.card-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--t2);
  line-height: 1.6;
  border-left: 2px solid rgba(59,130,246,0.3);
  padding-left: 0.65rem;
  max-height: 100px;
  overflow: hidden;
  position: relative;
}
.card-text::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 24px;
  background: linear-gradient(transparent, var(--card));
}

/* ── Empty states ─────────────────────────────────────────── */
.empty {
  text-align: center;
  padding: 4rem 1rem;
  color: var(--t3);
}
.empty-icon { font-size: 2.8rem; opacity: 0.45; display: block; margin-bottom: 0.7rem; }
.empty-title { font-size: 0.88rem; font-weight: 600; color: var(--t2); margin-bottom: 0.25rem; }
.empty-sub   { font-size: 0.76rem; line-height: 1.6; }

/* ── Indexed file items ───────────────────────────────────── */
.file-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.5rem;
  border-radius: 6px;
  background: rgba(59,130,246,0.05);
  border: 1px solid rgba(59,130,246,0.1);
  margin-bottom: 0.3rem;
  font-size: 0.72rem;
  color: var(--t2);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _resolve_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:    key = st.secrets.get("GROQ_API_KEY", "")
        except: pass
    if not key:
        key = st.session_state.get("manual_api_key", "")
    return key


def _db_info() -> dict:
    try:
        if "chroma_client" in st.session_state:
            client = st.session_state["chroma_client"]
        else:
            import chromadb
            client = chromadb.EphemeralClient()
        col    = client.get_or_create_collection("research_docs")
        count  = col.count()
        if count == 0:
            return {"count": 0, "files": []}
        meta  = col.get(include=["metadatas"])["metadatas"] or []
        files = sorted(set(m.get("filename", "unknown") for m in meta))
        return {"count": count, "files": files}
    except Exception:
        return {"count": 0, "files": []}


def _score_style(s: float):
    if s >= 65: return "card-high", "sp-high", "#10b981"
    if s >= 40: return "card-mid",  "sp-mid",  "#f59e0b"
    return         "card-low",  "sp-low",  "#ef4444"


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem;">
      <div style="font-size:2rem; filter:drop-shadow(0 0 12px rgba(59,130,246,0.6));">🔬</div>
      <div style="font-size:1.1rem; font-weight:800; letter-spacing:-0.02em;
                  background:linear-gradient(135deg,#60a5fa,#a78bfa);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text; margin-top:0.2rem;">CiteMind</div>
      <div style="font-size:0.65rem; color:#475569; margin-top:0.2rem; letter-spacing:0.06em; text-transform:uppercase;">RAG Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── API Key ──────────────────────────────────────────────
    st.markdown('<div class="sec-label">API Key</div>', unsafe_allow_html=True)
    api_key = _resolve_api_key()
    if api_key:
        st.markdown('<span class="pill pill-ok">● Groq Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-err">● Key Missing</span>', unsafe_allow_html=True)
        new_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
        )
        if new_key:
            st.session_state["manual_api_key"] = new_key
            os.environ["GROQ_API_KEY"] = new_key
            st.rerun()
        st.caption("Get free key at [console.groq.com](https://console.groq.com/keys)")

    st.divider()

    # ── Settings ─────────────────────────────────────────────
    st.markdown('<div class="sec-label">Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K Chunks", 1, 6, 3, help="How many document chunks to retrieve")

    st.divider()

    # ── Upload ───────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    replace_mode = st.checkbox("Replace existing index", value=True,
        help="ON = fresh index (clears old). OFF = append to existing.")

    if uploaded:
        if st.button("Ingest Documents", use_container_width=True):
            bar = st.progress(0, text="Processing files...")
            
            import tempfile
            import chromadb

            # Create a temporary directory that is automatically deleted when the context manager exits
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                for i, uf in enumerate(uploaded):
                    (temp_path / uf.name).write_bytes(uf.getbuffer())
                    bar.progress((i + 1) / (len(uploaded) + 2), text=f"Saved {uf.name}")

                bar.progress(0.65, text="Embedding chunks...")
                try:
                    if replace_mode:
                        st.session_state["chroma_client"] = chromadb.EphemeralClient()

                    from ingest import main as run_ingest
                    import rag_engine
                    
                    result = run_ingest(data_dir=str(temp_path), client=st.session_state["chroma_client"])
                    rag_engine.reset_singletons()
                    
                    bar.progress(1.0, text="Complete!")
                    st.success(
                        f"**{result['num_docs']}** doc(s) · **{result['num_pages']}** pages "
                        f"· **{result['num_chunks']}** chunks indexed"
                    )
                    st.rerun()
                except Exception as e:
                    bar.empty()
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # ── DB Status ────────────────────────────────────────────
    st.markdown('<div class="sec-label">Knowledge Base</div>', unsafe_allow_html=True)
    db = _db_info()
    if db["count"] > 0:
        st.markdown('<span class="pill pill-ok">● Index Ready</span>', unsafe_allow_html=True)
        st.caption(f"**{db['count']:,}** chunks · **{len(db['files'])}** document(s)")
        if db["files"]:
            for fname in db["files"]:
                st.markdown(
                    f'<div class="file-item">📄 {fname}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Database", key="clear_db_btn", use_container_width=True):
            import chromadb
            import rag_engine
            import shutil
            
            # Reset in-memory client
            st.session_state["chroma_client"] = chromadb.EphemeralClient()
            rag_engine.reset_singletons()
            
            # Try to physically clean up disk directories as well
            for folder in ["./chroma_db", "./data"]:
                path = Path(folder)
                if path.exists() and path.is_dir():
                    try:
                        shutil.rmtree(path)
                    except Exception:
                        # If locked, try deleting individual un-locked files inside
                        for f in path.glob("**/*"):
                            if f.is_file():
                                try:
                                    f.unlink()
                                except Exception:
                                    pass
            
            st.success("Database cleared!")
            st.rerun()
    else:
        st.markdown('<span class="pill pill-warn">● No Index</span>', unsafe_allow_html=True)
        st.caption("Upload PDFs above to get started.")

    st.divider()
    st.caption("🔒 Zero-Knowledge Mode · All answers cited from your documents only.")

    try:
        import google.protobuf
        st.markdown(f'<div style="font-size:0.65rem; color:#475569; text-align:center; margin-top:0.5rem;">System Diagnostics:<br>Protobuf v{google.protobuf.__version__}</div>', unsafe_allow_html=True)
    except:
        pass


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span class="app-logo">🔬</span>
  <h1 class="app-title">CiteMind</h1>
  <p class="app-sub">Strict Verification · Academic RAG · Groq + LLaMA 3.1</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Query Bar
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">🔍 Research Question</div>', unsafe_allow_html=True)

q_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
with q_col:
    query = st.text_area(
        "q",
        placeholder="e.g. What methodology was used? What are the key findings? Summarise the conclusions.",
        height=80,
        label_visibility="collapsed",
    )
with btn_col:
    analyze = st.button("Analyze", use_container_width=True)



# ─────────────────────────────────────────────────────────────
# Results — Dual Pane
# ─────────────────────────────────────────────────────────────
col_ans, col_src = st.columns([1.3, 1], gap="large")

with col_ans:
    st.markdown("""
    <div class="panel-hdr phdr-blue">
      <span class="blink blink-blue"></span>
      Synthesized Answer
    </div>
    """, unsafe_allow_html=True)

with col_src:
    st.markdown("""
    <div class="panel-hdr phdr-green">
      <span class="blink blink-green"></span>
      Source Evidence
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Process
# ─────────────────────────────────────────────────────────────
if analyze and query.strip():

    api_key = _resolve_api_key()

    if not api_key:
        st.error("Groq API key not set. Enter it in the sidebar.")
        st.stop()

    db = _db_info()
    if db["count"] == 0:
        st.error("No documents indexed. Upload PDFs in the sidebar first.")
        st.stop()

    try:
        from rag_engine import retrieve_chunks, build_context, SYSTEM_PROMPT_TEMPLATE, GROQ_MODEL
        from groq import Groq

        with st.spinner("Scanning index..."):
            chunks = retrieve_chunks(query.strip(), k=top_k)

        if not chunks:
            st.warning("No matching chunks found. Try rephrasing.")
            st.stop()

        # ── Source cards ──────────────────────────────────────
        with col_src:
            for i, c in enumerate(chunks, 1):
                card_cls, sp_cls, bar_col = _score_style(c["score_pct"])
                txt = c["text"][:380].replace("<","&lt;").replace(">","&gt;")
                if len(c["text"]) > 380: txt += "…"

                st.markdown(f"""
<div class="src-card {card_cls}">
  <div class="card-top">
    <div class="card-left">
      <span class="rank-badge">{i}</span>
      <div>
        <div class="card-fname">📄 {c['filename']}</div>
        <div class="card-page">Page {c['page']}</div>
      </div>
    </div>
    <span class="score-pill {sp_cls}">{c['score_pct']}%</span>
  </div>
  <div class="score-bar">
    <div class="score-fill" style="width:{c['score_pct']}%;background:linear-gradient(90deg,{bar_col},{bar_col}99);"></div>
  </div>
  <div class="card-text">{txt}</div>
</div>""", unsafe_allow_html=True)

        # ── Stream answer ─────────────────────────────────────
        with col_ans:
            ctx    = build_context(chunks)
            prompt = SYSTEM_PROMPT_TEMPLATE.format(context=ctx)
            client = Groq(api_key=api_key)
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": query.strip()},
                ],
                temperature=0.05,
                max_tokens=1500,
                stream=True,
            )
            holder, answer = st.empty(), ""
            for ev in stream:
                tok = ev.choices[0].delta.content
                if tok:
                    answer += tok
                    holder.markdown(
                        f'<div class="ans-box">{answer}<span style="color:var(--blue)">▌</span></div>',
                        unsafe_allow_html=True,
                    )
            holder.markdown(f'<div class="ans-box">{answer}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)


# ─────────────────────────────────────────────────────────────
# Empty States
# ─────────────────────────────────────────────────────────────
elif not analyze:
    with col_ans:
        st.markdown("""
        <div class="empty">
          <span class="empty-icon">💡</span>
          <div class="empty-title">Ready to Analyze</div>
          <div class="empty-sub">Type your research question above<br>and click <strong>Analyze</strong></div>
        </div>""", unsafe_allow_html=True)

    with col_src:
        st.markdown("""
        <div class="empty">
          <span class="empty-icon">📑</span>
          <div class="empty-title">Evidence Vault</div>
          <div class="empty-sub">Matching document chunks appear here<br>with page numbers and similarity scores</div>
        </div>""", unsafe_allow_html=True)

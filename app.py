"""
Multimodal Document Intelligence System — Streamlit UI
Supports: PDF, TXT, DOCX, MD upload · Semantic search · Citation-grounded answers
"""

import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from rag_pipeline import RAGPipeline, RetrievedChunk


st.set_page_config(
    page_title="DocIntel · Document Intelligence",
    
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0d0f12;
    color: #e8eaf0;
}

.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #64ffda;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: #8892a4;
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: #141720;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
}

.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #556070;
    margin-bottom: 2px;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    color: #64ffda;
    font-weight: 600;
}

.citation-block {
    background: #141720;
    border-left: 3px solid #64ffda;
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
}

.citation-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #64ffda;
    margin-bottom: 0.4rem;
}

.citation-text {
    color: #aab4c0;
    line-height: 1.6;
}

.score-bar-bg {
    background: #1e2535;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}

.score-bar-fill {
    background: linear-gradient(90deg, #64ffda, #00bcd4);
    border-radius: 4px;
    height: 6px;
}

.answer-box {
    background: #141720;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    line-height: 1.8;
    color: #d8dde8;
    font-size: 0.97rem;
}

.stButton > button {
    background: #64ffda;
    color: #0d0f12;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 6px;
    padding: 0.55rem 1.4rem;
    transition: opacity 0.15s;
}

.stButton > button:hover { opacity: 0.85; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #141720 !important;
    border: 1px solid #1e2535 !important;
    color: #e8eaf0 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #64ffda !important;
    box-shadow: 0 0 0 1px #64ffda33 !important;
}

div[data-testid="stFileUploader"] {
    background: #141720;
    border: 1.5px dashed #1e2535;
    border-radius: 10px;
    padding: 1rem;
}

section[data-testid="stSidebar"] {
    background: #0a0c0f;
    border-right: 1px solid #1a1f2c;
}

.tag {
    display: inline-block;
    background: #1e2535;
    color: #64ffda;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 4px;
    margin-bottom: 4px;
}

hr { border-color: #1a1f2c !important; }

.stSpinner > div { border-top-color: #64ffda !important; }

/* Sidebar labels */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #c8d0dc !important;
    font-size: 0.93rem !important;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] strong {
    color: #ffffff !important;
    font-size: 0.96rem !important;
}

/* Slider label */
section[data-testid="stSidebar"] .stSlider label p {
    color: #e0e6f0 !important;
    font-size: 0.93rem !important;
    font-weight: 600 !important;
}

/* Slider value */
section[data-testid="stSidebar"] .stSlider span {
    color: #64ffda !important;
    font-family: IBM Plex Mono, monospace !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

/* Info/alert box */
section[data-testid="stSidebar"] .stAlert {
    background: #0d1f1a !important;
    border: 1px solid #64ffda55 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] .stAlert p {
    color: #64ffda !important;
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)


def init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []
    if "history" not in st.session_state:
        st.session_state.history = []   # list of (question, response)

init_session()


with st.sidebar:
    st.markdown("<div style='padding:0.4rem 0 1.2rem'><span style='font-family:IBM Plex Mono;font-size:1.1rem;color:#64ffda;font-weight:600;'>⬡ DocIntel</span></div>", unsafe_allow_html=True)

    st.markdown("**Configuration**")
    st.info(" Running FREE local model\n\nNo API key needed!")
    top_k = st.slider("Retrieved Chunks (top-k)", 3, 10, 5)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512, step=64)

    st.divider()

    st.markdown("**Upload Documents**")
    uploaded = st.file_uploader(
        "PDF, TXT, DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    ingest_btn = st.button("⬡ Build Index", use_container_width=True)

    if ingest_btn and uploaded:
        pipeline = RAGPipeline(
            top_k=top_k,
            chunk_size=chunk_size,
        )
        with st.spinner("Embedding documents… (first run downloads model ~1GB, please wait)"):
            names = []
            for f in uploaded:
                suffix = Path(f.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                result = pipeline.ingest(tmp_path)
                names.append(f.name)
                os.unlink(tmp_path)

        st.session_state.pipeline = pipeline
        st.session_state.ingested_files = names
        st.success(f"Indexed {len(names)} file(s) · {pipeline.num_docs} chunks")

    if st.session_state.ingested_files:
        st.divider()
        st.markdown("**Indexed Files**")
        for f in st.session_state.ingested_files:
            st.markdown(f"<span class='tag'> {f}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem;color:#3a4555;'>FAISS · MiniLM-L6-v2<br>LangChain · GPT-4o-mini</div>",
        unsafe_allow_html=True,
    )


st.markdown("<div class='hero-title'>Document Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Semantic retrieval · Citation-grounded answers · FAISS + MiniLM</div>", unsafe_allow_html=True)


query = st.text_input(
    "Ask a question",
    placeholder="What are the main findings in the document?",
    label_visibility="collapsed",
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_btn = st.button("Ask →")
with col_clear:
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()


if ask_btn and query:
    if not st.session_state.pipeline:
        st.warning("Upload and index documents first (sidebar).")
    else:
        with st.spinner("Retrieving & generating…"):
            response = st.session_state.pipeline.query(query)
        st.session_state.history.insert(0, (query, response))


for q, resp in st.session_state.history:
    st.divider()

    col_q, col_m = st.columns([3, 1])

    with col_q:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"<div class='answer-box'>{resp.answer}</div>", unsafe_allow_html=True)

    with col_m:
        st.markdown("<div class='metric-card'><div class='metric-label'>Retrieval</div>"
                    f"<div class='metric-value'>{resp.retrieval_time*1000:.0f}ms</div></div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='metric-card'><div class='metric-label'>Generation</div>"
                    f"<div class='metric-value'>{resp.generation_time:.2f}s</div></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Chunks used</div>"
                    f"<div class='metric-value'>{len(resp.citations)}</div></div>",
                    unsafe_allow_html=True)

    # Citations
    if resp.citations:
        st.markdown("**Retrieved Context**")
        for i, c in enumerate(resp.citations, 1):
            score_pct = int(c.score * 100)
            preview = c.content[:260].replace("\n", " ")
            st.markdown(f"""
<div class='citation-block'>
  <div class='citation-meta'>#{i} · {c.source} · Page {c.page} · Score: {c.score:.3f}</div>
  <div class='score-bar-bg'><div class='score-bar-fill' style='width:{score_pct}%'></div></div>
  <div class='citation-text' style='margin-top:8px'>{preview}…</div>
</div>
""", unsafe_allow_html=True)
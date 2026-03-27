# DocIntel — Document Intelligence System
 
A multimodal RAG (Retrieval-Augmented Generation) app that lets you upload documents and ask questions with citation-grounded answers.
 
Built with **FAISS**, **MiniLM-L6-v2**, **LangChain**, and **Streamlit**.
 
---
 
## Features
 
- Upload **PDF, TXT, DOCX** documents
- Semantic search using **FAISS** vector store
- Citation-grounded answers with source, page, and relevance score
- Fully local embedding model — **no API key needed**
- Clean dark UI with retrieval + generation time metrics
 
---
 
## Project Structure
 
```
LLM_Project/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG logic (ingest, embed, retrieve, generate)
├── .gitignore
├── requirements.txt
└── README.md
```
 
---
 
## Setup & Installation
 
### 1. Clone the repository
```bash
git clone https://github.com/SahityaKotla123/Document-Intelligence-RAG.git
cd Document-Intelligence-RAG
```
 
### 2. Create and activate virtual environment
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# Mac/Linux
source venv/bin/activate
```
 
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 4. Run the app
```bash
streamlit run app.py
```
 
---
 
## How to Use
 
1. Open the app in your browser (usually `http://localhost:8501`)
2. Upload one or more documents using the sidebar
3. Click **Build Index** to embed and index the documents
4. Type your question in the search bar and click **Ask →**
5. View the AI-generated answer with citations and relevance scores
 
---
 
## Tech Stack
 
| Component | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | MiniLM-L6-v2 (local, free) |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| LLM | GPT-4o-mini |
 
---
 
## Notes
 
- First run will download the embedding model (~1GB), please be patient
- `venv/` is excluded from Git tracking via `.gitignore`
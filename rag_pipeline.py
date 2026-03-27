"""
Multimodal Document Intelligence System - RAG Pipeline
End-to-end Retrieval Augmented Generation using LangChain + FAISS + MiniLM
Uses Ollama (local, free, fast) - no API key needed!
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: int
    score: float
    chunk_id: int


@dataclass
class RAGResponse:
    answer: str
    citations: list[RetrievedChunk]
    retrieval_time: float
    generation_time: float
    model: str = "tinyllama (ollama)"


class FAISSVectorStore:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DIM = 384

    def __init__(self):
        self.encoder = SentenceTransformer(self.EMBED_MODEL)
        self.index = faiss.IndexFlatIP(self.DIM)
        self.chunks: list[Document] = []

    def add_documents(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        texts = [d.page_content for d in docs]
        embeddings = self._embed(texts)
        self.index.add(embeddings)
        self.chunks.extend(docs)
        return len(docs)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype("float32")

    def similarity_search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q_vec = self._embed([query])
        scores, indices = self.index.search(q_vec, k)
        results: list[RetrievedChunk] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            doc = self.chunks[idx]
            results.append(RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 0),
                score=float(score),
                chunk_id=int(idx),
            ))
        return results

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal


LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
}

def load_document(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(file_path).load()


def chunk_documents(docs, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)



class OllamaLLM:
    def __init__(self, model: str = "tinyllama"):
        self.model = model

    def generate(self, context: str, question: str) -> str:
        try:
            import ollama
            max_ctx = 2000
            if len(context) > max_ctx:
                context = context[:max_ctx] + "..."

            prompt = f"""You are a helpful document assistant. Answer the question using only the context below.
Always mention the source file name when referring to information.
If the answer is not in the context, say "I cannot find this in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

            response = ollama.generate(model=self.model, prompt=prompt)
            return response["response"].strip()

        except Exception as e:
            return f"Ollama error: {str(e)}\n\nMake sure:\n1. Ollama is installed and running\n2. You ran: ollama pull tinyllama"


class RAGPipeline:
    def __init__(self, top_k=5, chunk_size=512, chunk_overlap=64, **kwargs):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = FAISSVectorStore()
        self.llm = OllamaLLM(model="tinyllama")

    def ingest(self, file_path: str) -> dict:
        raw_docs = load_document(file_path)
        chunks = chunk_documents(raw_docs, self.chunk_size, self.chunk_overlap)
        for c in chunks:
            c.metadata["source"] = Path(file_path).name
        n = self.vector_store.add_documents(chunks)
        return {"file": Path(file_path).name, "chunks_added": n, "total_chunks": self.vector_store.total_chunks}

    def query(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        citations = self.vector_store.similarity_search(question, k=self.top_k)
        retrieval_time = time.perf_counter() - t0

        context = "\n\n".join(
            f"[{c.source} | Page {c.page}]\n{c.content}" for c in citations
        )

        t1 = time.perf_counter()
        answer = self.llm.generate(context, question)
        generation_time = time.perf_counter() - t1

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
        )

    @property
    def num_docs(self):
        return self.vector_store.total_chunks
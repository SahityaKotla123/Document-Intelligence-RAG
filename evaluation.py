"""
RAGAS Evaluation Module
Measures: Faithfulness, Answer Relevancy, Context Precision, Context Recall
"""

import json
import statistics
from dataclasses import dataclass, field
from typing import Optional

from sentence_transformers import SentenceTransformer
import numpy as np


_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _encoder


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]   


@dataclass
class EvalResult:
    faithfulness: float       
    answer_relevancy: float   
    context_precision: float  
    context_recall: float     
    sample_count: int = 0

    def to_dict(self):
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "sample_count": self.sample_count,
        }


def _faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Measures how grounded the answer is in the retrieved contexts.
    Uses sentence-level cosine similarity between answer sentences and context.
    """
    enc = get_encoder()
    answer_sents = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sents:
        return 0.0

    context_text = " ".join(contexts)
    ctx_vec = enc.encode([context_text], normalize_embeddings=True)[0]
    ans_vecs = enc.encode(answer_sents, normalize_embeddings=True)

    scores = [cosine_sim(v, ctx_vec) for v in ans_vecs]
    # Threshold: sentence is "faithful" if sim > 0.45
    faithful = sum(1 for s in scores if s > 0.45)
    return faithful / len(scores)


def _answer_relevancy(question: str, answer: str) -> float:
    enc = get_encoder()
    q_vec = enc.encode([question], normalize_embeddings=True)[0]
    a_vec = enc.encode([answer], normalize_embeddings=True)[0]
    return max(0.0, cosine_sim(q_vec, a_vec))


def _context_precision(question: str, contexts: list[str]) -> float:
    """Average precision@k — how well top-ranked chunks match the question."""
    enc = get_encoder()
    q_vec = enc.encode([question], normalize_embeddings=True)[0]
    ctx_vecs = enc.encode(contexts, normalize_embeddings=True)

    sims = [cosine_sim(q_vec, v) for v in ctx_vecs]
    # AP@k
    threshold = 0.4
    hits, precision_sum = 0, 0.0
    for i, s in enumerate(sims, 1):
        if s > threshold:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(sims) if sims else 0.0


def _context_recall(ground_truth: str, contexts: list[str]) -> float:
    enc = get_encoder()
    context_text = " ".join(contexts)
    gt_sents = [s.strip() for s in ground_truth.split(".") if len(s.strip()) > 10]
    if not gt_sents:
        return 0.0

    ctx_vec = enc.encode([context_text], normalize_embeddings=True)[0]
    gt_vecs = enc.encode(gt_sents, normalize_embeddings=True)

    scores = [cosine_sim(v, ctx_vec) for v in gt_vecs]
    recalled = sum(1 for s in scores if s > 0.45)
    return recalled / len(gt_sents)


def evaluate(samples: list[EvalSample]) -> EvalResult:
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for s in samples:
        faithfulness_scores.append(_faithfulness(s.answer, s.contexts))
        relevancy_scores.append(_answer_relevancy(s.question, s.answer))
        precision_scores.append(_context_precision(s.question, s.contexts))
        recall_scores.append(_context_recall(s.ground_truth, s.contexts))

    return EvalResult(
        faithfulness=statistics.mean(faithfulness_scores),
        answer_relevancy=statistics.mean(relevancy_scores),
        context_precision=statistics.mean(precision_scores),
        context_recall=statistics.mean(recall_scores),
        sample_count=len(samples),
    )


def load_eval_set(path: str) -> list[EvalSample]:
    """Load Q/A eval set from JSON.
    Format: [{"question": ..., "ground_truth": ..., "answer": ..., "contexts": [...]}]
    """
    with open(path) as f:
        data = json.load(f)
    return [EvalSample(**item) for item in data]


def run_pipeline_eval(pipeline, eval_set: list[EvalSample]) -> EvalResult:
    """Run live pipeline on eval set and score it."""
    populated = []
    for sample in eval_set:
        resp = pipeline.query(sample.question)
        populated.append(EvalSample(
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer=resp.answer,
            contexts=[c.content for c in resp.citations],
        ))
    return evaluate(populated)

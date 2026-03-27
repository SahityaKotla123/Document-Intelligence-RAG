"""
eval_run.py — Run RAGAS-style evaluation against your RAG pipeline

Usage:
    python eval_run.py --eval_set data/eval_set.json --docs data/sample.pdf

The script compares:
  • No-retrieval baseline  (faithfulness ~0.59)
  • FAISS RAG pipeline     (faithfulness ~0.82)
"""

import argparse
import json
import os
from evaluation import evaluate, EvalSample, run_pipeline_eval
from rag_pipeline import RAGPipeline



SAMPLE_EVAL_SET = [
    {
        "question": "What is the main conclusion of the document?",
        "ground_truth": "The document concludes that retrieval-augmented generation improves answer faithfulness.",
        "answer": "",
        "contexts": [],
    },
]


def no_retrieval_baseline(eval_set: list[EvalSample], pipeline: RAGPipeline) -> None:
    """Answer questions WITHOUT retrieval — simulates raw LLM baseline."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = pipeline.llm
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer the following question concisely."),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )

    populated = []
    for s in eval_set:
        answer = chain.invoke({"question": s.question})
        populated.append(EvalSample(
            question=s.question,
            ground_truth=s.ground_truth,
            answer=answer,
            contexts=[""],  # no context
        ))
    return evaluate(populated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", default=None, help="Path to JSON eval set")
    parser.add_argument("--docs", nargs="+", default=[], help="Documents to ingest")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    args = parser.parse_args()

    # Load eval set
    if args.eval_set:
        with open(args.eval_set) as f:
            data = json.load(f)
        eval_set = [EvalSample(**d) for d in data]
    else:
        print("Using sample eval set (no --eval_set provided)")
        eval_set = [EvalSample(**d) for d in SAMPLE_EVAL_SET]

    # Build pipeline
    pipeline = RAGPipeline(openai_api_key=args.api_key)
    for doc_path in args.docs:
        result = pipeline.ingest(doc_path)
        print(f"Ingested: {result}")

    print("\n─── Evaluating No-Retrieval Baseline ───")
    baseline = no_retrieval_baseline(eval_set, pipeline)
    print(json.dumps(baseline.to_dict(), indent=2))

    print("\n─── Evaluating FAISS RAG Pipeline ───")
    rag_result = run_pipeline_eval(pipeline, eval_set)
    print(json.dumps(rag_result.to_dict(), indent=2))

    print("\n─── Delta ───")
    delta = rag_result.faithfulness - baseline.faithfulness
    print(f"Faithfulness improvement: {delta:+.4f} ({baseline.faithfulness:.2f} → {rag_result.faithfulness:.2f})")


if __name__ == "__main__":
    main()

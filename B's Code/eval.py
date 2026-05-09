from reader import Reader
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


# Load datset
def load_eval_dataset(txt_path="RAG_Test_Questions_Answers.txt"):
    dataset = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            # skip header row
            if parts[0] == "#":
                continue
            try:
                int(parts[0])  # first col is row number
            except ValueError:
                continue

            query = parts[1].strip()
            ideal_answer = parts[2].strip()
            # source docs are semicolon-separated in col 4
            source_docs = [s.strip() for s in parts[3].split(";") if s.strip()]

            dataset.append({
                "query": query,
                "source_docs": source_docs,
                "ideal_answer": ideal_answer
            })
    return dataset


# Metrics
def precision_at_k(retrieved_ids, relevant_ids):
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


# Evaluation loop
def run_eval(retriever, dataset, top_k=2):
    reader = Reader()
    precisions, recalls, rrs, times, generated_answers, ideal_answers, semantic_scores = [], [], [], [], [], [], []

    for sample in dataset:
        query = sample["query"]
        relevant_ids = sample["source_docs"]
        t0 = time.perf_counter()
        docs = retriever.retrieve(query, top_k=top_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        retrieved_ids = [
            d["source"]
            for d in docs
        ]

        answer = reader.generate_answer(query, docs)
        generated_answers.append(answer)
        ideal_answers.append(sample["ideal_answer"])
        gen_emb = model.encode([answer])
        ideal_emb = model.encode([sample["ideal_answer"]])

        score = cosine_similarity(gen_emb, ideal_emb)[0][0]
        semantic_scores.append(score)

        precisions.append(precision_at_k(retrieved_ids, relevant_ids))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids))
        rrs.append(reciprocal_rank(retrieved_ids, relevant_ids))
        times.append(elapsed_ms)

    print(f"\n{'RAG RETRIEVAL EVALUATION':^40}")
    print("─" * 40)
    print(f"  Samples evaluated : {len(precisions)}")
    print(f"  Precision         : {np.mean(precisions):.3f}")
    print(f"  Recall            : {np.mean(recalls):.3f}")
    print(f"  MRR               : {np.mean(rrs):.3f}")
    print(f"  Avg latency       : {np.mean(times):.1f} ms")
    print(f"  Semantic Answer Similarity : {np.mean(semantic_scores):.3f}")
    print("─" * 40)
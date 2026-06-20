"""
eval_retrieval.py — HelpDesk Copilot
Evaluates ChromaDB retrieval using DevRev annotated_queries ground truth.
Metrics: Hit@K (K=1,3,5,10) and MRR
"""

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset

# ── PATHS ──
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── CONFIG ──
COLLECTION_NAME = "helpdesk_kb"
EMBED_MODEL     = "all-MiniLM-L6-v2"
K_VALUES        = [1, 3, 5, 10]


def load_ground_truth():
    """Load annotated queries with ground-truth retrievals."""
    print("Loading annotated_queries from DevRev...")
    ds = load_dataset("devrev/search", "annotated_queries", split="train")
    df = pd.DataFrame(ds)
    print(f"Loaded {len(df):,} annotated queries")
    return df


def extract_relevant_ids(retrievals):
    """Extract list of relevant article IDs from the retrievals column."""
    if isinstance(retrievals, list):
        return [item["id"] for item in retrievals if isinstance(item, dict) and "id" in item]
    return []


def compute_hit_at_k(retrieved_ids, relevant_ids, k):
    """1 if any relevant_id is in top-K retrieved, else 0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def compute_reciprocal_rank(retrieved_ids, relevant_ids):
    """1/rank of the first relevant result found."""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


if __name__ == "__main__":
    # Load ground truth
    gt_df = load_ground_truth()

    # Extract relevant IDs for each query
    gt_df["relevant_ids"] = gt_df["retrievals"].apply(extract_relevant_ids)

    # Drop rows with no relevant IDs
    gt_df = gt_df[gt_df["relevant_ids"].apply(len) > 0]
    print(f"Queries with valid ground truth: {len(gt_df)}")
    print(f"Avg relevant docs per query: {gt_df['relevant_ids'].apply(len).mean():.1f}")

    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    print(f"Collection '{COLLECTION_NAME}' has {collection.count():,} documents")

    # ── Check ID format overlap ──
    sample_gt_ids = gt_df["relevant_ids"].iloc[0]
    sample_db_ids = collection.peek(5)["ids"]
    print(f"\nSample ground-truth IDs: {sample_gt_ids[:3]}")
    print(f"Sample ChromaDB IDs:     {sample_db_ids[:3]}")

    # ── Run evaluation ──
    queries      = gt_df["query"].tolist()
    relevant_ids = gt_df["relevant_ids"].tolist()
    max_k        = max(K_VALUES)

    hits = {k: [] for k in K_VALUES}
    mrrs = []

    print(f"\nEvaluating {len(queries)} queries...")
    for i, (query, rel_ids) in enumerate(zip(queries, relevant_ids)):
        results = collection.query(query_texts=[query], n_results=max_k)
        retrieved = results["ids"][0]

        for k in K_VALUES:
            hits[k].append(compute_hit_at_k(retrieved, rel_ids, k))
        mrrs.append(compute_reciprocal_rank(retrieved, rel_ids))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} / {len(queries)}")

    # ── Results ──
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    for k in K_VALUES:
        score = sum(hits[k]) / len(hits[k])
        bar = "█" * int(score * 30)
        print(f"  Hit@{k:<4}  {score:.4f}  {bar}")

    mrr = sum(mrrs) / len(mrrs)
    bar = "█" * int(mrr * 30)
    print(f"  MRR      {mrr:.4f}  {bar}")
    print("=" * 50)

    print(f"\nInterpretation:")
    h1 = sum(hits[1]) / len(hits[1])
    h5 = sum(hits[5]) / len(hits[5])
    h10 = sum(hits[10]) / len(hits[10])
    print(f"  Hit@1  = {h1*100:.1f}% — correct article is the #1 result")
    print(f"  Hit@5  = {h5*100:.1f}% — correct article in top 5")
    print(f"  Hit@10 = {h10*100:.1f}% — correct article in top 10")
    print(f"  MRR    = {mrr:.4f} — avg rank of first correct result: {1/mrr:.1f}" if mrr > 0 else "  MRR    = 0.0")
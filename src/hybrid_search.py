"""
hybrid_search.py — HelpDesk Copilot
Hybrid retrieval: BM25 (keyword) + ChromaDB (vector) with score fusion.
"""

import os
import pandas as pd
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions

# ── PATHS ──
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ── CONFIG ──
COLLECTION_NAME = "helpdesk_kb"
EMBED_MODEL     = "all-MiniLM-L6-v2"
BM25_WEIGHT     = 0.4    # keyword score weight
VECTOR_WEIGHT   = 0.6    # vector score weight
TOP_K           = 10


def build_bm25_index(df):
    """Build BM25 index from KB text."""
    print("Building BM25 index...")
    # Tokenize by splitting on whitespace (simple but effective)
    tokenized = [doc.lower().split() for doc in df["text"].tolist()]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built over {len(tokenized):,} documents")
    return bm25


def bm25_search(bm25, query, doc_ids, top_k=50):
    """Return top-K doc IDs and scores from BM25."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-K indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((doc_ids[idx], float(scores[idx])))

    return results


def vector_search(collection, query, top_k=50):
    """Return top-K doc IDs and distances from ChromaDB."""
    results = collection.query(query_texts=[query], n_results=top_k)

    ids       = results["ids"][0]
    distances = results["distances"][0]

    # Convert cosine distance to similarity score (1 - distance)
    scored = [(doc_id, 1.0 - dist) for doc_id, dist in zip(ids, distances)]
    return scored


def normalize_scores(scored_results):
    """Min-max normalize scores to [0, 1]."""
    if not scored_results:
        return {}

    scores = [s for _, s in scored_results]
    min_s = min(scores)
    max_s = max(scores)
    rng = max_s - min_s if max_s > min_s else 1.0

    return {doc_id: (score - min_s) / rng for doc_id, score in scored_results}


def hybrid_search(bm25, collection, query, doc_ids,
                  bm25_weight=BM25_WEIGHT, vector_weight=VECTOR_WEIGHT,
                  top_k=TOP_K):
    """
    Combine BM25 and vector search with reciprocal rank fusion.
    Returns list of (doc_id, combined_score) sorted by score desc.
    """
    # Get results from both
    bm25_results   = bm25_search(bm25, query, doc_ids, top_k=50)
    vector_results = vector_search(collection, query, top_k=50)

    # Normalize scores to [0, 1]
    bm25_scores   = normalize_scores(bm25_results)
    vector_scores = normalize_scores(vector_results)

    # Combine all candidate doc IDs
    all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())

    # Weighted fusion
    combined = {}
    for doc_id in all_ids:
        bm25_s   = bm25_scores.get(doc_id, 0.0)
        vector_s = vector_scores.get(doc_id, 0.0)
        combined[doc_id] = bm25_weight * bm25_s + vector_weight * vector_s

    # Sort by combined score
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_k]


def load_resources():
    """Load KB, BM25 index, and ChromaDB collection."""
    # Load KB
    kb_df = pd.read_csv(os.path.join(DATA_DIR, "kb_final.csv"))
    doc_ids = kb_df["id"].tolist()

    # Build BM25
    bm25 = build_bm25_index(kb_df)

    # ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    print(f"ChromaDB collection: {collection.count():,} documents")

    return bm25, collection, doc_ids, kb_df


if __name__ == "__main__":
    bm25, collection, doc_ids, kb_df = load_resources()

    # Test query
    query = "how do I reset my password"
    print(f"\n── Hybrid Search: '{query}' ──\n")

    results = hybrid_search(bm25, collection, query, doc_ids)

    for i, (doc_id, score) in enumerate(results[:5]):
        row = kb_df[kb_df["id"] == doc_id].iloc[0]
        print(f"[{i+1}] {score:.4f}  {row['title']}")
        print(f"    {row['text'][:120]}...\n")

    # ── Run evaluation ──
    print("\n" + "=" * 50)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 50)

    from datasets import load_dataset

    ds = load_dataset("devrev/search", "annotated_queries", split="train")
    gt_df = pd.DataFrame(ds)

    def extract_ids(retrievals):
        if isinstance(retrievals, list):
            return [item["id"] for item in retrievals if isinstance(item, dict) and "id" in item]
        return []

    gt_df["relevant_ids"] = gt_df["retrievals"].apply(extract_ids)
    gt_df = gt_df[gt_df["relevant_ids"].apply(len) > 0]

    K_VALUES = [1, 3, 5, 10]
    hits = {k: [] for k in K_VALUES}
    mrrs = []

    print(f"Evaluating {len(gt_df)} queries...")
    for i, (_, row) in enumerate(gt_df.iterrows()):
        query = row["query"]
        rel_ids = set(row["relevant_ids"])

        results = hybrid_search(bm25, collection, query, doc_ids)
        retrieved = [doc_id for doc_id, _ in results]

        for k in K_VALUES:
            top_k = set(retrieved[:k])
            hits[k].append(1.0 if top_k & rel_ids else 0.0)

        for j, doc_id in enumerate(retrieved):
            if doc_id in rel_ids:
                mrrs.append(1.0 / (j + 1))
                break
        else:
            mrrs.append(0.0)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} / {len(gt_df)}")

    # Results
    print("\n" + "=" * 60)
    print("RETRIEVAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<30} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8}")
    print("-" * 60)
    print(f"{'Vector only (MiniLM)':<30} {'0.2749':>8} {'0.4880':>8} {'0.6014':>8} {'0.3636':>8}")

    h1  = sum(hits[1]) / len(hits[1])
    h5  = sum(hits[5]) / len(hits[5])
    h10 = sum(hits[10]) / len(hits[10])
    mrr = sum(mrrs) / len(mrrs)
    print(f"{'Hybrid (BM25 + MiniLM)':<30} {h1:>8.4f} {h5:>8.4f} {h10:>8.4f} {mrr:>8.4f}")
    print("=" * 60)

    if h10 > 0.6014:
        print(f"\n🎉 Hybrid beats vector-only by {(h10-0.6014)*100:+.1f} pp on Hit@10!")
    else:
        print(f"\n📊 Vector-only still leads on Hit@10")

    # Save BM25 index for agent use later
    bm25_path = os.path.join(MODEL_DIR, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)
    print(f"\n✅ BM25 index saved to {bm25_path}")
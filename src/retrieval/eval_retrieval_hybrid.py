"""
Evaluate hybrid retrieval (BM25 + Vector) using DevRev annotated_queries.
Same ground truth as eval_retrieval.py for fair comparison.
"""
import os, pandas as pd, numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "helpdesk_kb"
EMBED_MODEL     = "all-MiniLM-L6-v2"
BM25_WEIGHT     = 0.4
VECTOR_WEIGHT   = 0.6

# Load KB
kb = pd.read_csv(os.path.join(DATA_DIR, "kb_final.csv"))
doc_ids = kb["id"].tolist()
doc_texts = kb["text"].tolist()

# Build BM25
print("Building BM25 index...")
tokenized = [doc.lower().split() for doc in doc_texts]
bm25 = BM25Okapi(tokenized)
print(f"BM25 index built over {len(tokenized):,} documents")

# Connect ChromaDB
print("Connecting to ChromaDB...")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
print(f"Collection has {collection.count()} documents")

# Load ground truth from HuggingFace (same as eval_retrieval.py)
print("Loading annotated_queries from DevRev...")
ds = load_dataset("devrev/search", "annotated_queries", split="train")
df = pd.DataFrame(ds)
print(f"Loaded {len(df):,} annotated queries")

# Extract relevant IDs
def extract_relevant_ids(retrievals):
    if isinstance(retrievals, list):
        return [item["id"] for item in retrievals if isinstance(item, dict) and "id" in item]
    return []

df["relevant_ids"] = df["retrievals"].apply(extract_relevant_ids)
df = df[df["relevant_ids"].apply(len) > 0]
print(f"Queries with valid ground truth: {len(df)}")

queries = df["query"].tolist()
relevant_ids_list = df["relevant_ids"].tolist()

# Create doc_id to index mapping for BM25
id_to_idx = {did: i for i, did in enumerate(doc_ids)}

# Evaluate
hits = {1: 0, 3: 0, 5: 0, 10: 0}
rr_sum = 0
total = len(queries)

for i, (query, rel_ids) in enumerate(zip(queries, relevant_ids_list)):
    # BM25 scores
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1
    bm25_norm = bm25_scores / bm25_max

    # Vector scores
    results = collection.query(query_texts=[query], n_results=50)
    vector_ids = results["ids"][0]
    vector_dists = results["distances"][0]

    vector_score_map = {}
    if vector_dists:
        max_dist = max(vector_dists) if max(vector_dists) > 0 else 1
        for vid, vd in zip(vector_ids, vector_dists):
            vector_score_map[vid] = 1 - (vd / max_dist) if max_dist > 0 else 0

    # Fuse scores
    fused = {}
    for idx, did in enumerate(doc_ids):
        b_score = bm25_norm[idx] * BM25_WEIGHT
        v_score = vector_score_map.get(did, 0) * VECTOR_WEIGHT
        if b_score > 0 or v_score > 0:
            fused[did] = b_score + v_score

    # Rank top 10
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:10]
    ranked_ids = [r[0] for r in ranked]

    # Check hits
    rel_set = set(rel_ids)
    for k in hits:
        if rel_set & set(ranked_ids[:k]):
            hits[k] += 1

    # MRR
    for rank, rid in enumerate(ranked_ids, 1):
        if rid in rel_set:
            rr_sum += 1.0 / rank
            break

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1} / {total}")

print("\n" + "=" * 50)
print("HYBRID RETRIEVAL EVALUATION RESULTS")
print("=" * 50)
for k in [1, 3, 5, 10]:
    val = hits[k] / total
    bar = "█" * int(val * 30)
    print(f"  Hit@{k:<4} {val:.4f}  {bar}")
mrr = rr_sum / total
print(f"  MRR     {mrr:.4f}  {'█' * int(mrr * 30)}")
print("=" * 50)

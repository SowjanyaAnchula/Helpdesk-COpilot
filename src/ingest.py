"""
ingest.py — HelpDesk Copilot
Loads kb_clean.csv into ChromaDB with sentence-transformer embeddings.
Run once before starting the app.
"""

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ── PATHS ──
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── CONFIG ──
COLLECTION_NAME = "helpdesk_kb"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
BATCH_SIZE      = 500

def load_kb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} KB chunks from {path}")

    # Drop rows with empty text
    before = len(df)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    print(f"Dropped {before - len(df)} empty rows. Remaining: {len(df):,}")

    return df


def get_or_create_collection(client, name, embed_fn):
    # If collection exists, delete and recreate for fresh ingest
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        print(f"Collection '{name}' exists — deleting for fresh ingest...")
        client.delete_collection(name)

    collection = client.create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )
    print(f"Created collection '{name}'")
    return collection


def ingest(df: pd.DataFrame, collection) -> None:
    total = len(df)
    print(f"\nIngesting {total:,} chunks in batches of {BATCH_SIZE}...")

    for start in range(0, total, BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        ids       = batch["id"].astype(str).tolist()
        documents = batch["text"].tolist()
        metadatas = [{"title": str(t)} for t in batch["title"].tolist()]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        end = min(start + BATCH_SIZE, total)
        print(f"  Ingested {end:,} / {total:,}")

    print(f"\n✅ Done. {total:,} chunks indexed in ChromaDB.")


def test_query(collection) -> None:
    print("\n── Test Query ──")
    query = "how do I reset my password"
    results = collection.query(query_texts=[query], n_results=3)

    print(f"Query: '{query}'")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n[{i+1}] {meta['title']}")
        print(f"     {doc[:150]}...")


if __name__ == "__main__":
    # Load data
    kb_path = os.path.join(DATA_DIR, "kb_clean.csv")
    df = load_kb(kb_path)

    # Set up ChromaDB
    print(f"\nConnecting to ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Create collection and ingest
    collection = get_or_create_collection(client, COLLECTION_NAME, embed_fn)
    ingest(df, collection)

    # Quick sanity check
    test_query(collection)

    print(f"\nChromaDB persisted at: {CHROMA_DIR}")
    print("Ready for retrieval.")
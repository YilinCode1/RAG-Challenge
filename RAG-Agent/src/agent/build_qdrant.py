# scripts/build_qdrant.py

import json
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
import os

# ---------- 1. Load chunks JSON ----------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # /rag-agent/src/agent
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))   # /rag-agent
DATA_PATH = os.path.join(BASE_DIR, "data", "chunks_fixed.json")

with open(DATA_PATH, "r") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
ids = [c["id"] for c in chunks]

print(f"Loaded {len(chunks)} chunks")

# ---------- 2. Init Qdrant ----------
client = QdrantClient(path="qdrant_db")
collection_name = "deepseek_sparse_fixed"

# ---------- 3. Create / Reset collection ----------
if client.collection_exists(collection_name):
    print("Deleting old collection...")
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={},  # Sparse only
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)

print("Created new collection:", collection_name)

# ---------- 4. Compute BM25 sparse embeddings ----------
print("Generating BM25 sparse vectors…")
sparse_embedder = SparseTextEmbedding("Qdrant/bm25")
bm25_vectors = list(sparse_embedder.embed(texts))

# ---------- 5. Upload ----------
print("Uploading points…")
points = [
    models.PointStruct(
        id=ids[i],
        vector={"bm25": bm25_vectors[i].as_object()},
        payload={
            "chunk_id": ids[i],
            "text": texts[i]
        },
    )
    for i in range(len(texts))
]

client.upsert(collection_name=collection_name, points=points)

print("Qdrant vectorstore build COMPLETE - Ready for RAG Agent")

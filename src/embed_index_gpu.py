# src/embed_index_gpu.py
"""
Encode chunks via sentence-transformers and build FAISS index.
Save index to indices/faiss_index.faiss and mapping vectors_meta.json
Run on Colab (GPU) for speed; can run locally too.
"""
import os, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from utils import load_json, save_json

PROC = "./data/processed"
INDICES = "./indices"
os.makedirs(INDICES, exist_ok=True)
META_FILE = os.path.join(INDICES, "vectors_meta.json")
INDEX_FILE = os.path.join(INDICES, "faiss_index.faiss")

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(EMB_MODEL, device=device)

def collect_chunks():
    chunks = []
    for f in Path(PROC).glob("*.json"):
        j = load_json(str(f))
        for c in j.get("chunks", []):
            chunks.append({"doc_id": j["id"], "chunk_id": c["chunk_id"], "text": c["text"], "meta": c.get("meta", {})})
    return chunks

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=16)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    print("Embedding dim", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("Saved index to", INDEX_FILE)
    mapping = {i: {"doc_id": chunks[i]["doc_id"], "chunk_id": chunks[i]["chunk_id"], "meta": chunks[i]["meta"]} for i in range(len(chunks))}
    save_json(mapping, META_FILE)
    print("Saved mapping to", META_FILE)

if __name__ == "__main__":
    chunks = collect_chunks()
    print("Found", len(chunks), "chunks")
    build_index(chunks)

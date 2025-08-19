# src/retriever_gpu.py
import faiss, json, numpy as np, torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from utils import load_json
from fuzzywuzzy import fuzz

INDEX_FILE = "./indices/faiss_index.faiss"
MAPPING_FILE = "./indices/vectors_meta.json"
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 40
RERANK_K = 8

embedder = SentenceTransformer(EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
index = faiss.read_index(INDEX_FILE)
mapping = load_json(MAPPING_FILE)

def dense_search(query, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    ids = I[0].tolist()
    scores = D[0].tolist()
    hits = []
    for idx, score in zip(ids, scores):
        meta = mapping.get(str(idx)) or mapping.get(idx)
        if meta is None:
            continue
        # attempt to fetch text from processed JSONs (expensive). For speed, store a small text excerpt in mapping.meta earlier.
        hits.append({"row": idx, "score": float(score), "doc_id": meta["doc_id"], "chunk_id": meta["chunk_id"], "meta": meta.get("meta", {})})
    return hits

def lexical_score(query, text):
    if not text:
        return 0.0
    return fuzz.partial_ratio(query.lower(), text.lower())/100.0

def hybrid_rerank(query, candidates, cross_encoder=None, topn=RERANK_K, weight_sem=0.6, weight_lex=0.4):
    # If cross_encoder provided (a CrossEncoder object), use it; otherwise combine sem+lex
    if cross_encoder is not None:
        pairs = [(query, c.get("meta", {}).get("text", "")) for c in candidates]
        scores = cross_encoder.predict(pairs, batch_size=16)
        for c,s in zip(candidates, scores):
            c["cross_score"] = float(s)
            c["combined"] = 0.5*c.get("score",0) + 0.5*c["cross_score"]
    else:
        for c in candidates:
            sem = c.get("score", 0)
            text = c.get("meta", {}).get("text", "") or ""
            lex = lexical_score(query, text)
            c["lex"] = lex
            c["combined"] = weight_sem*sem + weight_lex*lex
    ranked = sorted(candidates, key=lambda x: x["combined"], reverse=True)
    return ranked[:topn]

def retrieve(query, cross_encoder=None):
    den = dense_search(query)
    # For speed, ensure mapping.meta contains small 'text' field (modify embed_index_gpu to store snippet)
    reranked = hybrid_rerank(query, den, cross_encoder=cross_encoder)
    return reranked

if __name__ == "__main__":
    print(retrieve("What was Apple's net income in FY2023?"))

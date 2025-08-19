# src/preprocess.py
"""
Chunk documents into chunks and convert tables to markdown chunks.
Writes back processed JSON files with "chunks" list.
"""
import os
from pathlib import Path
from utils import save_json, load_json
import pandas as pd

PROC = "./data/processed"
CHUNK_SIZE = 800  # chars approx
OVERLAP = 200

def chunk_text(text):
    n = len(text)
    start = 0
    i = 0
    chunks = []
    while start < n:
        end = min(n, start + CHUNK_SIZE)
        chunk = text[start:end]
        chunks.append({"chunk_id": f"txt_{i}", "start": start, "end": end, "text": chunk, "meta": {}})
        i += 1
        start = end - OVERLAP
    return chunks

def table_to_md(csv_text):
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(csv_text))
        md = df.head(12).to_markdown(index=False)
        return md
    except Exception:
        return f"```\n{csv_text[:2000]}\n```"

def run():
    for f in Path(PROC).glob("*.json"):
        doc = load_json(str(f))
        chunks = []
        if doc.get("text"):
            for c in chunk_text(doc["text"]):
                c["doc_id"] = doc["id"]
                c["chunk_id"] = f"{doc['id']}_{c['chunk_id']}"
                c["meta"]["is_table"] = False
                chunks.append(c)
        for t in doc.get("tables", []):
            md = table_to_md(t["csv"])
            chunks.append({"doc_id": doc["id"], "chunk_id": t["table_id"], "start": None, "end": None, "text": md, "meta": {"is_table": True, "page": t.get("page"), "csv": t["csv"]}})
        doc["chunks"] = chunks
        save_json(doc, str(f))
        print("Processed chunks for", doc["id"])

if __name__ == "__main__":
    run()

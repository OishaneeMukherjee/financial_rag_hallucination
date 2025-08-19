# src/ingest.py
"""
Ingest PDFs, HTML, CSV to data/processed/*.json with fields:
{id, title, text, tables: [{table_id, page, csv}], source}
"""
import os
from pathlib import Path
import pdfplumber
import camelot
import pandas as pd
from utils import save_json

RAW = "./data/raw"
PROC = "./data/processed"
os.makedirs(PROC, exist_ok=True)

def process_pdf(path):
    docid = Path(path).stem
    out = {"id": docid, "title": docid, "text": "", "tables": [], "source": path}
    with pdfplumber.open(path) as pdf:
        pages_text = []
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            pages_text.append(txt)
            # try Camelot for table extraction (works with good PDFs)
            try:
                tables = camelot.read_pdf(path, pages=str(i+1))
                for t_idx, t in enumerate(tables):
                    csv = t.df.to_csv(index=False)
                    out["tables"].append({"table_id": f"{docid}_p{i+1}_t{t_idx}", "page": i+1, "csv": csv})
            except Exception:
                # fallback: pdfplumber simple table extraction
                try:
                    p_tables = page.extract_tables()
                    for t_idx, t in enumerate(p_tables):
                        df = pd.DataFrame(t)
                        out["tables"].append({"table_id": f"{docid}_p{i+1}_pt{t_idx}", "page": i+1, "csv": df.to_csv(index=False)})
                except Exception:
                    pass
    out["text"] = "\n\n".join(pages_text)
    return out

def run():
    p = Path(RAW)
    for f in p.glob("*"):
        if f.suffix.lower() in [".pdf", ".html", ".htm", ".txt", ".csv"]:
            print("Processing", f)
            try:
                if f.suffix.lower() == ".csv":
                    # wrap CSV into a 'doc' with no text and one table
                    import pandas as pd
                    df = pd.read_csv(f)
                    docid = f.stem
                    out = {"id": docid, "title": docid, "text": "", "tables": [{"table_id": f"{docid}_csv_0", "page": 0, "csv": df.to_csv(index=False)}], "source": str(f)}
                else:
                    out = process_pdf(str(f))
                save_json(out, os.path.join(PROC, f"{out['id']}.json"))
            except Exception as e:
                print("Failed ingest", f, e)

if __name__ == "__main__":
    run()

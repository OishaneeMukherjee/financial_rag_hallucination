# src/utils.py
import re, json, os
from dateutil.parser import parse as dateparse
from typing import Optional

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r'[\$,\(\)]', '', t)
    if '%' in t:
        try:
            return float(t.replace('%',''))/100.0
        except:
            return None
    m = re.search(r"-?\d+(\.\d+)?", t.replace(',', ''))
    if m:
        try:
            return float(m.group(0))
        except:
            return None
    return None

def extract_dates(text: str):
    try:
        m = re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|\d{1,2})[^\n,]{0,40}\d{2,4}", text, flags=re.I)
        if m:
            d = dateparse(m.group(0), fuzzy=True)
            return d.isoformat()
    except Exception:
        pass
    return None

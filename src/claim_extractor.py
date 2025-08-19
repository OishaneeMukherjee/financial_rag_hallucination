# src/claim_extractor.py
import re
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)

CITATION_RE = re.compile(r"\[(SQL-\d+|\d+)\]")

def extract_claims(answer_text):
    sents = sent_tokenize(answer_text)
    claims = []
    for s in sents:
        cids = CITATION_RE.findall(s)
        claims.append({"text": s.strip(), "citations": cids})
    return claims

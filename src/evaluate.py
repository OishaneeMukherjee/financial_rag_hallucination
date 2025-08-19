# src/evaluate.py
"""
Skeleton evaluation: run pipeline on sample QA items and compute basic metrics.
Items format: [{"id":..,"question":..,"gold_answers":[..], "gold_claim_labels":[...]}]
"""
import json
from retriever_gpu import retrieve
from generator_gpu import generate_answer
from claim_extractor import extract_claims
from mes import minimal_evidence_set
from verifier_api import verifier_predict

def run_item(item, cross_encoder=None):
    q = item["question"]
    candidates = retrieve(q, cross_encoder=cross_encoder)
    evidence = candidates[:4]
    # if you have SQL probe implement and add here
    sql_hits = []
    ans = generate_answer(q, evidence, sql_hits)
    claims = extract_claims(ans)
    verifications = []
    for c in claims:
        # compute MES
        S, p = minimal_evidence_set(c["text"], evidence, verifier_predict, threshold=0.85)
        verifications.append({"claim": c["text"], "mes": [s["chunk_id"] for s in S], "p_entail": p})
    return {"question": q, "answer": ans, "verifications": verifications}

if __name__ == "__main__":
    items = [{"id":"1", "question":"What was Apple net income in FY2023?"}]
    for it in items:
        print(json.dumps(run_item(it), indent=2))

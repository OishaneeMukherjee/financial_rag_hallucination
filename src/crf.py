# src/crf.py
"""
Compute fragility by perturbing retrieval results:
- swap a top evidence with a near neighbor,
- re-run generator (or eval proxy) and see if claim flips.
For speed: compute reranker score differences or verifier prob differences.
"""
from copy import deepcopy
from retriever_gpu import dense_search, hybrid_rerank

def fragility_score(query, top_candidates, retriever_func, verifier_predict, n_perturb=5):
    """
    For each perturbation, replace one top candidate with a near neighbor and
    check effect on verifier(claim, evidence). Return fraction of perturbations that cause >delta change.
    """
    base_evidence = [c for c in top_candidates]
    base_prob = verifier_predict(query["claim"], [c.get("meta",{}).get("text","") for c in base_evidence])
    flips = 0
    for i in range(n_perturb):
        # pick an index to replace
        import random
        replace_idx = random.randrange(len(base_evidence))
        # get a dense neighbor by querying with the chosen chunk text
        neighbor_pool = dense_search(base_evidence[replace_idx].get("meta", {}).get("text",""), k=10)
        # pick a random neighbor not equal to original
        for neigh in neighbor_pool:
            if neigh["chunk_id"] != base_evidence[replace_idx]["chunk_id"]:
                break
        pert = deepcopy(base_evidence)
        pert[replace_idx] = neigh
        p = verifier_predict(query["claim"], [c.get("meta",{}).get("text","") for c in pert])
        if abs(p - base_prob) > 0.2:  # threshold for 'flip'
            flips += 1
    return flips / n_perturb

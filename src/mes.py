# src/mes.py
from copy import deepcopy

def minimal_evidence_set(claim, candidates, verifier_predict, threshold=0.85):
    """
    claim: string
    candidates: list of {"id","text",...} (ranked by relevance)
    verifier_predict: function(claim, list_of_texts) -> float (prob of entailment)
    """
    S = deepcopy(candidates)
    if not S:
        return [], 0.0
    best_prob = verifier_predict(claim, [c["text"] for c in S])
    if best_prob < threshold:
        return [], best_prob
    changed = True
    while changed:
        changed = False
        # try removing least-important (end) candidates first
        for i in range(len(S)-1, -1, -1):
            trial = S[:i] + S[i+1:]
            if not trial:
                continue
            p = verifier_predict(claim, [c["text"] for c in trial])
            if p >= threshold:
                S = trial
                changed = True
                break
    return S, verifier_predict(claim, [c["text"] for c in S])

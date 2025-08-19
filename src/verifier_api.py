# src/verifier_api.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tokenizer = AutoTokenizer.from_pretrained("./models/verifier")
model = AutoModelForSequenceClassification.from_pretrained("./models/verifier").to("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}  # depends on your label mapping

def verifier_predict(claim, evidence_texts):
    joint = " ||| ".join(evidence_texts)
    inputs = tokenizer(claim, joint, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # assume index 0 is entailment; adapt if different
        entail_prob = float(probs[0])
    return entail_prob

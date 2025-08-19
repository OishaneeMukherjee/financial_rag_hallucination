# src/generator_gpu.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

GEN_MODEL = "google/flan-t5-large"   # choose a model that fits Colab GPU (8bit recommended)
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_generator(model_path=None, use_8bit=True):
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto", load_in_8bit=use_8bit)
    else:
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL, device_map="auto", load_in_8bit=use_8bit)
    return tokenizer, model

def build_prompt(query, evidence_snips, sql_hits):
    evid = []
    for i,s in enumerate(evidence_snips):
        text = s.get("meta", {}).get("text", "") or ""
        evid.append(f"[{i}] {s.get('doc_id')} | {text[:400].replace('\\n',' ')}")
    for i,s in enumerate(sql_hits):
        evid.append(f"[SQL-{i}] {s['table_id']} | {s['col']}={s['raw']}")
    ev_block = "\n\n".join(evid) if evid else "(no evidence found)"
    prompt = (
        "You are a careful financial assistant. Use ONLY the evidence below. "
        "Cite inline with [i]/[SQL-j]. If conflicting or missing, say 'INSUFFICIENT EVIDENCE'.\n\n"
        f"EVIDENCE:\n{ev_block}\n\nQuestion: {query}\n\nAnswer:"
    )
    return prompt

def generate_answer(query, evidence_snips, sql_hits, model_path=None, max_new_tokens=180):
    tokenizer, model = load_generator(model_path=model_path)
    prompt = build_prompt(query, evidence_snips, sql_hits)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

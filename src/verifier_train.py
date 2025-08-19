# src/verifier_train.py
"""
Fine-tune a verifier (NLI) model to classify (claim, evidence) -> entailment/neutral/contradiction.
Provide training JSONL with {"claim":..,"evidence":..,"label":0/1/2}
"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

BASE = "roberta-large-mnli"
SAVE_DIR = "./models/verifier"

def main(train_jsonl):
    ds = load_dataset("json", data_files=train_jsonl)["train"]
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3)

    def preprocess(ex):
        t = tokenizer(ex["claim"], ex["evidence"], truncation=True, padding="max_length", max_length=256)
        t["labels"] = ex["label"]
        return t

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(output_dir=SAVE_DIR, per_device_train_batch_size=8, num_train_epochs=3, fp16=torch.cuda.is_available(), logging_steps=25, learning_rate=2e-5, save_strategy="epoch", report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Saved verifier to", SAVE_DIR)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    args = parser.parse_args()
    main(args.train_jsonl)
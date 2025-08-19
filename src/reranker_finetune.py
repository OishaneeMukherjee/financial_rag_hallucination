# src/reranker_finetune.py
"""
LoRA fine-tune a cross-encoder reranker on training pairs (query, passage) -> label.
Run on Colab (GPU) to fine-tune quickly with PEFT.
You MUST supply a training file (JSONL) with {"query":..,"passage":..,"label":0/1}.
"""
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SAVE_DIR = "./models/reranker"

def load_pairs(path):
    ds = load_dataset("json", data_files=path)["train"]
    return ds

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True, help="path to jsonl with fields query, passage, label")
    args = parser.parse_args()

    ds = load_pairs(args.train_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE)
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    def preprocess(ex):
        t = tokenizer(ex["query"], ex["passage"], truncation=True, padding="max_length", max_length=256)
        t["labels"] = ex["label"]
        return t

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args_train = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        learning_rate=2e-4,
        save_strategy="epoch",
        logging_steps=50,
        report_to=[]
    )
    trainer = Trainer(model=model, args=args_train, train_dataset=ds)
    trainer.train()
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Saved reranker to", SAVE_DIR)

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# PATHS (ABSOLUTE, SAFE)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = os.path.join(BASE_DIR, "intent_data.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "intent_model")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Dataset
# =========================
class IntentDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 🔧 NEW: Format with previous context
        prev = item.get("previous", "")
        curr = item["current"]
        
        # Combine previous and current with special tokens
        text = f"[PREV] {prev} [SEP] [CURR] {curr}"
        
        enc = self.tokenizer(
            text,  # ✅ NEW FORMAT
            truncation=True,
            padding="max_length",
            max_length=128,  # Increased from 64
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(item["label"]), dtype=torch.long),
        }


# =========================
# Train
# =========================
def main():
    print("📂 DATA PATH:", DATA_PATH)
    print("💾 FINAL MODEL PATH:", FINAL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    ).to(DEVICE)

    dataset = IntentDataset(DATA_PATH, tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,
        num_train_epochs=6,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    # =========================
    # SAVE FINAL MODEL (ONLY HERE)
    # =========================
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    print("✅ FINAL INTENT MODEL SAVED TO:", FINAL_DIR)


if __name__ == "__main__":
    main()

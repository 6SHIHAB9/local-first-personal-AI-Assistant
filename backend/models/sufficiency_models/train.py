# backend/models/sufficiency_models/train.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import SufficiencyModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "data/sufficiency_train.jsonl"
SAVE_PATH = "model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on device: {DEVICE}")

if DEVICE.type == "cpu":
    print("⚠️ WARNING: CUDA not available, training on CPU")

# =========================
# DATASET
# =========================
class SufficiencyDataset(Dataset):
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
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.float)
        }

# =========================
# TRAIN LOOP
# =========================
def train():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True
    )

    dataset = SufficiencyDataset(DATA_PATH, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=16,          # 🔥 GPU-friendly
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = SufficiencyModel(BASE_MODEL).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.SmoothL1Loss()

    model.train()

    for epoch in range(4):
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | avg loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    train()

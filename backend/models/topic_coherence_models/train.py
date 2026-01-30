# backend/models/topic_coherence_models/train.py

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import TopicCoherenceModel

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "data/topic_coherence.jsonl"
OUTPUT_PATH = "model.pt"
EPOCHS = 4
BATCH_SIZE = 16
LR = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using device: {device}")

class TopicDataset(Dataset):
    def __init__(self, path, tokenizer):
        print(f"📂 Loading dataset from {path}")
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append((obj["text"], float(obj["label"])))
        print(f"✅ Loaded {len(self.samples)} samples")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True
    )

    dataset = TopicDataset(DATA_PATH, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TopicCoherenceModel(BASE_MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"📉 Epoch {epoch + 1}/{EPOCHS} | avg loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"💾 Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

import os
import json
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "reference_ranker"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Model definition (MUST match training)
# -------------------------
class Ranker(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.scorer = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.scorer(cls).squeeze(-1)

# -------------------------
# Load trained model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base = AutoModel.from_pretrained(MODEL_NAME)
model = Ranker(base).to(DEVICE)

model.load_state_dict(torch.load("reference_ranker.pt", map_location=DEVICE))
model.eval()

# -------------------------
# Save model + tokenizer
# -------------------------
torch.save(model.state_dict(), f"{OUTPUT_DIR}/model.pt")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")

# -------------------------
# Save lightweight config
# -------------------------
config = {
    "model_type": "reference_ranker",
    "base_model": MODEL_NAME,
    "description": "Pairwise-trained reference scorer with scalar head",
    "input": ["query", "context"],
    "output": "relevance_score"
}

with open(f"{OUTPUT_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ Reference ranker exported to ./reference_ranker")


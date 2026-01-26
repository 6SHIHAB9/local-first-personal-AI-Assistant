import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class IntentClassifier:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(DEVICE)
        self.model.eval()

        # Label mapping (from training)
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()

        return self.id2label[pred_id]

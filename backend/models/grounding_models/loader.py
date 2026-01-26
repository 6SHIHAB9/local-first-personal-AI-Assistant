import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

class GroundingScorer:
    def __init__(self, model_dir: str, threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # Convert to absolute path and use forward slashes
        model_path = str(Path(model_dir).resolve()).replace('\\', '/')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()

    def score(self, question: str, sentence: str) -> float:
        inputs = self.tokenizer(
            question,
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        # label=1 is "allowed"
        return probs[0, 1].item()

    def filter_sentences(self, question, sentences, top_k=5):
        scored = []
        for s in sentences:
            scored.append((self.score(question, s), s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]
# backend/models/topic_coherence_models/scorer.py

import torch
from transformers import AutoTokenizer
from .model import TopicCoherenceModel



class TopicCoherenceScorer:
    def __init__(self, model_path: str, base_model: str, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=True
        )

        self.model = TopicCoherenceModel(base_model)
        self.model.load_state_dict(
            torch.load(f"{model_path}/model.pt", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def _format(self, question: str, sentence: str) -> str:
        return f"Question: {question}\nSentence: {sentence}"

    @torch.inference_mode()
    def score(self, question: str, sentence: str) -> float:
        text = self._format(question, sentence)
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return float(self.model(**enc).item())

    def filter(
        self,
        question: str,
        sentences: list[str],
        threshold: float = 0.5
    ) -> list[str]:
        kept = []
        for s in sentences:
            score = self.score(question, s)
            if score >= threshold:
                kept.append(s)
        return kept

# backend/models/topic_coherence_models/loader.py

from .scorer import TopicCoherenceScorer

def load_topic_coherence_scorer(
    model_path: str = "models/topic_coherence_models",
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    return TopicCoherenceScorer(
        model_path=model_path,
        base_model=base_model
    )

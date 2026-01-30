# backend/models/topic_coherence_models/test.py

from scorer import TopicCoherenceScorer

scorer = TopicCoherenceScorer(
    model_path=".",
    base_model="sentence-transformers/all-MiniLM-L6-v2"
)

tests = [
    # POSITIVES
    (
        "What is database indexing?",
        "Indexes allow the database engine to locate rows without scanning the entire table."
    ),
    (
        "What is a quorum?",
        "A quorum is the minimum number of nodes required to agree on an operation."
    ),

    # HARD NEGATIVES
    (
        "What is database indexing?",
        "Connection pooling reuses database connections to reduce overhead."
    ),
    (
        "What is a quorum?",
        "Leader election selects a coordinator node."
    ),

    # EASY NEGATIVE
    (
        "What is database indexing?",
        "Photosynthesis converts sunlight into chemical energy."
    ),
]

for q, s in tests:
    score = scorer.score(q, s)
    print(f"\nQ: {q}\nS: {s}\n→ score = {score:.4f}")

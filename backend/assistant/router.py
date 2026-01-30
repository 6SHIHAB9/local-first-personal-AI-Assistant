from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import ollama
import re
import time
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from models.reference_models.reference_ranker.loader import ReferenceRanker

from vault.ingest import scan_vault, retrieve_relevant_chunks
from config import VAULT_PATH
from context_manager import context_manager

from models.grounding_models.loader import GroundingScorer

from models.sufficiency_models.scorer import SufficiencyScorer

from models.topic_coherence_models.loader import load_topic_coherence_scorer

# =========================
# Global vault state
# =========================
current_vault_data = None
last_vault_mtime = None

router = APIRouter()

# =========================
# Model 1: Intent classifier
# =========================
INTENT_LABEL_MAP = {
    "LABEL_0": "factual",
    "LABEL_1": "continuation",
    "LABEL_2": "casual"
}

intent_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

intent_model_path = os.path.abspath("models/intent_models/intent_model/final")
intent_model = AutoModelForSequenceClassification.from_pretrained(
    intent_model_path,
    local_files_only=True
).to(intent_device)

intent_tokenizer = AutoTokenizer.from_pretrained(
    intent_model_path,
    local_files_only=True
)

intent_model.eval()

# =========================
# Model 2: Reference Ranker
# =========================
reference_ranker = ReferenceRanker(
    "models/reference_models/reference_ranker"
)

# =========================
# Model 3: Grounding Scorer
# =========================
grounding_scorer = GroundingScorer(
    "models/grounding_models/grounding_model"
)

# =========================
# Model 4: Sufficiency Scorer
# =========================
sufficiency_scorer = SufficiencyScorer(
    model_path="models/sufficiency_models",
    base_model="sentence-transformers/all-MiniLM-L6-v2"
)

SUFFICIENCY_THRESHOLD = 0.70

# =========================
# Model 5: Topic Coherence Scorer
# =========================
topic_scorer = load_topic_coherence_scorer()


# =========================
# Models
# =========================
class AskRequest(BaseModel):
    question: str


# =========================
# Helpers
# =========================
def normalize_chunks(results) -> list[str]:
    chunks = []
    for r in results:
        if isinstance(r, dict) and "chunk" in r:
            chunks.append(r["chunk"])
        elif isinstance(r, str):
            chunks.append(r)
    return chunks

from sentence_transformers import SentenceTransformer, util

topic_embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)


def select_dominant_topic(sentences: list[str], similarity_threshold=0.6) -> list[str]:
    """
    Clusters sentences by semantic similarity and keeps the largest cluster.
    """
    if len(sentences) <= 2:
        return sentences

    embeddings = topic_embedder.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)

    clusters = []
    visited = set()

    for i in range(len(sentences)):
        if i in visited:
            continue

        cluster = [i]
        visited.add(i)

        for j in range(len(sentences)):
            if j not in visited and sim_matrix[i][j] >= similarity_threshold:
                cluster.append(j)
                visited.add(j)

        clusters.append(cluster)

    # pick largest cluster
    dominant = max(clusters, key=len)
    return [sentences[i] for i in dominant]


def score_evidence_roles(question: str, sentences: list[str]):
    """
    Scores sentences by how explanatory they are *for this question*.
    """
    q_embed = topic_embedder.encode(question, convert_to_tensor=True)
    s_embeds = topic_embedder.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(q_embed, s_embeds)[0]

    weighted = []
    for score, sent in zip(scores, sentences):
        weighted.append((float(score), sent))

    # highest explanatory relevance first
    weighted.sort(key=lambda x: x[0], reverse=True)
    return weighted

def filter_by_topic_coherence(
    question: str,
    sentences: list[str],
    top_k: int = 8
) -> list[str]:
    if not sentences:
        return []

    scored = []
    for s in sentences:
        score = topic_scorer.score(question, s)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [s for _, s in scored[:top_k]]


def rerank_chunks(question: str, chunks: list[str], top_k: int = 5) -> list[str]:
    if not chunks:
        return []

    scored = []
    for chunk in chunks:
        score = reference_ranker.score(question, chunk)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def split_into_sentences(chunks: list[str]) -> list[str]:
    sentences = []
    for chunk in chunks:
        parts = re.split(r'(?<=[.!?])\s+', chunk)
        sentences.extend([p.strip() for p in parts if len(p.strip()) >= 10])
    return sentences


# =========================
# Intent Classification
# =========================
def classify_intent(question: str) -> str:
    inputs = intent_tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(intent_device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = intent_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    return INTENT_LABEL_MAP.get(f"LABEL_{pred_id}", "factual")


# =========================
# ML BASED RETRIEVAL
# =========================
def retrieve_for_question(question: str, intent: str, vault_data: dict) -> list[str]:
    results = retrieve_relevant_chunks(question, vault_data, limit=10)
    chunks = normalize_chunks(results)

    # 🔹 ONLY for continuation
    if intent == "continuation":
        chunks = rerank_chunks(question, chunks, top_k=5)

    return chunks[:5]


# =========================
# ML-BASED GROUNDING
# =========================
def ml_ground_sentences(
    question: str,
    sentences: list[str],
    intent: str,
    top_k: int = 8,
    min_relevance: float = 0.25,  # NEW: relevance threshold
    min_grounding: float = 0.35
) -> list[str]:

    if not sentences:
        return []

    print("🧪 SENTENCES BEFORE GROUNDING:")
    for s in sentences:
        print("  >", s)

    # Calculate relevance scores for all sentences at once
    q_embed = topic_embedder.encode(question, convert_to_tensor=True)
    s_embeds = topic_embedder.encode(sentences, convert_to_tensor=True)
    relevance_scores = util.cos_sim(q_embed, s_embeds)[0]

    scored = []

    for sentence, rel_score in zip(sentences, relevance_scores):
        rel_score = float(rel_score)
        
        # FILTER 1: Relevance check
        if rel_score < min_relevance:
            print(f"  ❌ NOT RELEVANT ({rel_score:.3f}): {sentence[:60]}...")
            continue
        
        # FILTER 2: Grounding check
        ground_score = grounding_scorer.score(question, sentence)
        
        if ground_score < min_grounding:
            print(f"  ❌ NOT GROUNDED ({ground_score:.3f}): {sentence[:60]}...")
            continue
        
        # Combine scores: relevance × grounding
        combined_score = rel_score * ground_score
        scored.append((combined_score, sentence))
        print(f"  ✅ KEPT (rel={rel_score:.3f}, ground={ground_score:.3f}, combined={combined_score:.3f})")

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return [sentence for _, sentence in scored[:top_k]]



# =========================
# Vault change detection
# =========================
def get_latest_vault_mtime():
    if not VAULT_PATH.exists():
        return None
    return max(
        (p.stat().st_mtime for p in VAULT_PATH.rglob("*") if p.is_file()),
        default=None,
    )


def vault_has_changed():
    """Check if vault has changed since last sync"""
    global last_vault_mtime
    latest = get_latest_vault_mtime()
    
    if last_vault_mtime is None:
        return True
    
    if latest is None:
        return False
    
    return latest > last_vault_mtime


# =========================
# Sync Vault (Internal)
# =========================
def _internal_sync():
    """Internal sync function that returns sync info"""
    global current_vault_data, last_vault_mtime

    print("🔄 SYNCING VAULT...")
    
    current_vault_data = scan_vault()
    last_vault_mtime = get_latest_vault_mtime()
    indexed_at = time.time()

    sync_info = {
        "vault_path": str(current_vault_data["vault_path"]),
        "file_count": current_vault_data["file_count"],
        "empty_files": current_vault_data["empty_files"],
        "indexed_files": current_vault_data["indexed_files"],
        "last_indexed": indexed_at
    }
    
    print(f"✅ VAULT SYNCED: {sync_info['indexed_files']} files indexed")
    
    return sync_info


# =========================
# Sync Vault (API Endpoint)
# =========================
@router.post("/sync")
def sync_vault():
    """Manual sync endpoint"""
    return _internal_sync()


# =========================
# Ask (MAIN)
# =========================
@router.post("/ask")
def ask(req: AskRequest):
    global current_vault_data

    try:
        # 0. Sync and track if it happened
        sync_info = None
        if current_vault_data is None or vault_has_changed():
            sync_info = _internal_sync()

        question = req.question.strip()
        
        print(f"\n📝 QUESTION: {question}")
        
        # =========================
        # 1. Intent Classification
        # =========================
        intent = classify_intent(question)
        print(f"🎯 INTENT: {intent}")

        previous_q = context_manager.get_previous_question()
        use_previous_context = False

        # =========================
        # Topic + Explanation Continuity
        # =========================
        if intent == "continuation" and previous_q:
            topic_score = topic_scorer.score(previous_q, question)
            explanation_score = topic_scorer.score(
                previous_q,
                previous_q + " " + question
            )

            print(f"🧠 TOPIC SCORE: {topic_score:.4f}")
            print(f"🧠 EXPLANATION SCORE: {explanation_score:.4f}")

            if topic_score >= 0.45 or explanation_score >= 0.55:
                use_previous_context = True
            else:
                print("🔁 Topic drift detected → re-anchoring topic")
                use_previous_context = False

                # establish NEW topic anchor
                context_manager.clear_session()
                context_manager.set_topic_anchor(question)
                intent = "factual"

        # Continuation without history is invalid
        if intent == "continuation" and not previous_q:
            print("🚫 Continuation without history → refusing")
            return {"answer": "I don't have that information in my vault yet."}

        # Clear history only for fresh factual questions
        if intent == "factual":
            previous_q = context_manager.get_previous_question()
            if previous_q:
                topic_score = topic_scorer.score(previous_q, question)
                if topic_score < 0.35:
                    context_manager.clear_session()

        # 2. Casual Chat
        if intent == "casual":
            res = ollama.generate(
                model="qwen2.5:7b",
                prompt=f"""You are a friendly conversational assistant.
Keep it casual and short.

User:
{question}

Response:
""",
                options={"temperature": 0.7, "num_predict": 80},
            )
            response_data = {"answer": res["response"].strip()}
            if sync_info:
                response_data["sync_performed"] = sync_info
            return response_data

        # 3. RETRIEVAL
        effective_question = question

        if intent == "continuation" and use_previous_context:
            effective_question = previous_q + " " + question

        chunks = retrieve_for_question(effective_question, intent, current_vault_data)

        print(f"📦 CHUNKS RETRIEVED: {len(chunks)}")
        
        if not chunks:
            response_data = {"answer": "I don't have that information in my vault yet."}
            if sync_info:
                response_data["sync_performed"] = sync_info
            return response_data

        # 4. SENTENCE-LEVEL PIPELINE

        ground_min_score = 0.35 if intent == "factual" else 0.25

        # 4a. Split chunks into sentences
        sentences = split_into_sentences(chunks)
        print(f"🧪 SENTENCES BEFORE TOPIC FILTER: {len(sentences)}")

        # STEP 1: Topic anchor
        topic_anchor = (
            context_manager.get_topic_anchor()
            if intent == "continuation"
            else question
        )

        topic_k = 12 if intent == "continuation" else 10

        # 4b. Topic coherence filtering
        sentences = filter_by_topic_coherence(
            question=topic_anchor,
            sentences=sentences,
            top_k=topic_k
        )

        print(f"🎯 SENTENCES AFTER TOPIC FILTER: {len(sentences)}")

        # 4c. ML-based grounding
        allowed = ml_ground_sentences(
            question=question,
            sentences=sentences,
            intent=intent,
            top_k=8,
            min_relevance=0.25,  # tune this if needed
            min_grounding=0.35 if intent == "factual" else 0.25
        )

        print(f"✅ SENTENCES GROUNDED: {len(allowed)}")

        # HARD REFUSAL - no grounded sentences
        if not allowed:
            print("❌ NO GROUNDED SENTENCES - REFUSING")
            response_data = {"answer": "I don't have that information in my vault yet."}
            if sync_info:
                response_data["sync_performed"] = sync_info
            return response_data

        # =========================
        # SUFFICIENCY CHECK (only for continuation)
        # =========================

        suff_score = None

        if intent == "continuation":
            suff_score = sufficiency_scorer.score(
                question=question,
                sentences=allowed,
                intent=intent
            )

            print(f"🧪 SUFFICIENCY SCORE: {suff_score:.4f} (threshold: {SUFFICIENCY_THRESHOLD})")

            if suff_score < SUFFICIENCY_THRESHOLD:
                print("🚫 INSUFFICIENT EVIDENCE — REFUSING")
                response_data = {
                    "answer": "I don't have enough information in my vault to answer that confidently.",
                    "metadata": {
                        "intent": intent,
                        "sentences_grounded": len(allowed),
                        "sufficiency_score": suff_score
                    }
                }
                if sync_info:
                    response_data["sync_performed"] = sync_info
                return response_data

        # =========================
        # 🔥 TOPIC CLUSTERING DISABLED
        # =========================
        # Topic clustering was deleting correct answers that were in the minority cluster
        # Example: "What does jitha teacher do?" - the correct sentence was deleted 
        # because it didn't match the dominant "distributed systems" cluster
        
        print(f"🧠 SKIPPED TOPIC CLUSTERING (keeps all {len(allowed)} sentences)")

        print("📄 FINAL SENTENCES:")
        for i, s in enumerate(allowed, 1):
            print(f"  {i}. {s}")

        # =========================
        # Evidence weighting
        # =========================
        ranked = score_evidence_roles(question, allowed)

        # Keep top 6 most relevant
        ranked = ranked[:6]

        print("🧠 EVIDENCE WEIGHTS:")
        for score, s in ranked:
            print(f"  {score:.3f} → {s}")

        allowed_text = "\n".join(f"- {s}" for _, s in ranked)

        # 5. ANSWER GENERATION
        # Build context for continuation
        context_instruction = ""
        if intent == "continuation":
            previous_q = context_manager.get_previous_question()
            if previous_q:
                prev_lower = previous_q.lower()
                if prev_lower.startswith(("why", "what happens", "why is")):
                    context_instruction = (
                        "CONTEXT: This is a WHY follow-up.\n"
                        "Explain CONSEQUENCES, IMPACTS, or RISKS.\n"
                        "Do NOT restate the original fact.\n"
                    )
                elif prev_lower.startswith("how"):
                    context_instruction = (
                        "CONTEXT: This is a HOW follow-up.\n"
                        "Explain the MECHANISM or PROCESS.\n"
                    )

        response = ollama.generate(
            model="qwen2.5:7b",
            prompt=f"""You are answering a question using ONLY the provided sentences.

RULES:
- Use ONLY the allowed sentences below
- You MAY rephrase and COMBINE them
- If the question asks "why", "why is that an issue", or "what happens if":
  → EXPLAIN CONSEQUENCES or IMPACTS implied by the sentences
  → Do NOT simply restate the sentences
- Do NOT add external facts
- Keep the answer concise and explanatory

{context_instruction}

ALLOWED SENTENCES:
{allowed_text}

QUESTION:
{question}

ANSWER:""",
            options={"temperature": 0.0, "top_p": 0.1, "num_predict": 150},
        )

        answer = response["response"].strip()
        print(f"💬 ANSWER: {answer}")

        # 6. Store Q&A in conversation history
        if answer != "I don't have that information in my vault yet.":
            context_manager.add_turn(question, answer)

        # 7. Build response
        response_data = {
            "answer": answer,
            "metadata": {
                "chunks_retrieved": len(chunks),
                "sentences_grounded": len(allowed),
                "intent": intent
            }
        }
        
        if sync_info:
            response_data["sync_performed"] = sync_info

        return response_data

    except Exception as e:
        print("ERROR:", e)
        import traceback
        traceback.print_exc()
        return {"answer": "My brain just lagged. Say that again?"}
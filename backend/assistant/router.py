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
    chunks: list[str],
    top_k: int = 6,
    min_score: float = 0.52  # 🔥 threshold
) -> list[str]:
    if not chunks:
        return []

    sentences = split_into_sentences(chunks)
    if not sentences:
        return []

    scored = []
    for sentence in sentences:
        score = grounding_scorer.score(question, sentence)
        if score >= min_score:  # 🚫 filter weak sentences
            scored.append((score, sentence))

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

    print("🔄 SYNCING VAULT...")  # Terminal feedback
    
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
    
    print(f"✅ VAULT SYNCED: {sync_info['indexed_files']} files indexed")  # Terminal feedback
    
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
        
        # 1. Intent Classification
        intent = classify_intent(question)
        print(f"🎯 INTENT: {intent}")
        
        # Prevent continuation without context
        if intent == "continuation":
            previous_q = context_manager.get_previous_question()
            if not previous_q:
                return {"answer": "I don't have that information in my vault yet."}
            print(f"🔗 PREVIOUS Q: {previous_q}")
        
        # Clear session for new factual questions (will add to history after answer)
        if intent == "factual":
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

        # 3. RETRIEVAL - Use full question or previous question
        chunks = retrieve_for_question(question, intent, current_vault_data)
        print(f"📦 CHUNKS RETRIEVED: {len(chunks)}")
        
        if not chunks:
            response_data = {"answer": "I don't have that information in my vault yet."}
            if sync_info:
                response_data["sync_performed"] = sync_info
            return response_data

        # 4. ML-BASED GROUNDING
        allowed = ml_ground_sentences(question, chunks)
        print(f"✅ SENTENCES GROUNDED: {len(allowed)}")

        # 🔒 HARD REFUSAL - Check BEFORE trying to answer
        if not allowed:
            print("❌ NO GROUNDED SENTENCES - REFUSING")
            response_data = {"answer": "I don't have that information in my vault yet."}
            if sync_info:
                response_data["sync_performed"] = sync_info
            return response_data

        print(f"📄 ALLOWED SENTENCES:")
        for i, s in enumerate(allowed, 1):
            print(f"  {i}. {s}")

        allowed_text = "\n".join(f"- {s}" for s in allowed)

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
            if intent == "factual":
                context_manager.add_turn(question, answer)

        # 7. Build response with sync info
        response_data = {
            "answer": answer,
            "metadata": {
                "chunks_retrieved": len(chunks),
                "sentences_grounded": len(allowed),
                "intent": intent
            }
        }
        
        # Add sync info if sync was performed
        if sync_info:
            response_data["sync_performed"] = sync_info

        return response_data

    except Exception as e:
        print("ERROR:", e)
        return {"answer": "My brain just lagged. Say that again?"}
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import ollama
import re
import time

from vault.ingest import scan_vault, retrieve_relevant_chunks
from config import VAULT_PATH
from context_manager import context_manager

# =========================
# Global vault state
# =========================
current_vault_data = None
last_vault_mtime = None

router = APIRouter()

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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# =========================
# Intent Classification
# =========================
def classify_intent(question: str) -> str:
    res = ollama.generate(
        model="qwen2.5:7b",
        prompt=f"""
Classify the user's message into ONE category.

Categories:
- factual: Asking for information or explanation about a specific topic
- continuation: Referring to previous context (uses "it", "that", "again", "more")
- casual: Casual chat, greetings, or general conversation

Rules:
- If the message contains "it", "that", "again", or "more" → continuation
- If asking to explain/rephrase something differently → continuation
- If asking about a new specific topic → factual
- If greeting or chatting → casual

Examples:
"What is CPU scheduling?" → factual
"Explain it again" → continuation
"Tell me more" → continuation
"In simpler terms" → continuation
"Explain it in simpler terms" → continuation
"What does the A stand for?" → factual
"Hello" → casual

Message:
{question}

Category:""",
        options={"temperature": 0.0, "num_predict": 5},
    )
    return res["response"].strip().lower()


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
    
    # First time check
    if last_vault_mtime is None:
        return True
    
    # No vault directory
    if latest is None:
        return False
    
    # Compare modification times
    return latest > last_vault_mtime


# =========================
# Sync Vault
# =========================
@router.post("/sync")
def sync_vault():
    global current_vault_data, last_vault_mtime

    current_vault_data = scan_vault()
    last_vault_mtime = get_latest_vault_mtime()
    indexed_at = time.time()

    return {
        "vault_path": current_vault_data["vault_path"],
        "file_count": current_vault_data["file_count"],
        "empty_files": current_vault_data["empty_files"],
        "indexed_files": current_vault_data["indexed_files"],
        "last_indexed": indexed_at
    }


# =========================
# Subject Extraction
# =========================
def extract_subject_tokens(question: str) -> list[str]:
    STOP = {
        "what","whats","is","are","does","do","did","explain","define","describe",
        "tell","me","about","in","of","the","simple","terms","please",
        "how","why","when","where","which","again","it","that","this",
        "can","you","could","would","should","will","may","might","say"
    }
    return [w for w in tokenize(question) if w not in STOP]


def get_active_subject() -> Optional[str]:
    """Get active subject from context manager"""
    return context_manager.get_active_subject()


# =========================
# Ask (MAIN)
# =========================
@router.post("/ask")
def ask(req: AskRequest):
    global current_vault_data

    try:
        # 0. Sync
        if current_vault_data is None or vault_has_changed():
            sync_vault()

        question = req.question.strip()
        
        # 1. Intent
        intent = classify_intent(question)
        
        # Clear context for new factual queries to avoid cross-conversation pollution
        if intent == "factual":
            context_manager.clear_session()

        # 2. Casual
        if intent == "casual":
            res = ollama.generate(
                model="mistral:7b-instruct",
                prompt=f"""
You are a friendly conversational assistant.
Keep it casual and short.

User:
{question}

Response:
""",
                options={"temperature": 0.7, "num_predict": 80},
            )
            return {"answer": res["response"].strip()}

        # 3. Retrieve (use active_subject for continuation queries)
        retrieval_query = question
        if intent == "continuation":
            active_subj = get_active_subject()
            if active_subj:
                retrieval_query = active_subj
        
        results = retrieve_relevant_chunks(retrieval_query, current_vault_data, limit=5)
        if not results:
            return {"answer": "I don't have that information in my vault yet."}

        chunks = normalize_chunks(results)
        vault_text = " ".join(chunks).lower()

        # 4. Subject resolution
        subjects = extract_subject_tokens(question)

        # Helper function for anchor checking
        def subject_anchored(subject: str, vault_text: str) -> bool:
            # For multi-word subjects, check if they appear as a phrase (within 2 words)
            words = subject.split()
            if len(words) == 1:
                # Single word: just check if it exists
                return words[0] in vault_text
            else:
                # Multi-word: check if words appear close together (phrase matching)
                # This prevents false positives like "algorithm" and "mention" scattered in text
                vault_tokens = vault_text.split()
                for i in range(len(vault_tokens) - len(words) + 1):
                    # Check if all subject words appear within a window
                    window = vault_tokens[i:i + len(words) + 2]  # Allow 2 extra words gap
                    if all(w in window for w in words):
                        return True
                return False

        # FIX: Try continuation if intent suggests it AND extracted subjects don't anchor
        if intent == "continuation":
            # Check if current subjects actually anchor to vault
            if not any(subject_anchored(s, vault_text) for s in subjects):
                last = get_active_subject()
                if last:
                    subjects = [last]

        if not subjects:
            return {"answer": "I don't have that information in my vault yet."}

        # 5. Anchor check - ensure the primary subject (first/most specific token) exists
        # This prevents false positives from generic words like "operating", "systems"
        # Only check for factual queries; continuations already resolved their subject
        if intent == "factual":
            primary_subject = subjects[0]  # First extracted subject is usually most specific
            if primary_subject not in vault_text:
                return {"answer": "I don't have that information in my vault yet."}


        # 6. Sentence grounding
        allowed = []
        for chunk in chunks:
            for sent in re.split(r'(?<=[.!?])\s+', chunk):
                sent_l = sent.lower()
                sent_words = tokenize(sent_l)
                
                # Check if any subject's words appear in the sentence
                for s in subjects:
                    subject_words = tokenize(s)
                    # All subject words must be present
                    if all(w in sent_words for w in subject_words):
                        # For multi-word subjects, require words to appear early/prominently
                        # (within first 10 words) to avoid tangential mentions
                        if len(subject_words) > 1:
                            first_positions = [sent_words.index(w) for w in subject_words if w in sent_words]
                            if first_positions and min(first_positions) < 10:
                                allowed.append(sent.strip())
                                break
                        else:
                            allowed.append(sent.strip())
                            break
        
        allowed = list(dict.fromkeys(allowed))
        if not allowed:
            return {"answer": "I don't have that information in my vault yet."}

        allowed_text = "\n".join(f"- {s}" for s in allowed)
        
        # Extract key subject from allowed sentences for context memory
        allowed_lower = " ".join(allowed).lower()
        allowed_subjects = extract_subject_tokens(allowed_lower)

        # 7. Transform
        response = ollama.generate(
            model="qwen2.5:7b",
            prompt=f"""
You are a language transformer.

Rewrite ONLY the sentences below to answer the question.

RULES:
- Use ONLY the allowed sentences
- Do NOT add new information
- Do NOT explain or infer
- Do NOT mention missing knowledge
- Do NOT wrap the answer in quotes

YOU MAY:
- Rephrase
- Merge
- Reorder
- Remove redundancy

ALLOWED SENTENCES:
{allowed_text}

QUESTION:
{question}

ANSWER:
""",
            options={"temperature": 0.0, "top_p": 0.1, "num_predict": 140},
        )

        answer = response["response"].strip()

        # 8. Context memory - save active subject for continuation queries
        # Priority: allowed_subjects (core concepts) > query_subjects (may be redundant)
        
        if allowed_subjects:
            # Use the core concepts from vault (e.g., "round robin" not "round robin scheduling")
            active_subject = " ".join(allowed_subjects[:2])
        else:
            # Fallback to query subjects if we can't extract from answer
            query_subject_phrase = " ".join(subjects) if subjects else ""
            active_subject = query_subject_phrase if query_subject_phrase else None
        
        if active_subject:
            context_manager.set_active_subject(active_subject)

        return {
            "answer": answer,
            "vault_status": {
                "file_count": current_vault_data.get("file_count", 0),
                "indexed_files": current_vault_data.get("indexed_files", []),
                "empty_files": current_vault_data.get("empty_files", []),
                "last_indexed": last_vault_mtime
            }
        }

    except Exception as e:
        print("ERROR:", e)
        return {"answer": "My brain just lagged. Say that again?"}
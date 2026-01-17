from fastapi import APIRouter
from pydantic import BaseModel
import ollama
import re

from vault.ingest import scan_vault, retrieve_relevant_chunks
from config import VAULT_PATH
from memory.context import get_context_block, update_context
from memory.extractor import extract_context_facts

# =========================
# Global vault state
# =========================
current_vault_data = None
last_vault_mtime = None

# Global context state (workaround for context storage issues)
active_subject_cache = None

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

- factual
- continuation
- casual

Respond with ONLY the category name.

Message:
{question}
""",
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
    global last_vault_mtime
    latest = get_latest_vault_mtime()
    if last_vault_mtime is None:
        return True
    return latest != last_vault_mtime


# =========================
# Sync Vault
# =========================
@router.post("/sync")
def sync_vault():
    global current_vault_data, last_vault_mtime
    current_vault_data = scan_vault()
    last_vault_mtime = get_latest_vault_mtime()
    return {"status": "vault synced"}


# =========================
# Subject Extraction
# =========================
def extract_subject_tokens(question: str) -> list[str]:
    STOP = {
        "what","is","are","does","do","did","explain","define","describe",
        "tell","me","about","in","of","the","simple","terms","please",
        "how","why","when","where","which","again","it","that","this"
    }
    return [w for w in tokenize(question) if w not in STOP]


def get_active_subject():
    global active_subject_cache
    # Try context storage first
    ctx = get_context_block()
    if isinstance(ctx, dict):
        subject = ctx.get("active_subject")
        if subject:
            return subject
    # Fall back to cache
    return active_subject_cache


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
        
        # Global declaration must come first
        global active_subject_cache
        
        # 1. Intent
        intent = classify_intent(question)
        
        # Clear cache for new factual queries to avoid cross-conversation pollution
        if intent == "factual":
            active_subject_cache = None

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
        primary_subject = subjects[0]  # First extracted subject is usually most specific
        if primary_subject not in vault_text:
            return {"answer": "I don't have that information in my vault yet."}


        # 6. Sentence grounding
        allowed = []
        for chunk in chunks:
            for sent in re.split(r'(?<=[.!?])\s+', chunk):
                sent_l = sent.lower()
                if any(s in sent_l for s in subjects):
                    allowed.append(sent.strip())

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

        # 8. Context memory
        # Save the subject based on what was actually discussed
        # Priority: query subjects (if they anchored as phrase) > allowed_subjects > None
        
        # Check if query subjects actually anchored to vault as a meaningful phrase
        # We need the full subject phrase to anchor, not just individual words
        query_subject_phrase = " ".join(subjects) if subjects else ""
        query_subjects_anchored = query_subject_phrase and subject_anchored(query_subject_phrase, vault_text)
        
        if query_subjects_anchored:
            # Use the query subjects since they successfully matched the vault
            active_subject = " ".join(subjects)
        elif allowed_subjects:
            # Query subjects were noise words, use what was actually in the answer
            active_subject = " ".join(allowed_subjects[:2])
        else:
            active_subject = None
        
        facts = extract_context_facts(question, answer)
        ctx = {"active_subject": active_subject}
        if isinstance(facts, dict):
            ctx.update(facts)
        update_context(ctx)
        # Also cache it globally as a backup
        active_subject_cache = active_subject

        return {"answer": answer}

    except Exception as e:
        print("ERROR:", e)
        return {"answer": "My brain just lagged. Say that again?"}
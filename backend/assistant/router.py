from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
import ollama

from vault.ingest import scan_vault, retrieve_relevant_chunks, vector_store
from config import VAULT_PATH
from memory.memory import load_memory   # needed for teach mode


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

class QuizRequest(BaseModel):
    topic: str
    answer: str | None = None


# =========================
# Vault change detection
# =========================
def get_latest_vault_mtime() -> float | None:
    if not VAULT_PATH.exists():
        return None

    mtimes = []
    for path in VAULT_PATH.rglob("*"):
        if path.is_file():
            mtimes.append(path.stat().st_mtime)

    return max(mtimes) if mtimes else None


def vault_has_changed() -> bool:
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

    vault_data = scan_vault()
    current_vault_data = vault_data
    last_vault_mtime = get_latest_vault_mtime()

    return {
        "status": "vault synced",
        "file_count": vault_data.get("file_count", 0),
    }


# =========================
# Ask (MAIN)
# =========================
@router.post("/ask")
def ask(req: AskRequest):
    global current_vault_data

    try:
        if current_vault_data is None or vault_has_changed():
            sync_vault()

        question = req.question.strip()
        style = "casual"

        # 1. Retrieval
        context_text = ""
        if current_vault_data:
            results = retrieve_relevant_chunks(question, vault_data=current_vault_data, limit=3)
            context_text = "\n\n".join(r["chunk"] for r in results).strip()

        # 2. The Dynamic Locked Prompt
        # We tell the AI what the boundaries are, but let it choose its own words.
        prompt = f"""
### IDENTITY
You are a chill person. You have permanent amnesia regarding facts, science, math, and general world knowledge. 
You ONLY know what is written in the VAULT KNOWLEDGE provided below.

### VAULT KNOWLEDGE
{context_text if context_text else "--- THE VAULT IS EMPTY ---"}

### USER QUESTION
"{question}"

### MANDATORY RULES
- If the user is just vibing (hi, hello, etc.), chat back casually.
- If the user asks for information NOT found in the VAULT KNOWLEDGE:
    1. You are FORBIDDEN from answering using your own brain.
    2. Instead, tell the user in your own chill style that you don't have that info in your vault yet.
    3. Do NOT provide definitions, math results, or explanations from outside the vault.
- Style: {style}.
- No "Hey there!" or "Based on the vault". Just talk.

### RESPONSE
"""

        response = ollama.generate(
            model="mistral:7b-instruct",
            prompt=prompt,
            options={
                "temperature": 0.0,  # Critical for keeping the "Amnesia" boundary strict
                "top_p": 0.1,
                "num_predict": 150
            }
        )

        return {"answer": response["response"].strip()}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"answer": "My brain just lagged. Say that again?"}
# =========================
# Summarize (stub)
# =========================
@router.post("/summarize")
def summarize():
    return {"summary": "Summarize mode response (stub)"}


# =========================
# Teach Mode
# =========================
@router.post("/teach")
def teach(req: AskRequest):
    scan_vault()
    memory = load_memory()
    context = vector_store.search(req.question)

    return {
        "mode": "teach",
        "question": req.question,
        "explanation_style": memory.get("learning_style"),
        "steps": [
            "First, let's understand the core idea.",
            "Then we'll look at an example.",
            "Finally, I'll ask you a question to check understanding.",
        ],
        "context": context,
    }


# =========================
# Quiz Mode
# =========================
@router.post("/quiz")
def quiz(req: QuizRequest):
    scan_vault()
    context = vector_store.search(req.topic)

    if req.answer is None:
        return {
            "mode": "quiz",
            "question": f"Can you explain: {req.topic}?",
            "context_hint": context[:1],
        }

    return {
        "mode": "quiz",
        "your_answer": req.answer,
        "feedback": "Good attempt. Here’s what matters most:",
        "reference": context[:1],
    }

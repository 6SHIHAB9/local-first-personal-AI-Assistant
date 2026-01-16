from fastapi import APIRouter
from pydantic import BaseModel
from vault.ingest import scan_vault, retrieve_relevant_chunks, vector_store
from memory.memory import load_memory
import ollama

# =========================
# Global vault state
# =========================
current_vault_data = None

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
# Sync Vault
# =========================
@router.post("/sync")
def sync_vault():
    global current_vault_data
    vault_data = scan_vault()
    current_vault_data = vault_data

    return {
        "status": "vault synced",
        "file_count": vault_data.get("file_count", 0)
    }


# =========================
# Ask (MAIN BRAIN)
# =========================
@router.post("/ask")
def ask(req: AskRequest):
    try:
        results = []
        context_text = ""
        question = req.question.strip()

        # 1. FASTER CHECK: Handle basic greetings immediately without an LLM call
        greetings = {"hi", "hello", "hey", "hola", "yo", "hii", "hi there"}
        if question.lower().rstrip("?!.") in greetings:
            return {"answer": "Hey there! How can I help you today?"}

        if current_vault_data:
            results = retrieve_relevant_chunks(
                question,
                vault_data=current_vault_data,
                limit=2
            )
            context_text = "\n\n".join(r["chunk"] for r in results)

        # =========================
        # NO KNOWLEDGE AVAILABLE
        # =========================
        if not context_text:
            # Step 1: More robust intent classification
            intent_check_resp = ollama.generate(
                model="qwen2.5:7b",
                prompt=f"Classify this message as 'casual' (greetings, small talk) or 'info' (questions about facts/data). Reply with only the word.\n\nUser: {question}\n\nCategory:",
                options={"temperature": 0, "num_predict": 5}
            )["response"].strip().lower()

            # Use 'in' to catch "Casual." or "It's casual" 
            if "casual" in intent_check_resp:
                casual_response = ollama.generate(
                    model="qwen2.5:7b",
                    prompt=f"Respond to this greeting naturally and briefly (1 sentence). No help offers. No mention of files.\n\nUser: {question}",
                    options={"temperature": 0.7, "num_predict": 40}
                )
                return {"answer": casual_response["response"].strip()}

            # Information request → dynamic refusal
            refusal_response = ollama.generate(
                model="qwen2.5:7b",
                prompt=f'''The user is asking for information you don't have.
                  Tell them you don't know yet and suggest adding files to the vault so you can sync and help.
                    Keep it friendly and short.Be casual and honest.No excitement.
                    No “happy to help”.Sound like a normal person.Language:
                    - Respond in English only.
                    - Never use any other language.
                    \n\nUser: {question}''',
                options={"temperature": 0.5, "num_predict": 80}
            )
            return {"answer": refusal_response["response"].strip()}

        # =========================
        # KNOWLEDGE MODE
        # =========================
        knowledge_prompt = f"""
        You are a conversational assistant. Use the provided context to answer. 
        Don't mention 'the context' or 'the files'. Just answer.

        Context: {context_text}
        User: {question}
        """

        response = ollama.generate(
            model="qwen2.5:7b",
            prompt=knowledge_prompt,
            options={"temperature": 0.3, "num_predict": 200}
        )

        return {"answer": response["response"].strip()}

    except Exception as e:
        print("ASK ERROR:", e)
        return {"answer": "The assistant ran into an internal error."}

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
            "Finally, I'll ask you a question to check understanding."
        ],
        "context": context
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
            "context_hint": context[:1]
        }

    return {
        "mode": "quiz",
        "your_answer": req.answer,
        "feedback": "Good attempt. Here’s what matters most:",
        "reference": context[:1]
    }

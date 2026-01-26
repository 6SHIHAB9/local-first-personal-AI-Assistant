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


# =========================
# Intent Classification
# =========================
def classify_intent(question: str) -> str:
    """
    Classify user intent with conversation history (last 3 questions).
    """
    # Get last 3 questions for context
    history = context_manager.get_last_n_questions(n=3)
    
    # Build context for the LLM
    context_text = ""
    if history:
        context_lines = []
        for i, prev_q in enumerate(history):
            # Most recent = position 1
            position = i + 1
            context_lines.append(f"Question {position} (most recent): {prev_q}")
        context_text = "\nCONVERSATION HISTORY:\n" + "\n".join(context_lines)
    
    res = ollama.generate(
        model="qwen2.5:7b",
        prompt=f"""Classify the user's message into ONE category.
{context_text}

Categories:
- factual: Asking for new information about a specific topic
- continuation: Referring to any previous question (follow-up, clarification, rephrasing request)
- casual: Casual chat, greetings, or general conversation

CRITICAL RULES FOR DETECTING CONTINUATION:
1. If the message contains pronouns (it, that, this, them, they) AND there's conversation history → ALWAYS continuation
2. If asking "why/how/what" + pronoun (e.g., "Why is it important?") → ALWAYS continuation
3. Words like "again", "more", "further", "elaborate" → ALWAYS continuation
4. If the message only makes sense in context of previous questions → continuation
5. ONLY classify as factual if asking about a completely NEW topic with NO pronouns

Examples WITH history showing PRONOUNS = CONTINUATION:
History: "What is Stoicism?"
Current: "Why is it important?" → continuation (contains "it")
Current: "Tell me more about it" → continuation (contains "it")
Current: "How does that work?" → continuation (contains "that")
Current: "Explain this better" → continuation (contains "this")

History: "What is caramelization?"
Current: "Why shouldn't you rush it?" → continuation (contains "it")

History: "What is mycelium?"  
Current: "What is photosynthesis?" → factual (new topic, no pronouns)

IMPORTANT: If you see "it", "that", "this" in the current message → It's ALWAYS a continuation!

CURRENT MESSAGE:
{question}

Category:""",
        options={"temperature": 0.0, "num_predict": 5},
    )
    
    return res["response"].strip().lower()


# =========================
# LLM-BASED RETRIEVAL
# =========================
def retrieve_for_question(question: str, intent: str, vault_data: dict) -> list[str]:
    """
    Retrieve chunks using semantic search.
    For continuations, resolve pronouns using previous context.
    """
    retrieval_query = question
    
    if intent == "continuation":
        previous_q = context_manager.get_previous_question()
        if previous_q:
            # Extract the topic from the previous question
            # This helps resolve pronouns like "it", "that", "this"
            
            # Simple pronoun resolution using LLM
            res = ollama.generate(
                model="qwen2.5:7b",
                prompt=f"""Resolve the pronouns in the current question using the previous question's topic.

PREVIOUS QUESTION: {previous_q}
CURRENT QUESTION: {question}

TASK:
Replace pronouns (it, that, this, them) in the current question with the actual topic from the previous question.

Examples:
Previous: "What is caramelization?"
Current: "Why shouldn't you rush it?"
Output: "Why shouldn't you rush caramelization?"

Previous: "What is mycelium?"
Current: "Tell me more"
Output: "Tell me more about mycelium"

Previous: "What is Stoicism?"
Current: "Explain it again"
Output: "Explain Stoicism again"

Output ONLY the resolved question, nothing else.

Resolved question:""",
                options={"temperature": 0.0, "num_predict": 20},
            )
            
            resolved = res["response"].strip()
            if resolved and len(resolved) > 3:
                retrieval_query = resolved
    
    results = retrieve_relevant_chunks(retrieval_query, vault_data, limit=5)
    return normalize_chunks(results)


# =========================
# LLM-BASED GROUNDING
# =========================
def llm_ground_sentences(question: str, chunks: list[str], intent: str) -> list[str]:
    """
    Use LLM to extract relevant sentences from chunks.
    This is more reliable than keyword matching.
    """
    if not chunks:
        return []
    
    # Build context for the LLM
    context_instruction = ""
    if intent == "continuation":
        previous_q = context_manager.get_previous_question()
        if previous_q:
            context_instruction = f"""CONTEXT: This is a follow-up question.
PREVIOUS QUESTION: {previous_q}

The current question refers to the topic from the previous question.
- Pronouns like "it", "that", "this" refer to the previous topic
- Extract sentences that help answer the current question about that topic
"""
    
    # Combine chunks
    chunks_text = "\n\n".join(f"CHUNK {i+1}:\n{chunk}" for i, chunk in enumerate(chunks[:5]))
    
    res = ollama.generate(
        model="qwen2.5:7b",
        prompt=f"""You are extracting relevant sentences from text chunks to answer a question.

{context_instruction}

CURRENT QUESTION:
{question}

TEXT CHUNKS:
{chunks_text}

INSTRUCTIONS:
Extract ALL sentences that could help answer the question.

Be INCLUSIVE and INTERPRETIVE:
- Include sentences that DIRECTLY answer the question
- Include sentences that INDIRECTLY answer the question (through implications, benefits, consequences, examples)
- Include sentences that provide necessary CONTEXT for understanding the answer
- When in doubt, INCLUDE rather than exclude

Think broadly about relevance:
- A question about "importance" can be answered by sentences about benefits, outcomes, or effects
- A question about "why" can be answered by sentences about causes, purposes, or consequences  
- A question about "how" can be answered by sentences about processes, mechanisms, or methods
- A question about symbolism can be answered by descriptive sentences

OUTPUT FORMAT:
Return ONLY the relevant sentences, one per line.
Do NOT add explanations or commentary.
If absolutely no relevant sentences found, output: NONE

Relevant sentences:
""",
        options={"temperature": 0.0, "num_predict": 300},
    )
    
    output = res["response"].strip()
    
    if output == "NONE" or not output:
        return []
    
    # Split into sentences and clean
    sentences = [s.strip() for s in output.split('\n') if s.strip()]
    
    # Remove any numbering or bullets
    sentences = [re.sub(r'^[\d\-\*\.]+\s*', '', s) for s in sentences]
    
    # Remove duplicates
    return list(dict.fromkeys([s for s in sentences if len(s) > 10]))


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
        
        # 1. Intent Classification
        intent = classify_intent(question)
        
        # Prevent continuation without context
        if intent == "continuation":
            previous_q = context_manager.get_previous_question()
            if not previous_q:
                return {"answer": "I don't have that information in my vault yet."}
        
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
        
        if not chunks:
            response_data = {"answer": "I don't have that information in my vault yet."}
            if sync_info:
                response_data["sync_performed"] = sync_info
            return response_data

        # 4. LLM-BASED GROUNDING - Let LLM extract relevant sentences
        allowed = llm_ground_sentences(question, chunks, intent)

        if not allowed:
            response_data = {"answer": "I don't have that information in my vault yet."}
            if sync_info:
                response_data["sync_performed"] = sync_info
            return response_data

        allowed_text = "\n".join(f"- {s}" for s in allowed)

        # 5. ANSWER GENERATION
        # Build context for continuation
        context_instruction = ""
        if intent == "continuation":
            previous_q = context_manager.get_previous_question()
            if previous_q:
                prev_lower = previous_q.lower()
                if prev_lower.startswith("why"):
                    context_instruction = "CONTEXT: The original question asked WHY. Focus on explaining the REASON or CAUSE.\n"
                elif prev_lower.startswith("how"):
                    context_instruction = "CONTEXT: The original question asked HOW. Focus on explaining the PROCESS or MECHANISM.\n"
                else:
                    context_instruction = f"CONTEXT: This is a follow-up to: {previous_q}\n"
        
        response = ollama.generate(
            model="qwen2.5:7b",
            prompt=f"""You are answering a question using ONLY the provided sentences.

RULES:
- Use ONLY the allowed sentences below
- You MAY rephrase, combine, and simplify them
- Do NOT add information not in the sentences
- Do NOT wrap answer in quotes
- Keep the answer clear and direct
{context_instruction}
ALLOWED SENTENCES:
{allowed_text}

QUESTION:
{question}

ANSWER:
""",
            options={"temperature": 0.0, "top_p": 0.1, "num_predict": 150},
        )

        answer = response["response"].strip()

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
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
# Comparison Detection
# =========================
def is_comparison_question(question: str) -> bool:
    """Detect if question is asking for a comparison"""
    question_lower = question.lower()
    comparison_keywords = ["compare", "difference", "vs", "versus"]
    return any(keyword in question_lower for keyword in comparison_keywords)


def extract_comparison_concepts(question: str) -> list[str]:
    """
    Extract concept phrases from comparison questions.
    Example: "Compare round robin and priority scheduling" 
    -> ["round robin", "priority"]
    
    Strategy: Split on "and" and "or", then extract subject tokens from each part.
    Returns the core concept from each side of the comparison.
    """
    question_lower = question.lower()
    
    # Remove comparison keywords
    for keyword in ["compare", "difference between", "vs", "versus", "the", "scheduling"]:
        question_lower = question_lower.replace(keyword, "")
    
    # Split on conjunctions that separate concepts
    parts = re.split(r'\s+(?:and|or)\s+', question_lower.strip())
    
    # Extract subject tokens from each part
    STOP = {
        "what","whats","is","are","does","do","did","explain","define","describe",
        "tell","me","about","in","of","the","simple","terms","please",
        "how","why","when","where","which","again","it","that","this",
        "can","you","could","would","should","will","may","might","say","a","an"
    }
    
    concepts = []
    for part in parts:
        tokens = [w for w in tokenize(part) if w not in STOP]
        if tokens:
            # Use the core concept (first significant tokens)
            concepts.append(" ".join(tokens))
    
    return concepts


# =========================
# PLANNING STAGE
# =========================
def plan_subquestions(question: str, intent: str) -> list[str]:
    """
    Break down complex questions into sub-questions for multi-hop retrieval.
    This improves coverage by retrieving information from multiple angles.
    """
    if intent == "casual":
        return []
    
    if intent == "continuation":
        # For continuations, generate 1-2 clarifying sub-questions using active subject
        active_subj = get_active_subject()
        if not active_subj:
            return []
        
        res = ollama.generate(
            model="qwen2.5:7b",
            prompt=f"""
Generate 1-2 focused sub-questions to clarify the user's request about "{active_subj}".

User's request: {question}
Active subject: {active_subj}

Rules:
- Generate ONLY the sub-questions, one per line
- Each sub-question must relate to "{active_subj}"
- Keep sub-questions concise and specific
- Do NOT explain or add commentary

Sub-questions:
""",
            options={"temperature": 0.0, "num_predict": 50},
        )
        subqs = [line.strip() for line in res["response"].strip().split('\n') if line.strip()]
        # Clean up numbering/bullets if present
        subqs = [re.sub(r'^[\d\-\*\.]+\s*', '', sq) for sq in subqs]
        return [sq for sq in subqs if len(sq) > 5][:2]  # Max 2 sub-questions
    
    if intent == "factual":
        # For factual queries, decompose into 2-4 sub-questions
        res = ollama.generate(
            model="qwen2.5:7b",
            prompt=f"""
Break down this question into 2-4 focused sub-questions that would help answer it comprehensively.

Question: {question}

Rules:
- Generate 2-4 sub-questions, one per line
- Each sub-question should focus on a specific aspect
- Keep sub-questions concise and specific
- Do NOT explain or add commentary

Sub-questions:
""",
            options={"temperature": 0.0, "num_predict": 80},
        )
        subqs = [line.strip() for line in res["response"].strip().split('\n') if line.strip()]
        # Clean up numbering/bullets if present
        subqs = [re.sub(r'^[\d\-\*\.]+\s*', '', sq) for sq in subqs]
        return [sq for sq in subqs if len(sq) > 5][:4]  # Max 4 sub-questions
    
    return []


# =========================
# MULTI-HOP RETRIEVAL
# =========================
def retrieve_with_planning(question: str, intent: str, retrieval_query: str, vault_data: dict) -> list[str]:
    """
    Perform multi-hop retrieval using sub-questions, then merge and deduplicate results.
    Falls back to original behavior if no sub-questions are generated.
    """
    subquestions = plan_subquestions(question, intent)
    
    if not subquestions:
        # No planning needed - use original retrieval behavior
        results = retrieve_relevant_chunks(retrieval_query, vault_data, limit=5)
        return normalize_chunks(results)
    
    # Multi-hop retrieval: retrieve for each sub-question
    all_chunks = []
    seen_chunks = set()
    
    # Retrieve for main question
    main_results = retrieve_relevant_chunks(retrieval_query, vault_data, limit=5)
    for chunk in normalize_chunks(main_results):
        chunk_key = chunk.lower().strip()
        if chunk_key not in seen_chunks:
            all_chunks.append(chunk)
            seen_chunks.add(chunk_key)
    
    # Retrieve for each sub-question
    for subq in subquestions:
        sub_results = retrieve_relevant_chunks(subq, vault_data, limit=3)
        for chunk in normalize_chunks(sub_results):
            chunk_key = chunk.lower().strip()
            if chunk_key not in seen_chunks:
                all_chunks.append(chunk)
                seen_chunks.add(chunk_key)
    
    return all_chunks


# =========================
# COVERAGE & CONFIDENCE ESTIMATION
# =========================
def estimate_confidence(num_chunks: int, num_grounded_sentences: int) -> dict:
    """
    Estimate confidence based on retrieval coverage and grounding quality.
    Returns confidence level and metadata.
    """
    if num_grounded_sentences >= 5:
        level = "HIGH"
    elif num_grounded_sentences >= 2:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    return {
        "level": level,
        "chunks_retrieved": num_chunks,
        "sentences_grounded": num_grounded_sentences
    }


# =========================
# SELF-CRITIQUE
# =========================
def critique_answer(answer: str, allowed_sentences: list[str]) -> dict:
    """
    Self-critique pass: verify that the answer is fully grounded in allowed sentences.
    Returns status and optional improvement note.
    """
    allowed_text = "\n".join(f"- {s}" for s in allowed_sentences)
    
    res = ollama.generate(
        model="qwen2.5:7b",
        prompt=f"""
Verify if the answer is fully supported by the allowed sentences.

ALLOWED SENTENCES:
{allowed_text}

ANSWER:
{answer}

Check:
- Is every claim in the answer supported by the allowed sentences?
- Is anything speculative or implied beyond what's stated?

Respond with ONLY a JSON object:
{{"status": "ok", "note": "fully grounded"}}
OR
{{"status": "needs_improvement", "note": "one short reason"}}

Response:
""",
        options={"temperature": 0.0, "num_predict": 50},
    )
    
    response_text = res["response"].strip()
    
    # Parse JSON response
    try:
        # Extract JSON if wrapped in markdown
        if "```" in response_text:
            response_text = re.search(r'\{[^}]+\}', response_text).group(0)
        
        # Simple JSON parsing
        status_match = re.search(r'"status"\s*:\s*"([^"]+)"', response_text)
        note_match = re.search(r'"note"\s*:\s*"([^"]+)"', response_text)
        
        status = status_match.group(1) if status_match else "ok"
        note = note_match.group(1) if note_match else ""
        
        return {
            "status": status,
            "note": note
        }
    except:
        # Default to ok if parsing fails
        return {"status": "ok", "note": ""}


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
        "can","you","could","would","should","will","may","might","say",
        # Action verbs that appear in questions but aren't subjects
        "reduce","reduces","increase","increases","improve","improves",
        "prevent","prevents","cause","causes","affect","affects",
        "eliminate","eliminates","create","creates","allow","allows",
        "enable","enables","make","makes","help","helps","use","uses",
        "work","works","happen","happens","occur","occurs"
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
        
        # BUG FIX 2: Prevent continuation after refusal
        # If active subject is None and intent is continuation, it means the previous
        # response was likely a refusal. Block continuation to prevent hallucination.
        if intent == "continuation":
            active_subj = get_active_subject()
            if not active_subj:
                return {"answer": "I don't have that information in my vault yet."}
        
        # Store previous question for factual queries
        if intent == "factual":
            # Clear session first, THEN store the new question
            # This ensures previous_question survives the clear
            context_manager.clear_session()
            context_manager.set_previous_question(question)

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

        # 3. MULTI-HOP RETRIEVAL (with planning if needed)
        retrieval_query = question
        if intent == "continuation":
            active_subj = get_active_subject()
            if active_subj:
                retrieval_query = active_subj
        
        # Use enhanced retrieval with planning
        chunks = retrieve_with_planning(question, intent, retrieval_query, current_vault_data)
        
        if not chunks:
            return {"answer": "I don't have that information in my vault yet."}

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

        # ABSOLUTE GUARD: Prevent out-of-domain hallucination
        # For comparison questions, check the actual comparison concepts (e.g., "round robin", "priority")
        # For other questions, require MAJORITY of subject tokens to anchor (not all, to handle synonyms)
        is_comparison = is_comparison_question(question)
        
        if is_comparison:
            # Use comparison concepts for anchoring (already multi-word phrases)
            comparison_concepts = extract_comparison_concepts(question)
            if not all(subject_anchored(concept, vault_text) for concept in comparison_concepts):
                return {"answer": "I don't have enough information in my vault to compare those topics."}
        else:
            # For non-comparison questions, require MAJORITY of subject tokens to anchor
            # This handles cases where questions use synonyms (reduce vs eliminate) or different word forms
            # But still prevents complete mismatches like "quantum tunneling" matching "time quantum"
            anchored_count = sum(1 for s in subjects if subject_anchored(s, vault_text))
            required_count = max(1, len(subjects) // 2 + 1)  # Majority, minimum 1
            
            if anchored_count < required_count:
                return {"answer": "I don't have that information in my vault yet."}

        # 5. Anchor check - ensure subjects exist in vault
        if intent == "factual":
            # For non-comparison factual queries, check primary subject
            if not is_comparison:
                primary_subject = subjects[0]
                if primary_subject not in vault_text:
                    return {"answer": "I don't have that information in my vault yet."}


        # 6. Sentence grounding - STRENGTHENED to prevent hallucination
        allowed = []
        
        # For continuation queries asking WHY/HOW, prioritize sentences with causal/explanatory words
        is_why_continuation = False
        is_how_continuation = False
        if intent == "continuation":
            previous_q = context_manager.get_previous_question()
            if previous_q:
                prev_lower = previous_q.lower()
                is_why_continuation = prev_lower.startswith("why")
                is_how_continuation = prev_lower.startswith("how")
        
        # Define explanatory/causal keywords
        explanatory_words = {"because", "since", "as", "due", "by", "through", "allows", "enables", "eliminates", "reduces", "prevents", "causes", "results", "leads", "means", "so", "thus", "therefore"}
        
        for chunk in chunks:
            for sent in re.split(r'(?<=[.!?])\s+', chunk):
                sent_l = sent.lower()
                sent_words = tokenize(sent_l)
                
                # Check if any subject's words appear in the sentence
                for s in subjects:
                    subject_words = tokenize(s)
                    
                    # STRENGTHENED MATCHING: Require majority of subject words to be present
                    matching_count = sum(1 for w in subject_words if w in sent_words)
                    required_match_count = max(1, len(subject_words) // 2 + 1)  # Majority
                    
                    if matching_count >= required_match_count:
                        # For multi-word subjects, require words to appear early/prominently
                        if len(subject_words) > 1:
                            first_positions = [sent_words.index(w) for w in subject_words if w in sent_words]
                            if not first_positions or min(first_positions) >= 10:
                                continue  # Skip this sentence
                        else:
                            # Single word: also check it appears early (within first 15 words)
                            if subject_words[0] not in sent_words[:15]:
                                continue  # Skip this sentence
                        
                        # INTENT FILTERING: For WHY/HOW continuations, be more selective
                        if is_why_continuation or is_how_continuation:
                            has_explanation = any(word in sent_words for word in explanatory_words)
                            
                            # Check for pure definition pattern (stricter check)
                            is_pure_definition = False
                            sent_text_lower = sent.lower()
                            # Only flag if it literally starts with "[subject] is a " or "[subject] is an "
                            for subj_word in subject_words:
                                pattern1 = f"{subj_word} is a "
                                pattern2 = f"{subj_word} is an "
                                if sent_text_lower.startswith(pattern1) or sent_text_lower.startswith(pattern2):
                                    is_pure_definition = True
                                    break
                            
                            # Accept if has explanation OR is not a pure definition
                            if has_explanation or not is_pure_definition:
                                # Prioritize explanatory sentences
                                if has_explanation:
                                    allowed.insert(0, sent.strip())
                                else:
                                    allowed.append(sent.strip())
                                break
                        else:
                            # For non-WHY/HOW queries, accept all matching sentences
                            allowed.append(sent.strip())
                            break
        
        allowed = list(dict.fromkeys(allowed))  # Remove duplicates while preserving order

        # ABSOLUTE RULE: If no grounded sentences, refuse immediately - DO NOT call LLM
        # This prevents hallucination on out-of-vault topics (e.g., quantum tunneling)
        if not allowed:
            return {"answer": "I don't have that information in my vault yet."}

        allowed_text = "\n".join(f"- {s}" for s in allowed)
        
        # COVERAGE & CONFIDENCE ESTIMATION
        confidence = estimate_confidence(len(chunks), len(allowed))
        
        # Extract key subject from allowed sentences for context memory
        allowed_lower = " ".join(allowed).lower()
        allowed_subjects = extract_subject_tokens(allowed_lower)

        # 7. Transform
        # Build intent instruction for continuation queries
        intent_instruction = ""
        if intent == "continuation":
            previous_q = context_manager.get_previous_question()
            if previous_q:
                prev_lower = previous_q.lower()
                if prev_lower.startswith("why"):
                    intent_instruction = """
CRITICAL INSTRUCTION - READ CAREFULLY:
The original question asked WHY (reason/cause).
You MUST answer WHY, explaining the REASON or CAUSE.
DO NOT answer WHAT (definition).
DO NOT describe the technique itself.
Focus ONLY on the reason/cause/explanation.
If the allowed sentences don't explain WHY, respond: "I don't have that information in my vault yet."
"""
                elif prev_lower.startswith("how"):
                    intent_instruction = """
CRITICAL INSTRUCTION - READ CAREFULLY:
The original question asked HOW (process/mechanism).
You MUST answer HOW, explaining the PROCESS or MECHANISM.
DO NOT answer WHAT (definition).
Focus ONLY on the process/mechanism.
If the allowed sentences don't explain HOW, respond: "I don't have that information in my vault yet."
"""
                elif prev_lower.startswith("what"):
                    intent_instruction = "\nThe original question asked WHAT. Provide a DEFINITION or DESCRIPTION."
        
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
- DO NOT wrap the answer in quotes
{intent_instruction}

ALLOWED SENTENCES:
{allowed_text}

ADDITIONAL TRANSFORM RULES:

- You MAY rephrase, merge, and simplify the allowed sentences.
- You do NOT need to preserve the original wording from the vault.

STRICT CONSTRAINTS:
- You MUST NOT introduce any information that is not explicitly present in the allowed sentences.
- Every fact in the answer must be directly supported by the allowed sentences.

CLEANUP RULES:
- Remove partial phrases, list headers, or sentence fragments.
- Do NOT include dangling words or prefixes (e.g. "Circular wait", "Techniques include:").
- Merge related sentences into clean, complete statements.

QUALITY RULES:
- Do NOT repeat the question in the answer.
- Do NOT add disclaimers if a complete answer is produced.
- If the allowed sentences do not clearly answer the question, respond exactly with:
  "I don't have that information in my vault yet."

QUESTION:
{question}

ANSWER:
""",
            options={"temperature": 0.0, "top_p": 0.1, "num_predict": 140},
        )

        answer = response["response"].strip()

        # HARD STOP: If LLM decided to refuse, return immediately without critique
        if answer == "I don't have that information in my vault yet.":
            return {"answer": answer}

        # SELF-CRITIQUE - verify answer is fully grounded
        critique = critique_answer(answer, allowed)
        
        # Annotate answer if critique suggests improvement needed
        if critique["status"] == "needs_improvement":
            answer = f"{answer}\n\nNote: Some parts of this answer may be incomplete based on the available vault data."

        # 8. Context memory - save active subject for continuation queries
        # Priority: allowed_subjects (core concepts) > query_subjects (may be redundant)
        
        # Enhanced stop words to filter out when extracting active subject
        ACTIVE_SUBJECT_STOP = {
            "what","whats","is","are","does","do","did","explain","define","describe",
            "tell","me","about","in","of","the","simple","terms","please",
            "how","why","when","where","which","again","it","that","this",
            "can","you","could","would","should","will","may","might","say",
            "a", "an", "and", "or", "but", "for", "to", "from", "with", "by",
            "used", "technique", "method", "system", "process", "management"
        }
        
        if allowed_subjects:
            # Filter out stop words more aggressively for active subject
            clean_subjects = [w for w in allowed_subjects if w not in ACTIVE_SUBJECT_STOP]
            if clean_subjects:
                active_subject = " ".join(clean_subjects[:2])  # Use top 2 clean subjects
            else:
                # Fallback if all filtered out
                active_subject = " ".join(allowed_subjects[:1])
        else:
            # Fallback to query subjects if we can't extract from answer
            query_subject_phrase = " ".join(subjects) if subjects else ""
            active_subject = query_subject_phrase if query_subject_phrase else None
        
        if active_subject:
            context_manager.set_active_subject(active_subject)

        return {
            "answer": answer,
            "confidence": confidence,
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
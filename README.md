# Local-First AI Assistant

A local-first AI assistant that answers questions **strictly using information from user-provided files**.

The assistant reads text documents from a local **vault** and generates answers **only when the information exists in those files**.  
If the information is not present, the system **refuses to answer instead of guessing**.

Everything runs locally.  
No data is uploaded.  
No external APIs are used.

---

## Core Idea

Most AI assistants prioritize fluent answers, even when information is missing.

This project prioritizes **correctness over fluency**.

> Answers are generated only when they can be fully grounded in user-provided data.

The system is designed to:
- avoid hallucinations
- refuse out-of-scope questions
- preserve intent across follow-up questions
- provide deterministic, explainable behavior

---

## How It Works

### Vault Ingestion

- Users place text files (`.txt`, `.md`) inside a local `vault/` directory
- Files are:
  - read locally
  - chunked into smaller segments
  - embedded using a local embedding model
  - stored in a local vector database

---

### Question Processing & Validation

Before answering, every user question goes through an **internal validation pipeline**:

1. **Intent classification**  
   The system determines whether the question is:
   - a new factual query  
   - a continuation of a previous question  
   - casual / non-informational  

2. **Subject resolution**  
   The system extracts the core subject(s) of the question and verifies that they are present in the retrieved vault content.

3. **Grounding check**  
   Retrieved document chunks are validated to ensure they actually support the question being asked.  
   If no valid grounding exists, the question is rejected.

This validation step prevents:
- answering unrelated questions
- leaking loosely related information
- accidental hallucinations due to semantic overlap

---

### Context Memory (Controlled)

The assistant maintains a **lightweight internal context memory** to support follow-up questions.

- The system tracks the **active subject** of the conversation
- Follow-up queries such as *“explain it again”* or *“explain it simply”* reuse the validated subject
- Context is **cleared automatically** when a new, unrelated factual question is asked

This ensures:
- continuity without long-term memory accumulation
- no cross-topic contamination
- predictable, bounded behavior

---

### Answer Generation

When a question passes validation:

1. Relevant chunks are retrieved using vector similarity
2. Individual sentences are filtered and cleaned
3. A language model rewrites the allowed sentences

**Rules:**
- The model may rephrase and merge sentences
- No new information may be introduced
- All facts must come directly from the vault

If a valid answer cannot be produced, the assistant responds with:

I don't have that information in my vault yet.


---

## Key Behaviors (By Design)

- **Strict grounding**  
  Answers are generated only from vault content

- **Out-of-scope refusal**  
  Questions not covered by the vault are explicitly rejected

- **No hallucinations**  
  The model is not allowed to invent or infer missing information

- **Intent-aware follow-ups**  
  Follow-up queries preserve the original intent (WHY vs WHAT)

- **Safe comparisons**  
  Comparisons are allowed only when all compared topics exist in the vault

- **Deterministic behavior**  
  No cloud calls, no hidden state, no nondeterminism

---

## Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** React + TypeScript
- **LLM:** Ollama (local models)
- **Embeddings:** nomic-embed-text (Ollama, local embedding model)
- **Vector Search:** FAISS (local vector database)

---

## Why Local-First?

- Full data privacy
- Works offline
- No API costs
- Complete user control over the knowledge source

---

## Project Status

**Stable and complete.**

The project implements a fully working local Retrieval-Augmented Generation (RAG) system with:
- grounded answers
- internal question validation
- controlled context memory
- safe refusals
- intent-preserving continuations
- controlled comparisons

Further improvements (UI polish, analytics, visualizations) are possible, but the **core system behavior is intentionally locked** to preserve correctness.

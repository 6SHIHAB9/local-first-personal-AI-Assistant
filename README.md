# Local-First AI Assistant

A local-first AI assistant that answers questions **only using information from user-provided files**.

The assistant reads text documents from a local **vault** and generates answers strictly grounded in those files.  
If the information is not present, the system **refuses to answer instead of guessing**.

Everything runs locally. No data is uploaded. No external APIs are used.

---

## Core Idea

Most AI assistants prioritize fluent responses, even when information is missing.

This project prioritizes **correctness and safety**.

> Answers are generated only when they can be fully grounded in user-provided data.

The system is designed to:
- avoid hallucinations
- refuse out-of-scope questions
- preserve intent across follow-up questions

---

## How It Works

### Vault Ingestion

- Users place text files (`.txt`, `.md`) inside a local `vault/` directory
- Files are:
  - read locally
  - chunked
  - embedded using a local embedding model
  - stored in a local vector database

### Question Answering

When a question is asked:

1. The query is embedded
2. Relevant chunks are retrieved using vector similarity
3. Retrieved content is validated against the question intent
4. An answer is generated **only from the retrieved content**

If no valid grounding exists, the assistant responds with:

I don't have that information in my vault yet.

yaml
Copy code

No fallback answers are generated.

---

## Key Behaviors

- **Strict grounding**  
  Answers are generated only from vault content

- **Out-of-scope refusal**  
  Questions not covered by the vault are explicitly rejected

- **No hallucinations**  
  The model is not allowed to invent or infer missing information

- **Intent-aware follow-ups**  
  Follow-up queries such as “explain it again” preserve the original intent (e.g. WHY vs WHAT)

- **Safe comparisons**  
  Comparisons are allowed only when all compared topics exist in the vault

- **Deterministic behavior**  
  No cloud calls, no hidden state, no nondeterminism

---

## Tech Stack

- Backend: FastAPI (Python)
- Frontend: React + TypeScript
- LLM: Ollama (local models)
- Embeddings: Local embedding model
- Vector Search: Local vector database

---

## Why Local-First?

- Full data privacy
- Works offline
- No API costs
- Complete control over the knowledge source

---

## Project Status

Stable and complete.

The project implements a fully working local Retrieval-Augmented Generation (RAG) system with:
- grounded answers
- safe refusals
- intent-preserving continuations
- controlled comparisons

Further improvements are possible, but the core system behavior is intentionally locked to preserve correctness.
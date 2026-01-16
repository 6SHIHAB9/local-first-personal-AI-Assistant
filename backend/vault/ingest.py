from pathlib import Path
import re

from config import VAULT_PATH
from vault.vector_store import VectorStore


# --------------------
# helpers
# --------------------

def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return ""


def chunk_text(text: str, chunk_size: int = 300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


# --------------------
# global vector store
# --------------------

vector_store = VectorStore()


# --------------------
# vault scan
# --------------------

def scan_vault():
    files = []
    all_chunks = []

    if not VAULT_PATH.exists():
        return {"error": "Vault folder not found"}

    for path in VAULT_PATH.rglob("*"):
        if path.suffix.lower() not in [".txt", ".md"]:
            continue

        content = read_text_file(path)
        chunks = chunk_text(content)

        files.append({
            "name": path.name,
            "path": str(path),
            "extension": path.suffix,
            "chunks": chunks
        })

        all_chunks.extend(chunks)

    # 🔥 build embeddings here
    vector_store.build(all_chunks)

    return {
        "vault_path": str(VAULT_PATH),
        "file_count": len(files),
        "files": files
    }


import re
from collections import defaultdict

MIN_SCORE = 0.2   # ← relevance threshold (important)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def keyword_score(query: str, text: str) -> float:
    q = tokenize(query)
    t = tokenize(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def extract_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("chunk") or item.get("text") or ""
    return ""


def extract_score(item):
    if isinstance(item, dict):
        return float(item.get("score", 1.0))
    return 1.0


def retrieve_relevant_chunks(query: str, vault_data: dict, limit: int = 3):
    scored = defaultdict(float)

    # -------------------------
    # 1. Semantic search
    # -------------------------
    semantic_results = vector_store.search(query, k=limit * 3)

    for r in semantic_results:
        text = extract_text(r)
        if not text:
            continue
        scored[text] += 0.7 * extract_score(r)

    # -------------------------
    # 2. Keyword overlap (FIXED)
    # -------------------------
    for file in vault_data.get("files", []):
        for chunk in file.get("chunks", []):
            if not isinstance(chunk, str):
                continue

            ks = keyword_score(query, chunk)
            if ks > 0:
                scored[chunk] += 0.3 * ks

    # -------------------------
    # 3. Filter + rank
    # -------------------------
    ranked = sorted(
        scored.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {"chunk": text, "score": score}
        for text, score in ranked
        if score >= MIN_SCORE
    ][:limit]

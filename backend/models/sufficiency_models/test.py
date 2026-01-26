# backend/models/sufficiency_models/test.py

import torch
from transformers import AutoTokenizer
from model import SufficiencyModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🧠 Device: {DEVICE}")

# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    local_files_only=True
)

model = SufficiencyModel(BASE_MODEL)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# SCORING FUNCTION
# =========================
@torch.inference_mode()
def score(text: str) -> float:
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    return model(**enc).item()

# =========================
# TEST CASES
# =========================

GOOD = """Question: Why does database indexing improve query performance?
Intent: continuation
Evidence:
1. Database indexes create optimized data structures that map column values to row locations.
2. Queries can use indexes to avoid scanning entire tables.
3. This reduces disk I/O and significantly lowers query execution time.
"""

HARD_NEG = """Question: Why does database indexing improve query performance?
Intent: continuation
Evidence:
1. Database indexing is widely used in relational database systems.
2. Query performance depends on execution time and resource usage.
3. Indexes are an important aspect of database design.
"""

FLUENCY_TRAP = """Question: Why does database indexing improve query performance?
Intent: continuation
Evidence:
1. Database indexing is a common optimization technique.
2. Large databases often contain millions of records.
3. Performance considerations are important when designing database schemas.
"""

EXTREME_NEG = """Question: Why does database indexing improve query performance?
Intent: continuation
Evidence:
1. Databases store structured information.
"""

# =========================
# RUN TESTS
# =========================
print("\n=== SUFFICIENCY SCORER CALIBRATION ===")
print(f"✅ GOOD        : {score(GOOD):.4f}")
print(f"⚠️  HARD NEG   : {score(HARD_NEG):.4f}")
print(f"🟡 FLUENCY     : {score(FLUENCY_TRAP):.4f}")
print(f"❌ EXTREME NEG: {score(EXTREME_NEG):.4f}")

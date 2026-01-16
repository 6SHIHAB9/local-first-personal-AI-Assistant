import json
from pathlib import Path

MEMORY_PATH = Path(__file__).parent / "memory.json"

def load_memory():
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory: dict):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
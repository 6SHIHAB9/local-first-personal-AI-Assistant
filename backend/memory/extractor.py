# memory/extractor.py
import ollama

def extract_context_facts(question: str, answer: str) -> list[str]:
    prompt = f"""
Extract short, temporary context facts from the exchange below.
Rules:
- Only conversational context (goals, preferences, topic)
- NOT factual knowledge
- NOT definitions
- Max 3 items
Return as a JSON array of strings.

USER:
{question}

ASSISTANT:
{answer}
"""

    res = ollama.generate(
        model="mistral:7b-instruct",
        prompt=prompt,
        options={"temperature": 0.0, "num_predict": 100},
    )

    try:
        return eval(res["response"])  # safe enough for local
    except:
        return []

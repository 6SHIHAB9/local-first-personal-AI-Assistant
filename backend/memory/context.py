# memory/context.py

_context = {
    "facts": [],
    "current_topic": None,
}

MAX_FACTS = 6


def get_context_block() -> str:
    if not _context["facts"]:
        return "None"
    return "\n".join(f"- {f}" for f in _context["facts"])


def update_context(new_facts: list[str]):
    for fact in new_facts:
        if fact not in _context["facts"]:
            _context["facts"].append(fact)

    # trim oldest
    _context["facts"] = _context["facts"][-MAX_FACTS:]


def reset_context():
    _context["facts"].clear()
    _context["current_topic"] = None

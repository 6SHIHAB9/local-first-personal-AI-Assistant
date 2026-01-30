"""
Context Manager for conversation state.
Handles conversation history AND topic anchoring.
"""

class ContextManager:
    def __init__(self, max_history: int = 3):
        self.history = []  # list of {"question": str, "answer": str}
        self.max_history = max_history
        self.topic_anchor = None

    # =========================
    # Topic Anchor (Fix #5)
    # =========================

    def get_topic_anchor(self):
        return self.topic_anchor

    def set_topic_anchor(self, text: str):
        self.topic_anchor = text

    def clear_topic_anchor(self):
        self.topic_anchor = None

    # =========================
    # Conversation History
    # =========================

    def add_turn(self, question: str, answer: str):
        self.history.append({
            "question": question,
            "answer": answer
        })

        # keep only last N turns
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # 🔑 update topic anchor AFTER answering
        self.topic_anchor = question

    def get_previous_question(self):
        if not self.history:
            return None
        return self.history[-1]["question"]

    def get_previous_answer(self):
        if not self.history:
            return None
        return self.history[-1]["answer"]

    def get_last_n_questions(self, n: int = 3):
        return [turn["question"] for turn in self.history[-n:]]

    # =========================
    # Session Control
    # =========================

    def clear_session(self):
        self.history.clear()
        self.topic_anchor = None

    # =========================
    # Debug / Optional
    # =========================

    def get_context_summary(self):
        if not self.history:
            return ""

        lines = []
        for i, turn in enumerate(reversed(self.history), 1):
            q = turn["question"]
            a = turn["answer"]
            lines.append(f"Q{i}: {q}")
            if a:
                lines.append(f"A{i}: {a[:100]}...")

        return "\n".join(lines)


# Global singleton
context_manager = ContextManager()

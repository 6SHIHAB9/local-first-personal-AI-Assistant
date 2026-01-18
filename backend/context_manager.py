"""
Context Memory Manager - Handles conversation context and subject tracking
"""
from typing import Optional, Dict, Any
import time


class ConversationContext:
    """Manages conversation context across requests"""
    
    def __init__(self, session_timeout: int = 600):  # 10 minutes default
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = session_timeout
    
    def _cleanup_expired(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired = [
            sid for sid, data in self.sessions.items()
            if current_time - data.get("last_updated", 0) > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]
    
    def get_active_subject(self, session_id: str = "default") -> Optional[str]:
        """Get the active subject for a session"""
        self._cleanup_expired()
        session = self.sessions.get(session_id, {})
        return session.get("active_subject")
    
    def set_active_subject(self, subject: str, session_id: str = "default"):
        """Set the active subject for a session"""
        self._cleanup_expired()
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        
        self.sessions[session_id]["active_subject"] = subject
        self.sessions[session_id]["last_updated"] = time.time()
    
    def get_previous_question(self, session_id: str = "default") -> Optional[str]:
        """Get the previous question for a session"""
        self._cleanup_expired()
        session = self.sessions.get(session_id, {})
        return session.get("previous_question")
    
    def set_previous_question(self, question: str, session_id: str = "default"):
        """Set the previous question for a session"""
        self._cleanup_expired()
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        
        self.sessions[session_id]["previous_question"] = question
        self.sessions[session_id]["last_updated"] = time.time()
    
    def clear_session(self, session_id: str = "default"):
        """Clear active subject but preserve previous question for continuations"""
        if session_id in self.sessions:
            # Preserve previous_question if it exists
            prev_q = self.sessions[session_id].get("previous_question")
            del self.sessions[session_id]
            # Restore previous_question after clearing
            if prev_q:
                self.sessions[session_id] = {
                    "previous_question": prev_q,
                    "last_updated": time.time()
            }
    
    def get_context(self, session_id: str = "default") -> Dict[str, Any]:
        """Get full context for a session"""
        self._cleanup_expired()
        return self.sessions.get(session_id, {})
    
    def update_context(self, updates: Dict[str, Any], session_id: str = "default"):
        """Update context with new data"""
        self._cleanup_expired()
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        
        self.sessions[session_id].update(updates)
        self.sessions[session_id]["last_updated"] = time.time()


# Global instance
context_manager = ConversationContext()
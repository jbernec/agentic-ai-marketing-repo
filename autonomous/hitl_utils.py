"""
HITL (Human-in-the-Loop) utility functions for autonomous agents.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def requires_review(user_query: str) -> bool:
    """Use LLM to detect if user is requesting a destructive operation"""
    from pydantic import BaseModel
    
    class IntentCheck(BaseModel):
        is_destructive: bool
        reasoning: str
    
    model = config.get_model()
    
    check_prompt = f"""Analyze the user's query and determine if they are requesting a destructive data operation.

User Query: {user_query}

Rules:
- Return TRUE if user wants to DELETE, REMOVE, DROP, TRUNCATE, UPDATE, or MODIFY data
- Return FALSE if they're asking to VIEW, SHOW, SELECT, ANALYZE, or RETRIEVE data
- Return FALSE if they're asking HOW TO do something (not actually doing it)
- Return TRUE for requests like: "delete old records", "remove inactive users", "update prices", "drop table"
- Return FALSE for requests like: "show me sales", "what's the revenue", "analyze customer data"

Respond with a structured object: is_destructive (boolean), reasoning (brief explanation)."""
    
    try:
        result = model.with_structured_output(IntentCheck).invoke(check_prompt)
        return result.is_destructive
    except Exception:
        # Fallback: check for destructive keywords in query
        destructive_keywords = ['delete', 'remove', 'drop', 'truncate', 'update', 'modify', 'alter']
        query_lower = user_query.lower()
        # Only trigger if not asking "how to" or "show me how"
        if any(phrase in query_lower for phrase in ['how to', 'how do i', 'show me how']):
            return False
        return any(keyword in query_lower for keyword in destructive_keywords)

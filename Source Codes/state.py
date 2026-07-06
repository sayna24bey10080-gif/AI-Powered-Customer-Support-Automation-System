"""state.py – Shared state schema for the LangGraph support workflow."""

from typing import TypedDict, List, Optional


class SupportState(TypedDict):
    customer_name: str
    query: str
    department: str
    retrieved_context: str      # chunks from RAG
    approval_required: bool
    approved: bool
    memory: List[str]
    response: str               # agent's draft response
    final_response: str         # supervisor-approved final response
    supervisor_feedback: str    # supervisor's reasoning / feedback
    conversation_history: List[dict]   # [{role, content}, ...] for multi-turn

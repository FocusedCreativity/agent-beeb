from typing import Annotated, Optional, Any, Dict
from langgraph.graph import MessagesState
from langmem.short_term import RunningSummary


class ChatbotState(MessagesState):
    """State for Agent Beeb - Personal Grocery Shopping Assistant with LangMem memory management."""
    
    # Running summary for LangMem short-term memory
    summary: Optional[RunningSummary] = None
    
    # Context dictionary for LangMem memory management
    context: Dict[str, Any] = {}
    
    # Thread ID for persistence
    thread_id: Optional[str] = None
    
    # User ID for personalized memory namespacing
    user_id: Optional[str] = None
    
    # Configuration for summarization
    max_messages: int = 6  # Maximum messages before summarization
    max_tokens: int = 256  # Maximum tokens before summarization
    max_summary_tokens: int = 128  # Maximum tokens for summary
    
    # User context for personalization (dietary preferences, budget, household size, etc.)
    user_context: Optional[Dict[str, Any]] = None 
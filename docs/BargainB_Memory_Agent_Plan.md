# BargainB Memory Agent Implementation Plan 🐝

## Overview
BargainB is a bee-themed grocery shopping assistant with long-term memory, conversation summarization, and specialized worker agents. The main agent "Beeb" supervises a hive of specialized worker bees.

## Current File Structure Analysis

### Existing Files (Post-Revert)
```
agentbb-main/
├── my_agent/
│   ├── __init__.py
│   ├── agent.py                    # Main agent entry point
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── database.py             # Database utilities & product search
│   │   ├── nodes.py                # Node functions
│   │   ├── state.py                # State definitions
│   │   └── tools.py                # Tool definitions
│   └── memory_agent/
│       ├── __init__.py
│       ├── agent.py                # Memory agent implementation
│       ├── database_integration.py # Database integration
│       ├── nodes.py                # Memory-specific nodes
│       ├── schemas.py              # Memory schemas
│       ├── simple_persistence.py   # Simple persistence layer
│       ├── state.py                # Memory state definitions
│       ├── supabase_persistence.py # Supabase persistence
│       └── tools.py                # Memory tools
├── test_*.py                       # Various test files
├── langgraph.json                  # LangGraph configuration
└── requirements.txt                # Dependencies
```

### Files to Keep & Modify
- ✅ `my_agent/utils/database.py` - Contains working product search
- ✅ `my_agent/memory_agent/schemas.py` - Update with BargainB memory types
- ✅ `my_agent/memory_agent/supabase_persistence.py` - Fix and enhance
- ✅ `test_memory_agent_with_summarization.py` - Working test reference

### Files to Replace/Restructure
- 🔄 `my_agent/memory_agent/agent.py` - Rebuild with supervisor pattern
- 🔄 `my_agent/memory_agent/nodes.py` - Rebuild with bee-themed agents
- 🔄 `my_agent/memory_agent/tools.py` - Update with handoff tools

## 🐝 Bee-Themed Agent Architecture

### Main Supervisor: **Beeb** 🐝👑
- **Role**: Queen Bee / Main Assistant
- **Responsibilities**: 
  - Always responds to users
  - Manages conversation flow
  - Delegates tasks to worker bees
  - Incorporates results seamlessly
  - Triggers memory updates

### Worker Bees:

#### **Scout Bee** 🐝🔍 (Product Search Agent)
- **Role**: Product Discovery Specialist
- **Responsibilities**:
  - Searches product database
  - Applies user preferences
  - Finds best deals and options
  - NEVER responds directly to users
  - Returns formatted results to Beeb

#### **Memory Bee** 🐝🧠 (Memory Management Agent)
- **Role**: Hive Memory Keeper
- **Responsibilities**:
  - Extracts and updates user memories
  - Manages semantic, episodic, procedural memory
  - Uses Trustcall for memory operations
  - Runs in background

#### **Scribe Bee** 🐝📝 (Conversation Summarizer)
- **Role**: Conversation Historian
- **Responsibilities**:
  - Summarizes long conversations
  - Preserves important context
  - Manages message truncation
  - Stores summaries in Supabase

## 🧠 BargainB Memory System

### Memory Types

#### Semantic Memory - User Profile & Preferences
```python
class SemanticMemory(BaseModel):
    # Food Preferences
    likes: List[str] = Field(default_factory=list)  
    # ["pasta", "avocado", "organic milk"]
    
    dislikes: List[str] = Field(default_factory=list)  
    # ["spicy food", "seafood"]
    
    dietary_restrictions: List[str] = Field(default_factory=list)  
    # ["vegetarian", "gluten-free"]
    
    allergies: List[str] = Field(default_factory=list)  
    # ["peanuts", "shellfish"]
    
    # Shopping Behavior
    budget_sensitivity: Literal["low", "medium", "high"] = "medium"
    preferred_stores: List[str] = Field(default_factory=list)  
    # ["Albert Heijn", "Jumbo"]
    
    shopping_habits: Dict[str, Any] = Field(default_factory=dict)
    # {"prefers_organic": True, "bulk_buyer": False}
```

#### Episodic Memory - Past Interactions
```python
class EpisodicMemory(BaseModel):
    date: str = Field(description="Date of interaction (YYYY-MM-DD)")
    interaction_type: str = Field(description="product_search, meal_planning, etc.")
    user_query: str = Field(description="What user asked for")
    system_response: str = Field(description="What Beeb provided")
    user_feedback: Optional[str] = Field(description="User's reaction")
    products_discussed: List[str] = Field(default_factory=list)
    outcome: str = Field(description="Successful, neutral, or negative")
```

#### Procedural Memory - System Behavior
```python
class ProceduralMemory(BaseModel):
    tone_preferences: str = Field(description="friendly, professional, casual")
    response_style: str = Field(description="detailed, concise, bullet-points")
    interaction_preferences: str = Field(description="User's preferred flow")
    custom_instructions: str = Field(description="Specific behavior rules")
    # Example: "Always suggest budget alternatives, be extra friendly"
```

## 🏗️ Implementation Plan

### Phase 1: Foundation Setup
```
Duration: 1-2 hours
Priority: HIGH

Tasks:
- [ ] Update memory schemas with BargainB types
- [ ] Fix Supabase persistence layer
- [ ] Create handoff tools for delegation
- [ ] Set up basic graph structure
```

### Phase 2: Beeb Supervisor Agent
```
Duration: 2-3 hours
Priority: HIGH

Tasks:
- [ ] Implement main Beeb supervisor
- [ ] Add memory context loading
- [ ] Create delegation logic
- [ ] Add conversation summarization triggers
```

### Phase 3: Worker Bee Agents
```
Duration: 2-3 hours
Priority: HIGH

Tasks:
- [ ] Scout Bee (Product Search)
  - Use existing database.py functions
  - Apply user preferences
  - Format results for Beeb
  
- [ ] Memory Bee (Memory Management)
  - Trustcall integration
  - Background memory updates
  - Memory extraction logic
  
- [ ] Scribe Bee (Summarization)
  - Conversation summarization
  - Message truncation
  - Summary storage
```

### Phase 4: Integration & Testing
```
Duration: 1-2 hours
Priority: MEDIUM

Tasks:
- [ ] Integrate all agents in single graph
- [ ] Test conversation flows
- [ ] Verify memory persistence
- [ ] Test with LangGraph CLI
```

## 🔧 Technical Implementation Details

### Graph Structure
```python
# Single unified graph following supervisor pattern
class BargainBState(MessagesState):
    summary: str = ""
    user_id: Optional[str] = None
    semantic_memory: Optional[Dict[str, Any]] = None
    episodic_memories: List[Dict[str, Any]] = []
    procedural_memory: Optional[Dict[str, Any]] = None

# Nodes
builder = StateGraph(BargainBState)
builder.add_node("beeb_supervisor", beeb_supervisor)
builder.add_node("scout_bee", scout_bee_worker)
builder.add_node("memory_bee", memory_bee_worker)
builder.add_node("scribe_bee", scribe_bee_worker)

# Routing
builder.add_edge(START, "beeb_supervisor")
builder.add_conditional_edges("beeb_supervisor", route_supervisor)
builder.add_edge("scout_bee", "beeb_supervisor")
builder.add_edge("memory_bee", "beeb_supervisor")
builder.add_edge("scribe_bee", END)
```

### Handoff Tools
```python
@tool("call_scout_bee")
def call_scout_bee(
    query: str,
    dietary_preferences: Optional[str] = None,
    budget_preference: Optional[str] = None,
    state: Annotated[MessagesState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Send product search request to Scout Bee"""
    
@tool("call_memory_bee")
def call_memory_bee(
    memory_type: Literal["semantic", "episodic", "procedural"],
    state: Annotated[MessagesState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Request memory update from Memory Bee"""

@tool("call_scribe_bee")
def call_scribe_bee(
    state: Annotated[MessagesState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Request conversation summarization from Scribe Bee"""
```

### Database Integration
```python
# Use existing database.py functions
from my_agent.utils.database import (
    semantic_search,
    BargainBDatabase,
    get_database_connection
)

# Enhance with memory-aware searches
class MemoryAwareDatabase(BargainBDatabase):
    async def personalized_product_search(
        self, 
        query: str, 
        user_preferences: Optional[Dict[str, Any]] = None,
        budget_sensitivity: str = "medium",
        limit: int = 5
    ) -> List[Document]:
        # Apply user preferences to search results
```

## 🗄️ Database Schema

### Supabase Tables
```sql
-- Memory store for long-term memory
CREATE TABLE memory_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(namespace, key)
);

-- Conversation summaries
CREATE TABLE conversation_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    summary_version INTEGER DEFAULT 1,
    message_count_at_summary INTEGER NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Message truncation log
CREATE TABLE message_truncation_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    messages_before_truncation INTEGER NOT NULL,
    messages_after_truncation INTEGER NOT NULL,
    messages_removed INTEGER NOT NULL,
    summary_tokens_used INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## 🧪 Testing Strategy

### Test Files to Update
- `test_memory_agent_with_summarization.py` - Main test file
- `test_beeb_delegation_agent.py` - Delegation testing
- `test_memory_agent.py` - Memory functionality

### Test Scenarios
1. **Memory Management**
   - User preferences extraction
   - Episodic memory creation
   - Procedural memory updates

2. **Product Search Delegation**
   - Beeb delegates to Scout Bee
   - Scout Bee applies preferences
   - Results returned to Beeb

3. **Conversation Summarization**
   - Trigger at 6+ messages
   - Summary creation and storage
   - Context preservation

4. **End-to-End Flow**
   - Complete conversation cycle
   - Memory persistence across sessions
   - LangGraph CLI compatibility

## 🚀 Deployment Configuration

### LangGraph Configuration
```json
{
    "dependencies": [
        "langchain",
        "langchain_openai",
        "langchain_core",
        "langchain_community",
        "langgraph",
        "trustcall",
        "asyncpg",
        "./my_agent"
    ],
    "graphs": {
        "bargainb_memory_agent": "./my_agent/memory_agent/agent.py:bargainb_memory_agent"
    },
    "env": "./.env"
}
```

### Environment Variables
```
OPENAI_API_KEY=your_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
DB_PASSWORD=AfdalX@20202
LANGSMITH_API_KEY=your_langsmith_key
```

## 📝 Next Steps

1. **Confirm Plan** - Review and approve this plan
2. **Start Implementation** - Begin with Phase 1 (Foundation)
3. **Iterative Development** - Build and test each phase
4. **Integration Testing** - Ensure all components work together
5. **Deployment Testing** - Verify LangGraph CLI compatibility

## 🐝 Bee Personality Guidelines

### Beeb (Main Assistant)
- Friendly and helpful
- Uses occasional bee/honey metaphors
- Budget-conscious
- Knowledgeable about Dutch grocery stores
- Warm and enthusiastic

### Scout Bee (Internal)
- Efficient and thorough
- Focuses on finding best deals
- Applies user preferences precisely
- Never speaks directly to users

### Memory Bee (Internal)
- Meticulous and organized
- Quietly manages user memories
- Learns from every interaction
- Background operations only

### Scribe Bee (Internal)
- Preserves important context
- Summarizes conversations effectively
- Maintains conversation history
- Background operations only

---

*This plan will evolve as we implement and test each component. The goal is a cohesive, bee-themed grocery assistant with sophisticated memory management and delegation capabilities.* 
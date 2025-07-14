import os
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.func import entrypoint

# LangMem imports for proper memory management
from langmem import ReflectionExecutor, create_memory_store_manager

from my_agent.utils.state import ChatbotState
from my_agent.utils.nodes import call_model, should_continue
from my_agent.utils.tools import (
    search_grocery_products,
    create_grocery_list,
    plan_meal_with_products,
    suggest_weekly_meal_plan
)

# Load environment variables
import pathlib
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize LLM
llm = init_chat_model("openai:gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# Create memory manager for extracting memories from conversations
# This will use the platform's store automatically when deployed
memory_manager = create_memory_store_manager(
    "openai:gpt-4o",
    # Store memories in the "memories" namespace with user_id
    namespace=("memories", "{user_id}"),
    # Extract general memories about user preferences and context
    instructions="Extract user preferences, dietary restrictions, shopping habits, favorite products, budget constraints, and any other grocery shopping related information.",
    enable_inserts=True,
    enable_deletes=True,
)

# ReflectionExecutor will automatically use the platform's store when deployed
memory_executor = ReflectionExecutor(memory_manager)


class AgentBeeb:
    """
    Agent Beeb - Your personal grocery shopping assistant with LangMem memory management.
    
    Agent Beeb helps you with:
    - Creating and managing shopping lists
    - Meal planning and recipe suggestions
    - Dietary preferences and restrictions
    - Budget tracking and smart shopping tips
    - Product recommendations and alternatives
    - Store navigation and shopping optimization
    """
    
    def __init__(self):
        """Initialize Agent Beeb for LangGraph Platform deployment."""
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with LangMem integration."""
        
        # Define the graph
        workflow = StateGraph(ChatbotState)
        
        # Add main conversation node
        workflow.add_node("conversation", call_model)
        
        # Add tool execution node
        workflow.add_node("tools", self._execute_tools)
        
        # Set the entry point
        workflow.add_edge(START, "conversation")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "conversation", 
            should_continue,
            {
                "tools": "tools",
                "END": END
            }
        )
        
        # Add edge from tools back to conversation
        workflow.add_edge("tools", "conversation")
        
        # Compile the graph - let LangGraph Platform handle persistence
        return workflow.compile()
    
    def _execute_tools(self, state: ChatbotState) -> Dict[str, Any]:
        """Execute tools based on the tool calls in the last message."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Validate that we have tool calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"messages": []}
        
        tool_messages = []
        
        for tool_call in last_message.tool_calls:
            try:
                # Extract tool call information safely
                tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                tool_args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                
                # Ensure we have all required information
                if not tool_id or not tool_name:
                    error_message = ToolMessage(
                        content="Error: Invalid tool call format",
                        tool_call_id=tool_id or "unknown"
                    )
                    tool_messages.append(error_message)
                    continue
                
                # Execute the appropriate tool
                if tool_name == "search_grocery_products":
                    result = search_grocery_products.invoke(tool_args)
                    
                    # Format the results for better agent understanding
                    if isinstance(result, list) and len(result) > 0:
                        formatted_results = []
                        for item in result:
                            product_info = {
                                "name": item.get("name", "Unknown"),
                                "brand": item.get("brand", "Unknown"),
                                "price": item.get("best_price", "N/A"),
                                "store": item.get("best_store", "Unknown"),
                                "description": item.get("description", "No description available")
                            }
                            formatted_results.append(product_info)
                        
                        # Create a readable summary
                        summary = f"Found {len(formatted_results)} products:\n"
                        for i, product in enumerate(formatted_results[:10], 1):  # Show top 10
                            summary += f"{i}. {product['name']} ({product['brand']}) - €{product['price']} at {product['store']}\n"
                        
                        content = summary
                    else:
                        content = "No products found matching your criteria."
                
                elif tool_name == "create_grocery_list":
                    result = create_grocery_list.invoke(tool_args)
                    content = f"Created grocery list: {result}"
                
                elif tool_name == "plan_meal_with_products":
                    result = plan_meal_with_products.invoke(tool_args)
                    content = f"Meal plan created: {result}"
                
                elif tool_name == "suggest_weekly_meal_plan":
                    result = suggest_weekly_meal_plan.invoke(tool_args)
                    content = f"Weekly meal plan: {result}"
                
                else:
                    content = f"Unknown tool: {tool_name}"
                
                # Create tool message
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_id
                )
                tool_messages.append(tool_message)
                    
            except Exception as e:
                # Always create a ToolMessage even if there's an error
                tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", "unknown")
                error_message = ToolMessage(
                    content=f"Error executing tool: {str(e)}",
                    tool_call_id=tool_id
                )
                tool_messages.append(error_message)
        
        return {"messages": tool_messages}
    
    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        max_tokens_before_summary: int = 3000,
        max_summary_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Chat with Agent Beeb about grocery shopping.
        
        Args:
            message: User's message
            thread_id: Optional thread ID for conversation persistence
            user_id: Optional user ID for personalized memory
            user_context: Optional user context (dietary preferences, budget, etc.)
            max_tokens: Maximum tokens before summarization
            max_summary_tokens: Maximum tokens for summary
            
        Returns:
            Dictionary containing Agent Beeb's response and metadata
        """
        
        # Generate thread_id if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        # Use default user_id if not provided
        if not user_id:
            user_id = "default_user"
        
        # Create configuration for the graph
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        # Prepare the input message
        input_message = HumanMessage(content=message)
        
        # Add grocery shopping context
        grocery_context = {
            "assistant_name": "Agent Beeb",
            "role": "personal grocery shopping assistant",
            "capabilities": [
                "shopping list management",
                "meal planning",
                "recipe suggestions", 
                "dietary preferences tracking",
                "budget optimization",
                "product recommendations",
                "store navigation tips",
                "product search with 40,000+ items",
                "price comparison across stores",
                "nutrition information lookup",
                "ingredient analysis",
                "dietary restriction filtering"
            ],
            **(user_context or {})
        }
        
        # Create initial state
        initial_state = {
            "messages": [input_message],
            "thread_id": thread_id,
            "user_id": user_id,
            "user_context": grocery_context,
            "max_tokens": max_tokens,
            "max_tokens_before_summary": max_tokens_before_summary,
            "max_summary_tokens": max_summary_tokens,
            "context": {}
        }
        
        # Invoke the graph
        try:
            result = self.graph.invoke(initial_state, config)
            
            # Extract the AI response
            ai_messages = [msg for msg in result["messages"] if hasattr(msg, "content")]
            ai_response = ai_messages[-1].content if ai_messages else "I'm sorry, I couldn't generate a response."
            
            # Process memories in background using ReflectionExecutor
            # This will automatically use the platform's store
            try:
                # Format conversation for memory processing (OpenAI message format)
                conversation_for_memory = {
                    "messages": [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": ai_response}
                    ]
                }
                
                # Submit for background processing with 30 second delay
                # In production, you might want a longer delay (30-60 minutes)
                memory_executor.submit(
                    conversation_for_memory,
                    after_seconds=30.0,  # Process after 30 seconds of inactivity
                    config=config
                )
            except Exception as e:
                # Silently handle memory processing errors to avoid disrupting user experience
                pass
            
            return {
                "response": ai_response,
                "thread_id": thread_id,
                "user_id": user_id,
                "summary": result.get("summary"),
                "context": result.get("context", {}),
                "success": True
            }
            
        except Exception as e:
            return {
                "response": "I'm sorry, I encountered an error processing your message. Let me try to help you with your grocery shopping needs anyway!",
                "thread_id": thread_id,
                "user_id": user_id,
                "error": str(e),
                "success": False
            }
    
    def get_conversation_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a conversation with Agent Beeb.
        
        Args:
            thread_id: Thread ID for the conversation
            
        Returns:
            Current conversation state or None if not found
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.graph.get_state(config)
            return state.values if state else None
        except Exception:
            return None
    
    def list_conversations(self, thread_id: str, limit: int = 10) -> list:
        """
        List recent conversations for a specific thread.
        
        Args:
            thread_id: Thread ID for the conversation
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation history
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get conversation history from checkpointer
            history = []
            state_history = self.graph.get_state_history(config, limit=limit)
            
            for state in state_history:
                if state.values.get("messages"):
                    history.append({
                        "timestamp": state.created_at,
                        "messages": state.values["messages"],
                        "summary": state.values.get("summary")
                    })
            
            return history
        except Exception:
            return []


# Create the main Agent Beeb graph for deployment
def create_agent_beeb() -> AgentBeeb:
    """Create an Agent Beeb instance."""
    return AgentBeeb()

# Create the compiled graph that LangGraph expects for deployment
agent_beeb = create_agent_beeb()
graph = agent_beeb.graph

# This is the main graph that should be deployed
agent = graph 
import os
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

# LangMem imports for proper memory management
from langmem import ReflectionExecutor, create_memory_store_manager

from my_agent.utils.state import ChatbotState
from my_agent.utils.nodes import call_model, should_continue
from my_agent.utils.tools import search_grocery_products

# Load environment variables
import pathlib
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize LangMem-compatible store with embedding support
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Initialize checkpointer for conversation persistence (local development only)
checkpointer = MemorySaver()

# Initialize LLM
llm = init_chat_model("openai:gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# Create memory manager for extracting memories from conversations
memory_manager = create_memory_store_manager(
    "openai:gpt-4o",
    # Store memories in the "memories" namespace with user_id
    namespace=("memories", "{user_id}"),
    # Extract general memories about user preferences and context
    instructions="Extract user preferences, dietary restrictions, shopping habits, favorite products, budget constraints, and any other grocery shopping related information.",
    enable_inserts=True,
    enable_deletes=True,
)

# Wrap memory manager with ReflectionExecutor for background processing
memory_executor = ReflectionExecutor(memory_manager, store=store)


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
    
    def __init__(self, use_platform_persistence: bool = False):
        """
        Initialize Agent Beeb.
        
        Args:
            use_platform_persistence: If True, don't use custom checkpointer/store
                                     (for LangGraph Platform deployment)
        """
        self.use_platform_persistence = use_platform_persistence
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
        
        # Compile the graph - with or without custom persistence
        if self.use_platform_persistence:
            # For LangGraph Platform - let platform handle persistence
            return workflow.compile()
        else:
            # For local development - use custom checkpointer and store
            return workflow.compile(
                checkpointer=checkpointer,
                store=store
            )
    
    def _execute_tools(self, state: ChatbotState) -> Dict[str, Any]:
        """Execute tools based on the tool calls in the last message."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"messages": []}
        
        tool_messages = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            
            if tool_name == "search_grocery_products":
                try:
                    # Execute the tool with the arguments from the tool call
                    result = search_grocery_products.invoke(tool_call["args"])
                    
                    # Format the results for better agent understanding
                    if isinstance(result, list) and len(result) > 0:
                        formatted_results = []
                        for i, product in enumerate(result[:10]):  # Limit to top 10 results
                            product_info = f"{i+1}. {product.get('title', 'Unknown Product')}"
                            if product.get('brand'):
                                product_info += f" by {product.get('brand')}"
                            if product.get('best_price'):
                                product_info += f" - €{product.get('best_price')}"
                            if product.get('best_store'):
                                product_info += f" at {product.get('best_store')}"
                            if product.get('suggestion'):
                                product_info += f" ({product.get('suggestion')})"
                            formatted_results.append(product_info)
                        
                        tool_content = "Product search results:\n" + "\n".join(formatted_results)
                    else:
                        tool_content = "No products found matching your search criteria."
                    
                    # Create a proper tool message with the formatted result
                    tool_message = ToolMessage(
                        content=tool_content,
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(tool_message)
                except Exception as e:
                    # Handle tool execution errors
                    error_message = ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(error_message)
                    
            elif tool_name == "create_grocery_list":
                try:
                    from my_agent.utils.tools import create_grocery_list
                    result = create_grocery_list.invoke(tool_call["args"])
                    
                    # Format grocery list results
                    if isinstance(result, dict) and not result.get("error"):
                        formatted_content = f"Grocery List:\n\n"
                        for item in result.get("items", []):
                            if item.get("product") != "Not found":
                                formatted_content += f"• {item.get('searched_for')}: {item.get('product')} ({item.get('brand', 'Unknown brand')}) - €{item.get('price', 0)} at {item.get('store', 'Unknown store')}\n"
                            else:
                                formatted_content += f"• {item.get('searched_for')}: Not found\n"
                        
                        formatted_content += f"\nTotal Cost: €{result.get('total_cost', 0):.2f}\n"
                        
                        if result.get("stores"):
                            formatted_content += f"Stores to visit: {', '.join(result['stores'].keys())}\n"
                        
                        if result.get("suggestions"):
                            formatted_content += f"\nSuggestions: {'; '.join(result['suggestions'])}"
                    else:
                        formatted_content = str(result)
                    
                    tool_message = ToolMessage(
                        content=formatted_content,
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(tool_message)
                except Exception as e:
                    error_message = ToolMessage(
                        content=f"Error creating grocery list: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(error_message)
                    
            elif tool_name == "plan_meal_with_products":
                try:
                    from my_agent.utils.tools import plan_meal_with_products
                    result = plan_meal_with_products.invoke(tool_call["args"])
                    
                    # Format meal plan results
                    if isinstance(result, dict) and not result.get("error"):
                        formatted_content = f"Meal Plan: {result.get('meal_name')}\n"
                        formatted_content += f"Servings: {result.get('servings')}\n\n"
                        formatted_content += "Ingredients:\n"
                        
                        for ingredient in result.get("ingredients", []):
                            if ingredient.get("product_found") != "Not available":
                                formatted_content += f"• {ingredient.get('ingredient')}: {ingredient.get('product_found')} ({ingredient.get('brand', 'Unknown brand')}) - €{ingredient.get('price', 0)} at {ingredient.get('store', 'Unknown store')}\n"
                            else:
                                formatted_content += f"• {ingredient.get('ingredient')}: Not available\n"
                        
                        formatted_content += f"\nTotal Cost: €{result.get('total_cost', 0):.2f}\n"
                        formatted_content += f"Cost per serving: €{result.get('cost_per_serving', 0):.2f}\n"
                        
                        if result.get("preparation_notes"):
                            formatted_content += f"\nNotes: {'; '.join(result['preparation_notes'])}"
                    else:
                        formatted_content = str(result)
                    
                    tool_message = ToolMessage(
                        content=formatted_content,
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(tool_message)
                except Exception as e:
                    error_message = ToolMessage(
                        content=f"Error planning meal: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(error_message)
                    
            elif tool_name == "suggest_weekly_meal_plan":
                try:
                    from my_agent.utils.tools import suggest_weekly_meal_plan
                    result = suggest_weekly_meal_plan.invoke(tool_call["args"])
                    
                    # Format weekly meal plan results
                    if isinstance(result, dict) and not result.get("error"):
                        formatted_content = f"Weekly Meal Plan ({len(result.get('meals', []))} meals):\n\n"
                        
                        for i, meal in enumerate(result.get("meals", []), 1):
                            formatted_content += f"{i}. {meal.get('meal_name')} - €{meal.get('total_cost', 0):.2f}\n"
                        
                        formatted_content += f"\nTotal Weekly Cost: €{result.get('total_weekly_cost', 0):.2f}\n"
                        
                        if result.get("budget_analysis"):
                            budget = result["budget_analysis"]
                            formatted_content += f"Budget Status: {budget.get('under_over')} budget by €{budget.get('difference', 0):.2f}\n"
                        
                        formatted_content += f"\nConsolidated Shopping List:\n"
                        for ingredient, info in result.get("consolidated_shopping_list", {}).items():
                            if info.get("product_found") != "Not available":
                                formatted_content += f"• {ingredient}: {info.get('product_found')} - €{info.get('price', 0)} at {info.get('store', 'Unknown store')}\n"
                    else:
                        formatted_content = str(result)
                    
                    tool_message = ToolMessage(
                        content=formatted_content,
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(tool_message)
                except Exception as e:
                    error_message = ToolMessage(
                        content=f"Error creating weekly meal plan: {str(e)}",
                        tool_call_id=tool_call["id"]
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
        
        # Search for relevant existing memories
        existing_memories = []
        try:
            if not self.use_platform_persistence:
                # Search for memories related to this user and query
                memory_results = store.search(
                    ("memories", user_id),
                    query=message,
                    limit=5  # Get top 5 relevant memories
                )
                existing_memories = [item.value.get("content", "") for item in memory_results]
        except Exception as e:
            print(f"Error searching memories: {e}")
        
        # Prepare the input message
        input_message = HumanMessage(content=message)
        
        # Add grocery shopping context with existing memories
        grocery_context = {
            "assistant_name": "Agent Beeb",
            "role": "personal grocery shopping assistant",
            "existing_memories": existing_memories,
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
            if not self.use_platform_persistence:
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
                    print(f"Error submitting memory processing: {e}")
            
            return {
                "response": ai_response,
                "thread_id": thread_id,
                "user_id": user_id,
                "summary": result.get("summary"),
                "context": result.get("context", {}),
                "memories_found": len(existing_memories),
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

    def get_user_memories(self, user_id: str, limit: int = 10) -> list:
        """
        Get stored memories for a specific user.
        
        Args:
            user_id: User ID to search memories for
            limit: Maximum number of memories to return
            
        Returns:
            List of stored memories for the user
        """
        if self.use_platform_persistence:
            return []
        
        try:
            memory_results = store.search(
                ("memories", user_id),
                limit=limit
            )
            
            memories = []
            for item in memory_results:
                memories.append({
                    "id": item.key,
                    "content": item.value.get("content", ""),
                    "created_at": item.created_at,
                    "updated_at": item.updated_at
                })
            
            return memories
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

    def clear_user_memories(self, user_id: str) -> bool:
        """
        Clear all stored memories for a specific user.
        
        Args:
            user_id: User ID to clear memories for
            
        Returns:
            True if successful, False otherwise
        """
        if self.use_platform_persistence:
            return False
        
        try:
            # Get all memories for the user
            memory_results = store.search(("memories", user_id))
            
            # Delete each memory
            for item in memory_results:
                store.delete(("memories", user_id), item.key)
            
            return True
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False
            
    def search_memories(self, user_id: str, query: str, limit: int = 5) -> list:
        """
        Search memories for a specific user using semantic search.
        
        Args:
            user_id: User ID to search memories for
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of relevant memories
        """
        if self.use_platform_persistence:
            return []
        
        try:
            memory_results = store.search(
                ("memories", user_id),
                query=query,
                limit=limit
            )
            
            memories = []
            for item in memory_results:
                memories.append({
                    "id": item.key,
                    "content": item.value.get("content", ""),
                    "score": item.score,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at
                })
            
            return memories
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []


# Create entrypoint for LangGraph Platform compatibility
@entrypoint(store=store)
def chat_entrypoint(message: str, user_id: str = "default_user") -> str:
    """
    Entrypoint function for LangGraph Platform - Agent Beeb.
    
    Args:
        message: User's message about grocery shopping
        user_id: User identifier for memory namespacing
        
    Returns:
        Agent Beeb's response
    """
    
    # Use the main conversation model with grocery shopping context
    system_message = """You are Agent Beeb, a helpful personal grocery shopping assistant with access to a comprehensive product database. 
    
    You can search through 40,000+ grocery products with pricing, nutrition info, ingredients, and availability across multiple stores.
    
    You help users with shopping lists, meal planning, recipes, dietary preferences, 
    budget optimization, product recommendations, price comparisons, and smart shopping tips. 
    
    Be friendly, knowledgeable, and focus on practical grocery shopping advice."""
    
    input_message = HumanMessage(content=message)
    response = llm.invoke([
        HumanMessage(content=system_message),
        input_message
    ])
    
    return response.content


# Create the compiled graph that LangGraph expects (for platform deployment)
def create_agent():
    """Create and return the compiled LangGraph agent for platform deployment."""
    agent_beeb = AgentBeeb(use_platform_persistence=True)
    return agent_beeb.graph


# Global compiled graph instance for LangGraph Platform
agent = create_agent()

# Global AgentBeeb instance for local development
agent_beeb = AgentBeeb(use_platform_persistence=False)


def create_agent_beeb(use_platform_persistence: bool = False) -> AgentBeeb:
    """
    Create and return a new Agent Beeb instance.
    
    Args:
        use_platform_persistence: If True, don't use custom checkpointer/store
    """
    return AgentBeeb(use_platform_persistence=use_platform_persistence)


def process_message(
    message: str,
    thread_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to process a message with Agent Beeb.
    
    Args:
        message: User's message
        thread_id: Optional thread ID (will generate if not provided)
        **kwargs: Additional arguments for the chat method
        
    Returns:
        Dictionary containing Agent Beeb's response and metadata
    """
    return agent_beeb.chat(message, thread_id, **kwargs)


# Example usage function
def example_conversation():
    """
    Example conversation showing Agent Beeb's capabilities with memory management.
    """
    print("=== Agent Beeb - Grocery Shopping Assistant with Memory ===")
    print("Testing conversation flow with memory management...\n")
    
    # Create agent instance
    agent = create_agent_beeb()
    
    # Test user ID
    test_user = "test_user_123"
    
    # Example conversation with memory building
    conversations = [
        "Hi! I'm vegetarian and I'm trying to eat more protein. Can you help me find some good protein sources?",
        "I have a budget of about €30 per week for groceries. What vegetarian proteins would be best for that budget?",
        "Can you help me plan a meal with those protein sources for tonight?",
        "What about planning meals for the whole week with my dietary preferences?"
    ]
    
    thread_id = None
    
    for i, message in enumerate(conversations, 1):
        print(f"User Message {i}: {message}")
        
        # Get response from Agent Beeb
        response = agent.chat(
            message=message,
            thread_id=thread_id,
            user_id=test_user
        )
        
        # Use the same thread for continuation
        if not thread_id:
            thread_id = response["thread_id"]
        
        print(f"Agent Beeb: {response['response']}")
        print(f"Memories found: {response.get('memories_found', 0)}")
        print("-" * 50)
        
        # Add a small delay to simulate natural conversation
        import time
        time.sleep(1)
    
    # Give time for background memory processing
    print("\nWaiting for background memory processing...")
    time.sleep(35)  # Wait for memory processing (30 second delay + buffer)
    
    # Show extracted memories
    print("\n=== Extracted Memories ===")
    memories = agent.get_user_memories(test_user)
    if memories:
        for memory in memories:
            print(f"- {memory['content']}")
    else:
        print("No memories extracted yet (may still be processing)")
    
    # Test memory search
    print("\n=== Memory Search Test ===")
    search_results = agent.search_memories(test_user, "vegetarian protein budget")
    if search_results:
        print("Relevant memories found:")
        for result in search_results:
            print(f"- {result['content']} (score: {result.get('score', 'N/A')})")
    else:
        print("No relevant memories found")
    
    print("\n=== Conversation Complete ===")
    print("Agent Beeb has learned about your preferences and will remember them for future conversations!")
    
    return agent, test_user, thread_id


if __name__ == "__main__":
    # Run example conversation
    example_conversation() 
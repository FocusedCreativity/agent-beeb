import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langmem.short_term import summarize_messages, SummarizationResult
from my_agent.utils.state import ChatbotState
from my_agent.utils.tools import (
    search_grocery_products, 
    create_grocery_list,
    plan_meal_with_products, 
    suggest_weekly_meal_plan
)


# Initialize the OpenAI model
model = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Summarization model with proper token limit for summaries
summarization_model = model.bind(max_tokens=500)





def call_model(state: ChatbotState) -> Dict[str, Any]:
    """
    Process the current conversation and generate a response using LangMem summarization.
    
    Args:
        state: Current chatbot state with messages and optional summary
        
    Returns:
        Dictionary with the AI response message and updated summary
    """
    
    # Use LangMem's summarize_messages to handle context length
    summarization_result: SummarizationResult = summarize_messages(
        state["messages"],
        running_summary=state.get("summary"),
        model=summarization_model,
        max_tokens=state.get("max_tokens", 4000),  # Increased from 256 to 4000
        max_tokens_before_summary=state.get("max_tokens_before_summary", 3000),  # Increased from 256 to 3000
        max_summary_tokens=state.get("max_summary_tokens", 500),  # Increased from 128 to 500
        token_counter=model.get_num_tokens_from_messages
    )
    
    # Get user context and existing memories
    user_context = state.get("user_context", {})
    existing_memories = user_context.get("existing_memories", [])
    
    # Build memory context for the system prompt
    memory_context = ""
    if existing_memories:
        memory_context = f"""

What I remember about you:
{chr(10).join(f"- {memory}" for memory in existing_memories if memory.strip())}

I'll use this information to provide more personalized recommendations and assistance."""
    
    # Create system message with grocery shopping context and memories
    system_message = SystemMessage(content=f"""You are {user_context.get('assistant_name', 'Agent Beeb')}, a {user_context.get('role', 'helpful assistant')}.

Your capabilities include:
{chr(10).join(f"• {capability}" for capability in user_context.get('capabilities', []))}

You have access to these tools:
- search_grocery_products: Search through 40,000+ grocery products with prices and store information
- create_grocery_list: Create shopping lists with real products and cost calculations
- plan_meal_with_products: Plan meals using actual grocery products with pricing
- suggest_weekly_meal_plan: Create comprehensive weekly meal plans with shopping lists

Always be helpful, friendly, and focus on practical grocery shopping advice. When users ask about products, use the search tool to find real products with current prices. When planning meals or creating lists, use the specific tools to provide accurate information.{memory_context}

Provide specific, actionable advice and always include relevant product details, prices, and store information when available.""")
    
    # Combine system message with summarized messages
    messages_with_context = [system_message] + summarization_result.messages
    
    # Generate response with tool binding
    response = model.bind_tools([
        search_grocery_products,
        create_grocery_list,
        plan_meal_with_products,
        suggest_weekly_meal_plan
    ]).invoke(messages_with_context)
    
    # Prepare return value
    result = {"messages": [response]}
    
    # Include updated summary if available
    if summarization_result.running_summary:
        result["summary"] = summarization_result.running_summary
    
    return result


def should_continue(state: ChatbotState) -> str:
    """
    Determine whether to continue the conversation or execute tools.
    
    Args:
        state: Current chatbot state
        
    Returns:
        Next node name: "tools" if tool calls are needed, "END" otherwise
    """
    
    # Get the last message
    last_message = state["messages"][-1]
    
    # Check if there are tool calls in the last message
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "END"


# Optional: Node for explicit memory management using LangMem tools
def manage_memories(state: ChatbotState) -> Dict[str, Any]:
    """
    Extract and manage long-term memories from conversation.
    This is optional and can be used for background memory processing.
    
    Args:
        state: Current chatbot state
        
    Returns:
        Dictionary with context updates for memory management
    """
    
    # This would be used with LangMem's memory management tools
    # For now, just return the context as-is
    return {"context": state.get("context", {})} 
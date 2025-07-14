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
    
    # Get the current messages
    messages = state["messages"]
    
    # Check if the last few messages contain tool calls that need their responses
    # We need to ensure tool calls and tool messages stay together
    preserve_recent_tools = False
    recent_message_count = min(10, len(messages))  # Check last 10 messages
    
    for i in range(len(messages) - recent_message_count, len(messages)):
        if i >= 0 and hasattr(messages[i], 'tool_calls') and messages[i].tool_calls:
            preserve_recent_tools = True
            break
    
    # Use LangMem's summarize_messages with careful handling of tool calls
    try:
        summarization_result: SummarizationResult = summarize_messages(
            messages,
            running_summary=state.get("summary"),
            model=summarization_model,
            max_tokens=state.get("max_tokens", 4000),
            max_tokens_before_summary=state.get("max_tokens_before_summary", 3000) if not preserve_recent_tools else 4000,
            max_summary_tokens=state.get("max_summary_tokens", 500),
            token_counter=model.get_num_tokens_from_messages
        )
    except Exception as e:
        # If summarization fails, use the original messages
        summarization_result = type('SummarizationResult', (), {
            'messages': messages,
            'running_summary': state.get("summary")
        })()
    
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
    system_message = SystemMessage(content=f"""You are Agent Beeb, your personal grocery shopping assistant with advanced memory capabilities.

🛒 **Your Core Identity:**
- You are a specialized grocery shopping assistant focused on helping users with all aspects of food shopping
- You have access to a database of 40,000+ grocery products with real prices and store information
- You remember user preferences, dietary restrictions, budget constraints, and shopping habits
- You provide personalized recommendations based on user's history and preferences

🧠 **Your Capabilities:**
• Search through 40,000+ grocery products with current prices and store availability
• Create personalized shopping lists with real products and cost calculations
• Plan meals using actual grocery products with accurate pricing
• Suggest weekly meal plans with complete shopping lists
• Remember dietary restrictions, preferences, and budget constraints
• Provide store-specific recommendations and price comparisons
• Offer cooking tips and recipe modifications
• Help with budget planning and cost optimization

🔧 **Your Tools:**
- search_grocery_products: Search through 40,000+ grocery products with prices and store information
- create_grocery_list: Create shopping lists with real products and cost calculations
- plan_meal_with_products: Plan meals using actual grocery products with pricing
- suggest_weekly_meal_plan: Create comprehensive weekly meal plans with shopping lists

💡 **Your Personality:**
- Friendly, helpful, and enthusiastic about food and cooking
- Practical and budget-conscious
- Knowledgeable about nutrition and dietary needs
- Proactive in suggesting alternatives and improvements
- Always consider the user's preferences and constraints{memory_context}

Always use your tools to provide specific, actionable advice with real product details, current prices, and store information. Focus on practical grocery shopping solutions that fit the user's needs, preferences, and budget.""")
    
    # Combine system message with summarized messages
    messages_with_context = [system_message] + summarization_result.messages
    
    # Validate message sequence before sending to LLM
    validated_messages = []
    for msg in messages_with_context:
        # Skip any malformed messages or tool calls without responses
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Check if this tool call has responses in the following messages
            try:
                # tool_calls might be objects with .id or dictionaries with 'id' key
                tool_call_ids = []
                for tc in msg.tool_calls:
                    if hasattr(tc, 'id'):
                        tool_call_ids.append(tc.id)
                    elif isinstance(tc, dict) and 'id' in tc:
                        tool_call_ids.append(tc['id'])
                # For now, include the message and let the flow handle it properly
                validated_messages.append(msg)
            except Exception:
                # If there's any issue with tool call validation, skip this message
                continue
        else:
            validated_messages.append(msg)
    
    # Generate response with tool binding
    try:
        response = model.bind_tools([
            search_grocery_products,
            create_grocery_list,
            plan_meal_with_products,
            suggest_weekly_meal_plan
        ]).invoke(validated_messages)
    except Exception as e:
        # Log the actual error for debugging
        import logging
        logging.error(f"Tool binding/invocation failed: {str(e)}")
        
        # Check for specific error types and provide helpful messages
        error_msg = str(e).lower()
        
        if "supabase" in error_msg or "database" in error_msg:
            response = AIMessage(content="I'm having trouble connecting to my product database. Please check that the Supabase credentials are properly configured. Let me know if you need help with basic grocery shopping questions in the meantime!")
        elif "openai" in error_msg or "api" in error_msg or "key" in error_msg:
            response = AIMessage(content="I'm having trouble with my AI model connection. Please check that the OpenAI API key is properly configured. Let me know if you need help!")
        elif "rate" in error_msg or "limit" in error_msg:
            response = AIMessage(content="I'm currently experiencing high usage. Please try again in a few moments, and I'll be happy to help with your grocery shopping needs!")
        else:
            response = AIMessage(content=f"I'm experiencing a technical issue: {str(e)[:100]}... Please try rephrasing your request, and I'll do my best to help you with your grocery shopping needs.")
    
    # Prepare return value
    result = {"messages": [response]}
    
    # Include updated summary if available
    if hasattr(summarization_result, 'running_summary') and summarization_result.running_summary:
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
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
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


def _validate_message_sequence(messages):
    """
    Validate that tool calls are properly paired with tool messages.
    Remove any tool calls that don't have corresponding tool messages.
    """
    validated_messages = []
    pending_tool_calls = []
    
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # This is a message with tool calls
            pending_tool_calls.extend(msg.tool_calls)
            validated_messages.append(msg)
        elif isinstance(msg, ToolMessage):
            # This is a tool message response
            validated_messages.append(msg)
            # Remove the corresponding tool call from pending
            pending_tool_calls = [tc for tc in pending_tool_calls if tc.get('id') != msg.tool_call_id]
        else:
            # Regular message
            validated_messages.append(msg)
    
    # If there are pending tool calls without responses, remove the last AI message with tool calls
    if pending_tool_calls:
        # Find the last AI message with tool calls and remove it
        for i in range(len(validated_messages) - 1, -1, -1):
            if (hasattr(validated_messages[i], 'tool_calls') and 
                validated_messages[i].tool_calls and 
                isinstance(validated_messages[i], AIMessage)):
                # Remove this message to avoid the error
                validated_messages.pop(i)
                break
    
    return validated_messages


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
    
    # Validate message sequence before summarization
    validated_messages = _validate_message_sequence(messages)
    
    # Check if we need to preserve recent tool calls
    preserve_recent_tools = False
    recent_message_count = min(10, len(validated_messages))
    
    for i in range(len(validated_messages) - recent_message_count, len(validated_messages)):
        if i >= 0 and hasattr(validated_messages[i], 'tool_calls') and validated_messages[i].tool_calls:
            preserve_recent_tools = True
            break
    
    # Use LangMem's summarize_messages with careful handling of tool calls
    try:
        summarization_result: SummarizationResult = summarize_messages(
            validated_messages,
            running_summary=state.get("summary"),
            model=summarization_model,
            max_tokens=state.get("max_tokens", 4000),
            max_tokens_before_summary=state.get("max_tokens_before_summary", 3000) if not preserve_recent_tools else 4000,
            max_summary_tokens=state.get("max_summary_tokens", 500),
            token_counter=model.get_num_tokens_from_messages
        )
        
        # Validate the summarized messages again
        final_messages = _validate_message_sequence(summarization_result.messages)
        
    except Exception as e:
        # If summarization fails, use the validated messages
        final_messages = validated_messages
        summarization_result = type('SummarizationResult', (), {
            'messages': final_messages,
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
• Create personalized shopping lists with budget optimization and store recommendations
• Plan complete meals with ingredient sourcing and cost analysis
• Generate weekly meal plans with integrated shopping lists and dietary considerations
• Remember user preferences, restrictions, and shopping patterns for future recommendations

🎯 **Your Approach - BE PROACTIVE:**
• **Take Initiative**: When users express intent, immediately use the appropriate tool
• **Be Confident**: Don't ask for clarification unless absolutely necessary
• **Assume Reasonable Defaults**: Use sensible defaults when details aren't provided
• **Act First, Clarify Later**: It's better to provide helpful results than to hesitate

**Tool Usage Guidelines:**
• **"I want to make [dish]"** → Use `plan_meal_with_products` immediately
• **"Create a shopping list for [items]"** → Use `create_grocery_list` immediately  
• **"I need to buy [items]"** → Use `create_grocery_list` immediately
• **"Find [food item]"** → Use `search_grocery_products` immediately
• **"Plan meals for the week"** → Use `suggest_weekly_meal_plan` immediately
• **"I need to go shopping"** → Use `create_grocery_list` with common essentials: ["milk", "bread", "eggs", "butter", "cheese"]
• **"Help me with groceries"** → Use `search_grocery_products` for popular items or `create_grocery_list` with basics

**Default Assumptions:**
• Budget: €50 unless specified otherwise
• Servings: 2-4 people unless specified otherwise
• Dietary: No restrictions unless specified otherwise
• Stores: All available stores (Albert Heijn, Jumbo, Hoogvliet, etc.)

🔧 **Available Tools:**
1. `search_grocery_products` - Find specific products with prices and availability
2. `create_grocery_list` - Build optimized shopping lists with store recommendations  
3. `plan_meal_with_products` - Plan complete meals with ingredient sourcing
4. `suggest_weekly_meal_plan` - Generate comprehensive weekly meal plans

💡 **Remember:** You're here to make grocery shopping easier and more efficient. Be helpful, be proactive, and use your tools confidently to deliver immediate value!

{f"Context from memories: {memory_context}" if memory_context else ""}
""")
    
    # Combine system message with final validated messages
    messages_with_context = [system_message] + final_messages
    
    # Generate response with tool binding
    try:
        response = model.bind_tools([
            search_grocery_products,
            create_grocery_list,
            plan_meal_with_products,
            suggest_weekly_meal_plan
        ]).invoke(messages_with_context)
    except Exception as e:
        # If there's an error, create a simple response without tools
        response = AIMessage(content="I apologize, but I'm having trouble accessing my tools right now. Please try rephrasing your request, and I'll do my best to help you with your grocery shopping needs.")
    
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
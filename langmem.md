LangMem¶
LangMem helps agents learn and adapt from their interactions over time.

It provides tooling to extract important information from conversations, optimize agent behavior through prompt refinement, and maintain long-term memory.

It offers both functional primitives you can use with any storage system and native integration with LangGraph's storage layer.

This lets your agents continuously improve, personalize their responses, and maintain consistent behavior across sessions.

Key features¶
🧩 Core memory API that works with any storage system
🧠 Memory management tools that agents can use to record and search information during active conversations "in the hot path"
⚙️ Background memory manager that automatically extracts, consolidates, and updates agent knowledge
⚡ Native integration with LangGraph's Long-term Memory Store, available by default in all LangGraph Platform deployments

Memories can be created in two ways:

In the hot path: the agent consciously saves notes using tools (see Hot path quickstart).
👉In the background (this guide): memories are "subconsciously" extracted automatically from conversations.
Hot Path Quickstart Diagram

This guide shows you how to extract and consolidate memories in the background using create_memory_store_manager. The agent will continue as normal while memories are processed in the background.

Prerequisites¶
First, install LangMem:


pip install -U langmem
Configure your environment with an API key for your favorite LLM provider:


export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider
Basic Usage¶
API: init_chat_model | entrypoint | ReflectionExecutor | create_memory_store_manager


from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore

from langmem import ReflectionExecutor, create_memory_store_manager

store = InMemoryStore( 
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)  
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Create memory manager Runnable to extract memories from conversations
memory_manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    # Store memories in the "memories" namespace (aka directory)
    namespace=("memories",),  
)

@entrypoint(store=store)  # Create a LangGraph workflow
async def chat(message: str):
    response = llm.invoke(message)

    # memory_manager extracts memories from conversation history
    # We'll provide it in OpenAI's message format
    to_process = {"messages": [{"role": "user", "content": message}] + [response]}
    await memory_manager.ainvoke(to_process)  
    return response.content
# Run conversation as normal
response = await chat.ainvoke(
    "I like dogs. My dog's name is Fido.",
)
print(response)
# Output: That's nice! Dogs make wonderful companions. Fido is a classic dog name. What kind of dog is Fido?
If you want to see what memories have been extracted, you can search the store:


# (in case our memory manager is still running)
print(store.search(("memories",)))
# [
#     Item(
#         namespace=["memories"],
#         key="0145905e-2b78-4675-9a54-4cb13099bd0b",
#         value={"kind": "Memory", "content": {"content": "User likes dogs as pets"}},
#         created_at="2025-02-06T18:54:32.568595+00:00",
#         updated_at="2025-02-06T18:54:32.568596+00:00",
#         score=None,
#     ),
#     Item(
#         namespace=["memories"],
#         key="19cc4024-999a-4380-95b1-bb9dddc22d22",
#         value={"kind": "Memory", "content": {"content": "User has a dog named Fido"}},
#         created_at="2025-02-06T18:54:32.568680+00:00",
#         updated_at="2025-02-06T18:54:32.568682+00:00",
#         score=None,
#     ),
# ]
For more efficient processing

💡 For active conversations, processing every message can be expensive. See Delayed Memory Processing to learn how to defer processing until conversation activity settles down.

Delayed Background Memory Processing¶
When conversations are active, an agent may receive many messages in quick succession. Instead of processing each message immediately for long-term memory management, you can wait for conversation activity to settle. This guide shows how to use ReflectionExecutor to debounce memory processing.

Problem¶
Processing memories on every message has drawbacks: - Redundant work when messages arrive in quick succession - Incomplete context when processing mid-conversation - Unnecessary token consumption

ReflectionExecutor defers memory processing and cancels redundant work:


from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import ReflectionExecutor, create_memory_store_manager

# Create memory manager to extract memories from conversations 
memory_manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("memories",),
)
# Wrap memory_manager to handle deferred background processing 
executor = ReflectionExecutor(memory_manager)
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

@entrypoint(store=store)
def chat(message: str):
    response = llm.invoke(message)
    # Format conversation for memory processing
    # Must follow OpenAI's message format
    to_process = {"messages": [{"role": "user", "content": message}] + [response]}

    # Wait 30 minutes before processing
    # If new messages arrive before then:
    # 1. Cancel pending processing task
    # 2. Reschedule with new messages included
    delay = 0.5 # In practice would choose longer (30-60 min)
    # depending on app context.
    executor.submit(to_process, after_seconds=delay)
    return response.content

How to Extract Semantic Memories¶
Need to extract multiple related facts from conversations? Here's how to use LangMem's collection pattern for semantic memories. For single-document patterns like user profiles, see Manage User Profile.

Without storage¶
Extract semantic memories:

API: create_memory_manager


from langmem import create_memory_manager 
from pydantic import BaseModel

class Triple(BaseModel): 
    """Store all new facts, preferences, and relationships as triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

# Configure extraction
manager = create_memory_manager(  
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Triple], 
    instructions="Extract user preferences and any other useful information",
    enable_inserts=True,
    enable_deletes=True,
)
After the first short interaction, the system has extracted some semantic triples:


# First conversation - extract triples
conversation1 = [
    {"role": "user", "content": "Alice manages the ML team and mentors Bob, who is also on the team."}
]
memories = manager.invoke({"messages": conversation1})
print("After first conversation:")
for m in memories:
    print(m)
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
The second conversation updates some existing memories. Since we have enabled "deletes", the manager will return RemoveDoc objects to indicate that the memory should be removed, and a new memory will be created in its place. Since this uses the core "functional" API (aka, it doesn't read or write to a database), you can control what "removal" means, be that a soft or hard delete, or simply a down-weighting of the memory.


# Second conversation - update and add triples
conversation2 = [
    {"role": "user", "content": "Bob now leads the ML team and the NLP project."}
]
update = manager.invoke({"messages": conversation2, "existing": memories})
print("After second conversation:")
for m in update:
    print(m)
# ExtractedMemory(id='65fd9b68-77a7-4ea7-ae55-66e1dd603046', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='7f8be100-5687-4410-b82a-fa1cc8d304c0', content=Triple(subject='Bob', predicate='leads', object='ML_team', context=None))
# ExtractedMemory(id='f4c09154-2557-4e68-8145-8ccd8afd6798', content=Triple(subject='Bob', predicate='leads', object='NLP_project', context=None))
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
existing = [m for m in update if isinstance(m.content, Triple)]
The third conversation overwrites even more memories.


# Delete triples about an entity
conversation3 = [
    {"role": "user", "content": "Alice left the company."}
]
final = manager.invoke({"messages": conversation3, "existing": existing})
print("After third conversation:")
for m in final:
    print(m)
# ExtractedMemory(id='7ca76217-66a4-4041-ba3d-46a03ea58c1b', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='35b443c7-49e2-4007-8624-f1d6bcb6dc69', content=RemoveDoc(json_doc_id='0214f151-b0c5-40c4-b621-db36b845956c'))
# ExtractedMemory(id='65fd9b68-77a7-4ea7-ae55-66e1dd603046', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='7f8be100-5687-4410-b82a-fa1cc8d304c0', content=Triple(subject='Bob', predicate='leads', object='ML_team', context=None))
# ExtractedMemory(id='f4c09154-2557-4e68-8145-8ccd8afd6798', content=Triple(subject='Bob', predicate='leads', object='NLP_project', context=None))
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
For more about semantic memories, see Memory Types.

With storage¶
The same extraction can be managed automatically by LangGraph's BaseStore:

API: init_chat_model | entrypoint | create_memory_store_manager


from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

# Set up store and models
store = InMemoryStore(  
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("chat", "{user_id}", "triples"),  
    schemas=[Triple],
    instructions="Extract all user information and events as triples.",
    enable_inserts=True,
    enable_deletes=True,
)
my_llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
You can also extract multiple memory types at once:


schemas=[Triple, Preference, Relationship]
Each type can have its own extraction rules and storage patterns Namespaces let you organize memories by user, team, or domain:


# User-specific memories
("chat", "user_123", "triples")

# Team-shared knowledge
("chat", "team_x", "triples")

# Domain-specific extraction
("chat", "user_123", "preferences")
The {user_id} placeholder is replaced at runtime:


# Extract memories for User A
manager.invokse(
    {"messages": [{"role": "user", "content": "I prefer dark mode"}]},
    config={"configurable": {"user_id": "user-a"}}  
)

# Define app with store context
@entrypoint(store=store) 
def app(messages: list):
    response = my_llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            *messages
        ]
    )

    # Extract and store triples (Uses store from @entrypoint context)
    manager.invoke({"messages": messages}) 
    return response
Then running the app:


# First conversation
app.invoke(
    [
        {
            "role": "user",
            "content": "Alice manages the ML team and mentors Bob, who is also on the team.",
        },
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Second conversation
app.invoke(
    [
        {"role": "user", "content": "Bob now leads the ML team and the NLP project."},
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Third conversation
app.invoke(
    [
        {"role": "user", "content": "Alice left the company."},
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Check stored triples
for item in store.search(("chat", "user123")):
    print(item.namespace, item.value)

# Output:
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'is_member_of', 'object': 'ML_team', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'leads', 'object': 'ML_team', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'leads', 'object': 'NLP_project', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Alice', 'predicate': 'employment_status', 'object': 'left_company', 'context': None}}
See Storage System for namespace patterns. This approach is also compatible with the ReflectionExecutor to defer & deduplicate memory processing.

Using a Memory Manager Agent¶
The technique above tries to manage memory, including insertions, deletions, and deletions, in a single LLM call. If there is a lot of new information, this may be complicated for the LLM to multi-task. Alternatively, you could create an agent, similar to that in the quick start, which you prompt to manage memory over multiple LLM calls. You can still serparate this agent from your user-facing agent, but it can give your LLM extra time to process new information and complex updates.

API: init_chat_model | entrypoint | create_react_agent | create_manage_memory_tool | create_search_memory_tool


from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up store and checkpointer
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
my_llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


def prompt(state):
    """Prepare messages with context from existing memories."""
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a memory manager. Extract and manage all important knowledge, rules, and events using the provided tools.



Existing memories:
<memories>
{memories}
</memories>

Use the manage_memory tool to update and contextualize existing memories, create new ones, or delete old ones that are no longer valid.
You can also expand your search of existing memories to augment using the search tool."""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


# Create the memory extraction agent
manager = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    prompt=prompt,
    tools=[
        # Agent can create/update/delete memories
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
)


# Run extraction in background
@entrypoint(store=store)
def app(messages: list):
    response = my_llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            *messages,
        ]
    )

    # Extract and store triples (Uses store from @entrypoint context)
    manager.invoke({"messages": messages})
    return response


app.invoke(
    [
        {
            "role": "user",
            "content": "Alice manages the ML team and mentors Bob, who is also on the team.",
        }
    ]
)

print(store.search(("memories",)))

# [
#     Item(
#         namespace=["memories"],
#         key="5ca8dacc-7d46-40bb-9b3d-f4c2dc5c4b30",
#         value={"content": "Alice is the manager of the ML (Machine Learning) team"},
#         created_at="2025-02-11T00:28:01.688490+00:00",
#         updated_at="2025-02-11T00:28:01.688499+00:00",
#         score=None,
#     ),
#     Item(
#         namespace=["memories"],
#         key="586783fa-e501-4835-8651-028c2735f0d0",
#         value={"content": "Bob works on the ML team"},
#         created_at="2025-02-11T00:28:04.408826+00:00",
#         updated_at="2025-02-11T00:28:04.408841+00:00",
#         score=None,
#     ),
#     Item(
#         namespace=["memories"],
#         key="19f75f64-8787-4150-a439-22068b00118a",
#         value={"content": "Alice mentors Bob on the ML team"},
#         created_at="2025-02-11T00:28:06.951838+00:00",
#         updated_at="2025-02-11T00:28:06.951847+00:00",
#         score=None,
#     ),
# ]
This approach is also compatible with the ReflectionExecutor to defer & deduplicate memory processing.

When to Use Semantic Memories¶
Semantic memories help agents learn from conversations. They extract and store meaningful information that might be useful in future interactions. For example, when discussing a project, the agent might remember technical requirements, team structure, or key decisions - anything that could provide helpful context later.

The goal is to build understanding over time, just like humans do through repeated interactions. Not everything needs to be remembered - focus on information that helps the agent be more helpful in future conversations. Semantic memory works best when the agent is able to save important memories and the dense relationships between them so that it can later recall not just "what" but "why" and "how".

ow to Manage Long Context with Summarization¶
In modern LLM applications, context size can grow quickly and hit provider limitations, whether you're building chatbots with many conversation turns or agentic systems with numerous tool calls.

One effective strategy for handling this is to summarize earlier messages once they reach a certain threshold. This guide demonstrates how to implement this approach in your LangGraph application using LangMem's prebuilt summarize_messages and SummarizationNode.

Using in a Simple Chatbot¶
Below is an example of a simple multi-turn chatbot with summarization:

API: ChatOpenAI | StateGraph | START | summarize_messages | RunningSummary


from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import summarize_messages, RunningSummary
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=128)  

# We will keep track of our running summary in the graph state
class SummaryState(MessagesState):
    summary: RunningSummary | None

# Define the node that will be calling the LLM
def call_model(state: SummaryState) -> SummaryState:
    summarization_result = summarize_messages(  
        state["messages"],
        # IMPORTANT: Pass running summary, if any
        running_summary=state.get("summary"),  
        token_counter=model.get_num_tokens_from_messages,
        model=summarization_model, 
        max_tokens=256,  
        max_tokens_before_summary=256,  
        max_summary_tokens=128
    )
    response = model.invoke(summarization_result.messages)
    state_update = {"messages": [response]}
    if summarization_result.running_summary:  
        state_update["summary"] = summarization_result.running_summary
    return state_update


checkpointer = InMemorySaver()
builder = StateGraph(SummaryState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)  

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
graph.invoke({"messages": "what's my name?"}, config)
Using in UI

An important question is how to present messages to the users in the UI of your app. We recommend rendering the full, unmodified message history. You may choose to additionally render the summary and messages that are passed to the LLM. We also recommend using separate LangGraph state keys for the full message history (e.g., "messages") and summarization results (e.g., "summary"). In SummarizationNode, summarization results are stored in a separate state key called context (see example below).

Using SummarizationNode¶
You can also separate the summarization into a dedicated node. Let's explore how to modify the above example to use SummarizationNode for achieving the same results:

API: ChatOpenAI | StateGraph | START | SummarizationNode | RunningSummary


from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=128)


class State(MessagesState):
    context: dict[str, Any]  


class LLMInputState(TypedDict):  
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]

summarization_node = SummarizationNode(  
    token_counter=model.get_num_tokens_from_messages,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

# IMPORTANT: we're passing a private input state here to isolate the summarization
def call_model(state: LLMInputState):  
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
graph = builder.compile(checkpointer=checkpointer)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
graph.invoke({"messages": "what's my name?"}, config)
Using in a ReAct Agent¶
A common use case is summarizing message history in a tool calling agent. Below example demonstrates how to implement this in a ReAct-style LangGraph agent:

API: ChatOpenAI | tool | StateGraph | START | END | ToolNode | SummarizationNode | RunningSummary


from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

class State(MessagesState):
    context: dict[str, Any]

def search(query: str):
    """Search the web."""
    if "weather" in query.lower():
        return "The weather is sunny in New York, with a high of 104 degrees."
    elif "broadway" in query.lower():
        return "Hamilton is always on!"
    else:
        raise "Not enough information"

tools = [search]

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=128)

summarization_node = SummarizationNode(
    token_counter=model.get_num_tokens_from_messages,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=1024,
    max_summary_tokens=128,
)

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]

def call_model(state: LLMInputState):
    response = model.bind_tools(tools).invoke(state["summarized_messages"])
    return {"messages": [response]}

# Define a router that determines whether to execute tools or exit
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("summarize_node", summarization_node)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("summarize_node")
builder.add_edge("summarize_node", "call_model")
builder.add_conditional_edges("call_model", should_continue, path_map=["tools", END])
builder.add_edge("tools", "summarize_node")  
graph = builder.compile(checkpointer=checkpointer)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, i am bob"}, config)
graph.invoke({"messages": "what's the weather in nyc this weekend"}, config)
graph.invoke({"messages": "what's new on broadway?"}, config)

Short Term Memory API Reference¶
Classes:

SummarizationNode – A LangGraph node that summarizes messages when they exceed a token limit and replaces them with a summary message.
SummarizationResult – Result of message summarization.
RunningSummary – Object for storing information about the previous summarization.
Functions:

summarize_messages – Summarize messages when they exceed a token limit and replace them with a summary message.
 SummarizationNode ¶
Bases: RunnableCallable

A LangGraph node that summarizes messages when they exceed a token limit and replaces them with a summary message.

Methods:

__init__ – A LangGraph node that summarizes messages when they exceed a token limit and replaces them with a summary message.
 __init__ ¶

__init__(
    *,
    model: LanguageModelLike,
    max_tokens: int,
    max_tokens_before_summary: int | None = None,
    max_summary_tokens: int = 256,
    token_counter: TokenCounter = count_tokens_approximately,
    initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
    final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
    input_messages_key: str = "messages",
    output_messages_key: str = "summarized_messages",
    name: str = "summarization",
) -> None
A LangGraph node that summarizes messages when they exceed a token limit and replaces them with a summary message.

Processes the messages from oldest to newest: once the cumulative number of message tokens reaches max_tokens_before_summary, all messages within max_tokens_before_summary are summarized (excluding the system message, if any) and replaced with a new summary message. The resulting list of messages is [summary_message] + remaining_messages.

Parameters:

model (LanguageModelLike) – The language model to use for generating summaries.
max_tokens (int) – Maximum number of tokens to return in the final output. Will be enforced only after summarization.
max_tokens_before_summary (int | None, default: None ) – Maximum number of tokens to accumulate before triggering summarization. Defaults to the same value as max_tokens if not provided. This allows fitting more tokens into the summarization LLM, if needed.
Note

If the last message within max_tokens_before_summary is an AI message with tool calls, all of the subsequent, corresponding tool messages will be summarized as well.

Note

If the number of tokens to be summarized is greater than max_tokens, only the last max_tokens amongst those will be summarized. This is done to prevent exceeding the context window of the summarization LLM (assumed to be capped at max_tokens).

max_summary_tokens (int, default: 256 ) – Maximum number of tokens to budget for the summary.
Note

This parameter is not passed to the summary-generating LLM to limit the length of the summary. It is only used for correctly estimating the maximum allowed token budget. If you want to enforce it, you would need to pass model.bind(max_tokens=max_summary_tokens) as the model parameter to this function.

token_counter (TokenCounter, default: count_tokens_approximately ) – Function to count tokens in a message. Defaults to approximate counting. For more accurate counts you can use model.get_num_tokens_from_messages.
initial_summary_prompt (ChatPromptTemplate, default: DEFAULT_INITIAL_SUMMARY_PROMPT ) – Prompt template for generating the first summary.
existing_summary_prompt (ChatPromptTemplate, default: DEFAULT_EXISTING_SUMMARY_PROMPT ) – Prompt template for updating an existing (running) summary.
final_prompt (ChatPromptTemplate, default: DEFAULT_FINAL_SUMMARY_PROMPT ) – Prompt template that combines summary with the remaining messages before returning.
input_messages_key (str, default: 'messages' ) – Key in the input graph state that contains the list of messages to summarize.
output_messages_key (str, default: 'summarized_messages' ) – Key in the state update that contains the list of updated messages.
Warning

By default, the output_messages_key is different from the input_messages_key. This is done to decouple summarized messages from the main list of messages in the graph state (i.e., input_messages_key). You should only make them the same if you want to overwrite the main list of messages (i.e., input_messages_key).

name (str, default: 'summarization' ) – Name of the summarization node.
Returns:

None – LangGraph state update in the following format:

{
    "output_messages_key": <list of updated messages ready to be input to the LLM after summarization, including a message with a summary (if any)>,
    "context": {"running_summary": <RunningSummary object>}
}
Example

from typing import Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=128)


class State(MessagesState):
    context: dict[str, Any]


class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]


summarization_node = SummarizationNode(
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)


def call_model(state: LLMInputState):
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}


checkpointer = InMemorySaver()
workflow = StateGraph(State)
workflow.add_node(call_model)
workflow.add_node("summarize", summarization_node)
workflow.add_edge(START, "summarize")
workflow.add_edge("summarize", "call_model")
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
graph.invoke({"messages": "what's my name?"}, config)
 SummarizationResult dataclass ¶
Result of message summarization.

Attributes:

messages (list[AnyMessage]) – List of updated messages that are ready to be input to the LLM after summarization, including a message with a summary (if any).
running_summary (RunningSummary | None) – Information about previous summarization (the summary and the IDs of the previously summarized messages.
 messages instance-attribute ¶

messages: list[AnyMessage]
List of updated messages that are ready to be input to the LLM after summarization, including a message with a summary (if any).

 running_summary class-attribute instance-attribute ¶

running_summary: RunningSummary | None = None
Information about previous summarization (the summary and the IDs of the previously summarized messages. Can be None if no summarization was performed (not enough messages to summarize).

 RunningSummary dataclass ¶
Object for storing information about the previous summarization.

Used on subsequent calls to summarize_messages to avoid summarizing the same messages.

Attributes:

summary (str) – Latest summary of the messages, updated every time the summarization is performed.
summarized_message_ids (set[str]) – The IDs of all of the messages that have been previously summarized.
last_summarized_message_id (str | None) – The ID of the last message that was summarized.
 summary instance-attribute ¶

summary: str
Latest summary of the messages, updated every time the summarization is performed.

 summarized_message_ids instance-attribute ¶

summarized_message_ids: set[str]
The IDs of all of the messages that have been previously summarized.

 last_summarized_message_id instance-attribute ¶

last_summarized_message_id: str | None
The ID of the last message that was summarized.

 summarize_messages ¶

summarize_messages(
    messages: list[AnyMessage],
    *,
    running_summary: RunningSummary | None,
    model: LanguageModelLike,
    max_tokens: int,
    max_tokens_before_summary: int | None = None,
    max_summary_tokens: int = 256,
    token_counter: TokenCounter = count_tokens_approximately,
    initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
    final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
) -> SummarizationResult
Summarize messages when they exceed a token limit and replace them with a summary message.

This function processes the messages from oldest to newest: once the cumulative number of message tokens reaches max_tokens_before_summary, all messages within max_tokens_before_summary are summarized (excluding the system message, if any) and replaced with a new summary message. The resulting list of messages is [summary_message] + remaining_messages.

Parameters:

messages (list[AnyMessage]) – The list of messages to process.
running_summary (RunningSummary | None) – Optional running summary object with information about the previous summarization. If provided: - only messages that were not previously summarized will be processed - if no new summary is generated, the running summary will be added to the returned messages - if a new summary needs to be generated, it is generated by incorporating the existing summary value from the running summary
model (LanguageModelLike) – The language model to use for generating summaries.
max_tokens (int) – Maximum number of tokens to return in the final output. Will be enforced only after summarization. This will also be used as the maximum number of tokens to feed to the summarization LLM.
max_tokens_before_summary (int | None, default: None ) – Maximum number of tokens to accumulate before triggering summarization. Defaults to the same value as max_tokens if not provided. This allows fitting more tokens into the summarization LLM, if needed.
Note

If the last message within max_tokens_before_summary is an AI message with tool calls, all of the subsequent, corresponding tool messages will be summarized as well.

Note

If the number of tokens to be summarized is greater than max_tokens, only the last max_tokens amongst those will be summarized. This is done to prevent exceeding the context window of the summarization LLM (assumed to be capped at max_tokens).

max_summary_tokens (int, default: 256 ) – Maximum number of tokens to budget for the summary.
Note

This parameter is not passed to the summary-generating LLM to limit the length of the summary. It is only used for correctly estimating the maximum allowed token budget. If you want to enforce it, you would need to pass model.bind(max_tokens=max_summary_tokens) as the model parameter to this function.

token_counter (TokenCounter, default: count_tokens_approximately ) – Function to count tokens in a message. Defaults to approximate counting. For more accurate counts you can use model.get_num_tokens_from_messages.
initial_summary_prompt (ChatPromptTemplate, default: DEFAULT_INITIAL_SUMMARY_PROMPT ) – Prompt template for generating the first summary.
existing_summary_prompt (ChatPromptTemplate, default: DEFAULT_EXISTING_SUMMARY_PROMPT ) – Prompt template for updating an existing (running) summary.
final_prompt (ChatPromptTemplate, default: DEFAULT_FINAL_SUMMARY_PROMPT ) – Prompt template that combines summary with the remaining messages before returning.
Returns:

SummarizationResult – A SummarizationResult object containing the updated messages and a running summary. - messages: list of updated messages ready to be input to the LLM - running_summary: RunningSummary object - summary: text of the latest summary - summarized_message_ids: set of message IDs that were previously summarized - last_summarized_message_id: ID of the last message that was summarized
Example

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import summarize_messages, RunningSummary
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=128)


class SummaryState(MessagesState):
    summary: RunningSummary | None


def call_model(state):
    summarization_result = summarize_messages(
        state["messages"],
        running_summary=state.get("summary"),
        model=summarization_model,
        max_tokens=256,
        max_tokens_before_summary=256,
        max_summary_tokens=128,
    )
    response = model.invoke(summarization_result.messages)
    state_update = {"messages": [response]}
    if summarization_result.running_summary:
        state_update["summary"] = summarization_result.running_summary
    return state_update


checkpointer = InMemorySaver()
workflow = StateGraph(SummaryState)
workflow.add_node(call_model)
workflow.add_edge(START, "call_model")
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
graph.invoke({"messages": "what's my name?"}, config)

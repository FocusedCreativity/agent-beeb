# Agent Beeb 🛒

**Agent Beeb** is an intelligent grocery shopping assistant powered by LangGraph and LangMem. It helps users with personalized grocery shopping through product search, meal planning, list creation, and memory-based recommendations.

## ✨ Features

### 🧠 **Smart Memory Management**
- **Long-term Memory**: Remembers user preferences, dietary restrictions, and shopping habits
- **Semantic Memory Search**: Finds relevant past interactions to provide personalized recommendations
- **Background Processing**: Automatically extracts and consolidates memories from conversations

### 🛍️ **Comprehensive Product Database**
- **40,000+ Products**: Access to extensive grocery product database with pricing
- **Multi-store Support**: Compare prices across Albert Heijn, Jumbo, Hoogvliet, and more
- **Real-time Pricing**: Current prices and availability information
- **Nutrition & Ingredients**: Detailed product information including dietary compatibility

### 🍽️ **Meal Planning & Shopping Lists**
- **Intelligent Meal Planning**: Plan single meals or weekly meal plans with real products
- **Smart Shopping Lists**: Create lists with actual products, prices, and store optimization
- **Budget Tracking**: Monitor spending and optimize for budget constraints
- **Dietary Preferences**: Filter for vegetarian, vegan, gluten-free, organic, and more

### 🔧 **Advanced Capabilities**
- **Conversation Context**: Maintains context across multiple interactions
- **Tool Integration**: Seamlessly uses multiple specialized tools
- **Summarization**: Handles long conversations with intelligent summarization
- **Store Optimization**: Suggests optimal shopping routes and store visits

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- Supabase account (for product database)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yusephswessi/agent-beeb.git
   cd agent-beeb
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the example**
   ```bash
   python -c "from my_agent.agent import example_conversation; example_conversation()"
   ```

## 🛠️ Configuration

Create a `.env` file in the root directory:

```env
# OpenAI API Key for LLM and embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Supabase Configuration for Product Database
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

## 📖 Usage

### Basic Usage

```python
from my_agent.agent import create_agent_beeb

# Create Agent Beeb instance
agent = create_agent_beeb()

# Have a conversation
response = agent.chat(
    message="I'm vegetarian and need help planning meals for this week",
    user_id="user_123"
)

print(response["response"])
```

### Advanced Usage

```python
# Chat with memory and context
response = agent.chat(
    message="Find me some protein sources under €5",
    user_id="user_123",
    thread_id="conversation_1",
    user_context={
        "dietary_preferences": ["vegetarian"],
        "budget": "€30 per week"
    }
)

# View user memories
memories = agent.get_user_memories("user_123")
for memory in memories:
    print(f"- {memory['content']}")

# Search memories
results = agent.search_memories("user_123", "vegetarian protein")
```

## 🏗️ Architecture

### Core Components

```
Agent Beeb/
├── my_agent/
│   ├── agent.py           # Main agent class and orchestration
│   └── utils/
│       ├── state.py       # LangGraph state management
│       ├── nodes.py       # Conversation and memory nodes
│       └── tools.py       # Grocery shopping tools
├── docs/                  # Documentation and guides
├── requirements.txt       # Python dependencies
└── langgraph.json        # LangGraph configuration
```

### Key Technologies

- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Workflow orchestration and state management
- **[LangMem](https://github.com/langchain-ai/langmem)**: Memory management and conversation summarization
- **[LangChain](https://github.com/langchain-ai/langchain)**: LLM integration and tool calling
- **[Supabase](https://supabase.com)**: Product database and pricing information
- **[OpenAI](https://openai.com)**: Language model and embeddings

### Memory Architecture

Agent Beeb uses LangMem for sophisticated memory management:

1. **Short-term Memory**: Conversation context and summarization
2. **Long-term Memory**: User preferences, dietary restrictions, shopping patterns
3. **Semantic Search**: Vector-based memory retrieval for personalized responses
4. **Background Processing**: Automated memory extraction and consolidation

## 🔧 Available Tools

### Product Search
- **`search_grocery_products`**: Search through 40,000+ products with price optimization
- Supports dietary filtering, budget constraints, and store preferences

### List Management
- **`create_grocery_list`**: Build shopping lists with real products and costs
- Optimizes for budget and store efficiency

### Meal Planning
- **`plan_meal_with_products`**: Plan specific meals with actual ingredients
- **`suggest_weekly_meal_plan`**: Create comprehensive weekly meal plans

## 🎯 Use Cases

### Personal Shopping Assistant
- "Find me gluten-free pasta under €3"
- "What's the cheapest milk at Albert Heijn?"
- "I need protein sources for my vegetarian diet"

### Meal Planning
- "Plan a healthy dinner for 4 people under €15"
- "Create a weekly meal plan for my family"
- "I want to cook Italian food tonight"

### Budget Management
- "Help me shop for €25 this week"
- "What's the best value for breakfast items?"
- "Compare prices across different stores"

### Dietary Management
- "Find vegan alternatives to dairy products"
- "I'm on a keto diet, what can I buy?"
- "Show me organic options for baby food"

## 📊 Memory Examples

Agent Beeb automatically learns from conversations:

```
User: "I'm vegetarian and have a €30 weekly budget"
↓
Memories Extracted:
- "User follows a vegetarian diet"
- "User has a budget of €30 per week for groceries"
- "User is price-conscious and looks for budget-friendly options"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the foundational LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [LangMem](https://github.com/langchain-ai/langmem) for memory management
- [Supabase](https://supabase.com) for the product database infrastructure

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yusephswessi/agent-beeb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yusephswessi/agent-beeb/discussions)
- **Email**: [support@agent-beeb.com](mailto:support@agent-beeb.com)

---

**Agent Beeb** - Making grocery shopping intelligent, personalized, and efficient! 🛒✨ 
from setuptools import setup, find_packages

setup(
    name="my_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.2.0",
        "langchain-core>=0.3.0",
        "langchain-openai>=0.2.0",
        "langgraph-checkpoint-postgres>=0.0.1",
        "psycopg2-binary>=2.9.0",
        "asyncpg>=0.29.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.12.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="LangGraph chatbot with message summarization and external DB memory",
) 
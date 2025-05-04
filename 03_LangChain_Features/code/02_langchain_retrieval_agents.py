import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage

# --- Setup ---
# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Ensure your OPENAI_API_KEY is set in your environment or a .env file
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize LLM and Embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
embeddings = OpenAIEmbeddings()
print("LLM and Embeddings Initialized.")
print("----------------------------------------\n")


# --- 1. Document Loading (In-Memory) & Splitting ---
print("--- 1. Document Loading & Splitting ---")
# Using an in-memory string instead of loading from a file
document_text = """
LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).
It provides a standard interface for chains, integrations with tools, and end-to-end chains for common applications.
Key components include Models (LLMs, Chat Models, Text Embedding Models), Prompts (Prompt Templates, Output Parsers),
Indexes (Document Loaders, Text Splitters, Vector Stores, Retrievers), Memory (short-term and long-term),
Chains (sequences of calls), and Agents (which use LLMs to decide actions).
Vector stores like FAISS, Chroma, and Pinecone are used to store text embeddings for efficient similarity searches.
Agents utilize tools, which can be other chains, external APIs (like Google Search), or custom Python functions.
LangChain helps developers build context-aware and reasoning applications.
"""
print(f"Loaded Document Text (excerpt):\n{document_text[:100]}...\n")

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
document_chunks = text_splitter.create_documents([document_text])
print(f"Document split into {len(document_chunks)} chunks.")
print("----------------------------------------\n")


# --- 2. Embeddings & Vector Store (FAISS) ---
print("--- 2. Embeddings & Vector Store (FAISS) ---")
# Create embeddings for the chunks and store them in FAISS
print("Creating vector store...")
vector_store = FAISS.from_documents(document_chunks, embeddings)
print("FAISS vector store created successfully.")

# Perform a similarity search
query = "What are the main components of LangChain?"
search_results = vector_store.similarity_search(query, k=2) # Get top 2 results

print(f"\nSimilarity search for: '{query}'")
if search_results:
    for i, doc in enumerate(search_results):
        print(f"Result {i+1}:\n{doc.page_content}\n---")
else:
    print("No relevant results found.")
print("----------------------------------------\n")


# --- 3. Agents and Tools ---
print("--- 3. Agents and Tools ---")

# Tool 1: DuckDuckGo Search
search_tool = DuckDuckGoSearchResults()
dd_tool = Tool(
    name="DuckDuckGo Search",
    func=search_tool.run,
    description="Useful for when you need to answer questions about current events or find up-to-date information."
)

# Tool 2: Custom Python Function Tool
def get_langchain_info(topic: str) -> str:
    """Provides specific information about LangChain components based on the input topic."""
    # You could make this more sophisticated, e.g., use the vector store
    if "agent" in topic.lower():
        return "Agents in LangChain use an LLM to choose a sequence of actions to take, often using Tools."
    elif "memory" in topic.lower():
        return "Memory in LangChain allows chains or agents to remember previous interactions."
    else:
        # Fallback to vector store search if topic is not specific
        results = vector_store.similarity_search(f"Information about {topic} in LangChain", k=1)
        if results:
            return results[0].page_content
        return f"I don't have specific pre-defined info on '{topic}' in LangChain, but search might help."

custom_tool = Tool(
    name="LangChain Info",
    func=get_langchain_info,
    description="Useful for getting specific information about LangChain components like 'agents' or 'memory'."
)

# Initialize the Agent
tools = [dd_tool, custom_tool]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False # Set to True to see the agent's thought process
)

# Run the agent with different queries
print("\nRunning Agent Query 1 (needs specific LangChain info):")
agent_response1 = agent.run("Tell me about agents in LangChain.")
print(f"Agent Response 1:\n{agent_response1}")

print("\nRunning Agent Query 2 (needs current info):")
agent_response2 = agent.run("What is the latest version of LangChain released?")
print(f"Agent Response 2:\n{agent_response2}")

print("\nRunning Agent Query 3 (general LangChain query handled by custom tool fallback):")
agent_response3 = agent.run("What are vector stores used for in LangChain?")
print(f"Agent Response 3:\n{agent_response3}")
print("----------------------------------------\n")



# --- 4. Streaming ---
print("--- 4. Streaming ---")
print("Streaming a simple joke from the LLM:")

full_response = ""
for chunk in llm.stream([HumanMessage(content="Tell me a short joke about AI.")]):
    print(chunk.content, end="", flush=True)
    full_response += chunk.content
    time.sleep(0.5) # Optional small delay for visual effect

print("\n\nStreaming finished.")
# print(f"Full streamed response: {full_response}") # Optionally print the full response

print("----------------------------------------\n")

print("Retrieval and Agent examples finished.")

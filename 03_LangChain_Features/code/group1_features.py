from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchResults

# Initialize the OpenAI Chat Model (GPT-3.5 Turbo)
# - `temperature=0.7` makes responses slightly more creative.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

print("----------------START----------------\n")

# ───────────────────────────────────────────
# 🧩 **Chaining & Prompt Templates**
# ───────────────────────────────────────────

print("----------------🧩 Chaining & Prompt Templates ----------------\n")

# Example 1: **Chaining & Prompt Templates** 🧩
# - A **Prompt Template** is created to structure user queries.
# - The **LLMChain** connects this prompt with the LLM to process input dynamically.

prompt_template = "Define the term: {word}"
prompt = PromptTemplate(input_variables=["word"], template=prompt_template)

# Creating a chain that links the model to the prompt template
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Running the chain with a specific input
result = llm_chain.run("Artificial Intelligence")
print(f"Definition of AI: {result}\n")

print("----------------🧩 Chaining & Prompt Templates ----------------\n")



# ───────────────────────────────────────────
# 🧠 **Memory - Conversation Buffer**
# ───────────────────────────────────────────

print("----------------🧠 Memory - Conversation Buffer ----------------\n")

# Example 3: **Memory (Conversation Buffer)** 🧠
# - Memory stores conversation history, allowing context retention between queries.

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Creating an LLMChain that uses memory to retain responses
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

# User asks about AI
response1 = conversation.run("Artificial Intelligence")

# User asks about Machine Learning (previous context is retained)
response2 = conversation.run("Machine Learning")

# Printing stored memory
print("Conversation Memory:", memory.load_memory_variables({}), "\n")
# View in any JSON formatter

print("----------------🧠 Memory - Conversation Buffer ----------------\n")







# ───────────────────────────────────────────
# 🤖 **Agent Initialization - Searching & Summarization**
# ───────────────────────────────────────────

print("----------------🤖 Agent Initialization ----------------\n")

# Example 4: **Agent Initialization (Using tools to search & summarize)** 🤖
# - Agents allow the model to interact with external tools dynamically.
# - Here, an agent is initialized with a search tool.

tools = [
    Tool(name="search", func=lambda query: query, description="Performs a web search"),
]

# Initializing an agent with search capabilities
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

# Asking a factual question
agent_response = agent.run("What is the capital of France?")
print(f"Agent Response: {agent_response}\n")

print("----------------🤖 Agent Initialization ----------------\n")






# ───────────────────────────────────────────
# 🛠️ **Tool Integration with LLM**
# ───────────────────────────────────────────

print("----------------🛠️ Tool Integration with LLM ----------------\n")

# Example 5: **Tool Integration with LLM** 🛠️
# - Here, we integrate **DuckDuckGo** as a search tool within the LLM.

tools = [
    Tool(name="duckduckgo", func=search.run, description="Fetches real-time search results"),
]

# Creating an LLM-based agent with tool integration
llm_chain_with_tools = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

# Asking a real-world query
agent_response2 = llm_chain_with_tools.run("What's the weather in Paris?")
print(f"Agent Response with tools: {agent_response2}\n")

print("----------------🛠️ Tool Integration with LLM ----------------\n")






# ───────────────────────────────────────────
# ⚙️ **Tool Execution & Custom Logic**
# ───────────────────────────────────────────

print("----------------⚙️ Tool Execution & Custom Logic ----------------\n")

# Example 6: **Tool Execution with Custom Logic** ⚙️
# - Custom tools can be created to handle specific types of queries.

# Custom function to process AI-related queries
def custom_tool_function(query):
    if "AI" in query:
        return "AI stands for Artificial Intelligence, enabling machines to mimic human intelligence."
    else:
        return "I am not sure about that."

# Creating a tool that specializes in AI-related queries
tool = Tool(name="AI_Info", func=custom_tool_function, description="Provides AI-related information")

# Creating an agent that utilizes the custom AI tool
custom_agent = initialize_agent([tool], llm, agent_type="zero-shot-react-description", verbose=True)

# Asking an AI-related question
custom_response = custom_agent.run("Tell me about AI")
print(f"Custom Agent Response: {custom_response}\n")

print("----------------⚙️ Tool Execution & Custom Logic ----------------\n")


print("----------------END----------------")





# Group 1:
# Chaining: Combining prompts into sequential steps.
# Prompt Templates: Generating responses based on a template.
# Text Generation: Simple text-based model generation.
# Tools: Integrating external tools like web scraping.
# Web Scraping: Extracting search results.
# Memory: Maintaining state across multiple interactions.

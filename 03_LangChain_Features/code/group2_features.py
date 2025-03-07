from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool

# ───────────────────────────────────────────
# 🚀 **Initializing LLM & Embeddings**
# ───────────────────────────────────────────

# 🧠 **Initialize OpenAI Chat Model** 
# - Used for generating text-based responses.
# - `temperature=0.7` makes responses slightly creative.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 🔍 **Initialize OpenAI Embeddings**
# - Used to convert text into vector representations for similarity search.
embeddings = OpenAIEmbeddings()

print("\n\n----------------- START -----------------\n")







# ───────────────────────────────────────────
# 🤖🔧 **Agent with Multiple Tools**
# ───────────────────────────────────────────

print("\n\n------ 🤖🔧 **Agent with Multiple Tools** ------")

# Example 1: **Agent with Multiple Tools** 🔎🤖
# - Agents are AI-driven assistants that can use tools.
# - Here, we initialize an agent with a search tool.

tools = [
    Tool(name="search", func=lambda query: query, description="Searches for information"),
]

# Creating the agent with tool support
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

# Querying the agent
agent_response = agent.run("What is the capital of France?")
print(f"Agent Response: {agent_response}\n")

print("------ 🤖🔧 **Agent with Multiple Tools** ------\n")






# ───────────────────────────────────────────
# 🗂️🔍 **Vector Store (Storing & Retrieving Documents)**
# ───────────────────────────────────────────

print("\n\n------ 🗂️🔍 **Vector Store (Storing & Retrieving Documents)** ------")

# Example 2: **Vector Store using FAISS** 🗂️🔍
# - Loads text documents and stores them as vector embeddings.
# - Enables similarity search for retrieving relevant documents.

# 📄 **Load Documents**
loader = TextLoader('example_text.txt')
documents = loader.load()

# 🏦 **Store Documents in FAISS**
vector_store = FAISS.from_documents(documents, embeddings)

# 🔍 **Perform Similarity Search**
query = "Tell me more about AI"
results = vector_store.similarity_search(query)

# Display the first 3 search results
print(f"Vector Store Results: {results[:3]}\n")

print("------ 🗂️🔍 **Vector Store (Storing & Retrieving Documents)** ------\n")






# ───────────────────────────────────────────
# ✂️📜 **Custom Chain to Summarize a Document**
# ───────────────────────────────────────────

print("\n\n------ ✂️📜 **Custom Chain to Summarize a Document** ------")

# Example 3: **Custom Chain for Summarization** ✂️📜
# - A **Prompt Template** is created for summarization.
# - The **LLMChain** executes the summarization task.

summary_prompt = PromptTemplate(input_variables=["text"], template="Summarize the following: {text}")
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Sample document text for summarization
document_text = "LangChain is a framework for building language model applications..."
summary = summary_chain.run(document_text)

print(f"Document Summary: {summary}\n")

print("------ ✂️📜 **Custom Chain to Summarize a Document** ------\n")






# ───────────────────────────────────────────
# ⚙️ **Execution & Custom Logic**
# ───────────────────────────────────────────

print("\n\n------ ⚙️ **Execution & Custom Logic** ------")

# Example 4: **Custom Execution Logic** ⚙️
# - A function that checks if input data is long enough.

def process_data(data):
    """Processes input data based on length."""
    if len(data.split()) < 50:
        return "Data is too short."
    return "Data is processed."

# Sample input data
data = "This is a sample sentence."
execution_response = process_data(data)

print(f"Execution Response: {execution_response}\n")

print("------ ⚙️ **Execution & Custom Logic** ------\n")






# ───────────────────────────────────────────
# 🧠⚖️ **Conditional Logic for Different Actions**
# ───────────────────────────────────────────

print("\n\n------ 🧠⚖️ **Conditional Logic for Different Actions** ------")

# Example 5: **Conditional Logic for Document Processing** 🧠⚖️
# - If a document is too short, it returns an alert.
# - Otherwise, it confirms the document is long enough.

def check_document_length(document):
    """Determines if a document is sufficiently long."""
    if len(document.split()) < 50:
        return "Document is too short."
    return "Document is long enough."

doc_check = check_document_length(document_text)

print(f"Document Length Check: {doc_check}\n")

print("------ 🧠⚖️ **Conditional Logic for Different Actions** ------\n")





# ───────────────────────────────────────────
# 🔄🗂️ **Combining Vector Store with Conditional Logic**
# ───────────────────────────────────────────

print("\n\n------ 🔄🗂️ **Combining Vector Store with Conditional Logic** ------")

# Example 6: **Conditional Logic in Vector Search** 🔄
# - Checks if relevant results were found and displays them.

if len(results) > 0:
    print(f"✅ Found relevant results: {results[:3]}")
else:
    print("❌ No relevant results found.")

print("------ 🔄🗂️ **Combining Vector Store with Conditional Logic** ------\n")


print("\n\n----------------- END -----------------\n")



# Group 2:
# Agents: Building intelligent agents that can perform specific tasks.
# Vector Stores: Using FAISS to store and retrieve documents.
# Document Loaders: Loading documents to be processed by LLMs.
# Custom Chains: Building custom workflows to process data.
# Execution: Running custom functions based on conditions.
# Conditional Logic: Making decisions based on data or query length.
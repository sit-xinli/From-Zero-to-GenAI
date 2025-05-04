import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, load_summarize_chain, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory, VectorStoreRetrieverMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.callbacks.stdout import StdOutCallbackHandler # Simple callback
from langchain.evaluation.qa import QAEvalChain
from langchain_core.documents import Document

# --- Setup ---
# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Ensure your OPENAI_API_KEY is set in your environment or a .env file
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# Initialize LLM and Embeddings
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
embeddings = OpenAIEmbeddings()
print("LLM and Embeddings Initialized.")

# --- Setup Vector Store (Reusing logic from 02_*) ---
# Create a simple document and vector store for examples
document_text = """
LangChain is a framework for developing applications powered by language models.
It enables applications that are data-aware, agentic, and differentiated.
Key components include Models, Prompts, Memory, Indexes, Chains, Agents, and Callbacks.
Data-aware means connecting models to other data sources.
Agentic means allowing models to interact with their environment.
Use cases include chatbots, summarization, Q&A over documents, and more.
LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications.
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
document_chunks = text_splitter.create_documents([document_text])
vector_store = FAISS.from_documents(document_chunks, embeddings)
doc_retriever = vector_store.as_retriever() # Explicit Retriever
print(f"Vector Store and Retriever created with {len(document_chunks)} chunks.")
print("----------------------------------------\n")

# --- 1. Explicit Retrievers & RetrievalQA Chain ---
print("--- 1. Explicit Retrievers & RetrievalQA Chain ---")
# Use the retriever created above directly in a standard chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Options: stuff, map_reduce, refine, map_rerank
    retriever=doc_retriever,
    return_source_documents=True # Optionally return source docs
)

query = "What is LangSmith used for?"
result = qa_chain({"query": query})

print(f"Query: {query}")
print(f"Answer: {result['result']}")
# print(f"Source Documents: {result['source_documents']}")
print("----------------------------------------\n")



# --- 2. Summarization Chain ---
print("--- 2. Summarization Chain ---")
# Example long text for summarization
long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals.
Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
Some popular accounts use the term 'artificial intelligence' to describe machines that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving'.
AI research has been defined as the field of study of intelligent agents. An agent is anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators.
A rational agent is one that acts so as to achieve the best outcome or, when there is uncertainty, the best expected outcome.
The field was founded on the assumption that human intelligence can be so precisely described that a machine can be made to simulate it.
This raises philosophical arguments about the nature of the mind and the ethics of creating artificial beings endowed with human-like intelligence.
These issues have been explored by myth, fiction and philosophy since antiquity.
Some people also consider AI to be a danger to humanity if it progresses unabatedly.
"""
docs_to_summarize = [Document(page_content=long_text)]

# Load a summarization chain (map_reduce is good for long docs)
summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
summary = summary_chain.run(docs_to_summarize)

print(f"Original Text Length: {len(long_text)} characters")
print(f"Summary:\n{summary}")
print("----------------------------------------\n")




# --- 3. Router Chain (LCEL RunnableBranch) ---
print("--- 3. Router Chain (LCEL RunnableBranch) ---")

# Define prompt templates
physics_template = """You are a very smart physics professor. You are great at answering questions about physics in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. You are great at answering math questions.
You are so good because you are able to break down hard problems into their component parts,
answer the component parts, and then put them back together to answer the broader question.

Here is a question:
{input}"""

# Define prompt info list
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    }
]

# --- LCEL Refactoring Start ---

# 1. Create destination chains using LCEL
physics_prompt = ChatPromptTemplate.from_template(physics_template)
math_prompt = ChatPromptTemplate.from_template(math_template)

physics_chain = physics_prompt | llm | StrOutputParser()
math_chain = math_prompt | llm | StrOutputParser()

# Default chain using LCEL
default_prompt = ChatPromptTemplate.from_template("Here's the question: {input}")
default_chain = default_prompt | llm | StrOutputParser()

# 2. Create the router prompt and chain
router_template = """Given the user question below, classify it as either `physics` or `math`.

Do not respond with more than one word.

<question>
{input}
</question>

Classification:"""
router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | llm | StrOutputParser()

# 3. Create the RunnableBranch
# The branch takes the input dictionary {"topic": ..., "input": ...}
# It routes based on "topic" and passes "input" to the chosen chain.
branch = RunnableBranch(
    (lambda x: "physics" in x["topic"].lower(), physics_chain), # Check if router output contains 'physics'
    (lambda x: "math" in x["topic"].lower(), math_chain),     # Check if router output contains 'math'
    default_chain, # Default case
)


# 4. Create the full chain
# It first runs the router_chain to get the topic,
# then passes both the original input and the topic to the branch.
full_chain = {
    "topic": router_chain,
    "input": lambda x: x["input"] # Pass original input through
} | branch

# --- LCEL Refactoring End ---

# Test the new LCEL Router Chain
print("Testing Router Chain with Physics Question:")
physics_question = "What is the formula for escape velocity?"
print(f"Q: {physics_question}")
# Use invoke with a dictionary input for LCEL chains
response_physics = full_chain.invoke({"input": physics_question})
print(f"A: {response_physics}")

print("\nTesting Router Chain with Math Question:")
math_question = "What is the square root of 144?"
print(f"Q: {math_question}")
response_math = full_chain.invoke({"input": math_question})
print(f"A: {response_math}")

print("\nTesting Router Chain with General Question:")
general_question = "What is the capital of France?"
print(f"Q: {general_question}")
response_general = full_chain.invoke({"input": general_question})
print(f"A: {response_general}")

print("----------------------------------------\n")



# --- 4. Advanced Memory Types ---
print("--- 4. Advanced Memory Types ---")

# 4a. ConversationSummaryMemory
print("\n--- 4a. ConversationSummaryMemory ---")
# Instantiate the memory
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history", # Standard key for ConversationSummaryMemory with LLMChain
    input_key="input"         # Make sure input key matches LLMChain expectation
)

# Note: Using deprecated LLMChain here because RunnableWithMessageHistory
# expects a memory object with a `.messages` attribute (like ChatMessageHistory),
# which ConversationSummaryMemory doesn't have.
summary_convo_chain = LLMChain(
    llm=llm,
    # The prompt for the chain itself can be simple, memory handles context
    prompt=ChatPromptTemplate.from_template("Respond to the following input: {input}"),
    memory=summary_memory,
    verbose=True
)

# Run the chain - LLMChain expects keyword arguments or a single positional arg
print(summary_convo_chain.run(input="My favorite color is blue."))
print(summary_convo_chain.run(input="My favorite sport is soccer."))
print(summary_convo_chain.run(input="What is my favorite color?")) # Test memory retrieval

# Check the memory
final_memory_vars = summary_memory.load_memory_variables({})
print("\nFinal Memory:", final_memory_vars)

# 4b. VectorStoreRetrieverMemory
print("\n--- 4b. VectorStoreRetrieverMemory (Example Concept) ---")
# Note: For a real application, use a separate FAISS index for memory.
# Here, we reuse the doc store for simplicity, which isn't ideal.
memory_vector_store = FAISS.from_texts(["Initial memory entry"], embeddings)
memory_retriever = memory_vector_store.as_retriever(search_kwargs=dict(k=1))
vector_memory = VectorStoreRetrieverMemory(retriever=memory_retriever, memory_key="vector_history")

# Add context to memory (normally done within a chain)
vector_memory.save_context({"input": "My dog's name is Sparky"}, {"output": "That's a great name!"})
vector_memory.save_context({"input": "My cat's name is Luna"}, {"output": "Luna is a lovely name for a cat."})

print("Vector Memory State (Conceptual):")
print(vector_memory.load_memory_variables({"prompt": "What is my dog's name?"})) # Retrieval query
print("(Note: Output shows retrieved docs, needs integration into prompt)")
print("----------------------------------------\n")




# --- 5. Callbacks & LangSmith Introduction ---
print("--- 5. Callbacks & LangSmith Introduction ---")
# Simple StdOut Callback
stdout_handler = StdOutCallbackHandler()

# Use the callback in a chain run
qa_chain_with_callback = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=doc_retriever,
    callbacks=[stdout_handler] # Add the handler here
)

print("\nRunning QA Chain with StdOut Callback:")
qa_chain_with_callback({"query": "What does agentic mean in LangChain?"})

print("\nLangSmith: To enable LangSmith tracing:")
print("1. Sign up at https://smith.langchain.com/")
print("2. Create an API Key.")
print("3. Set Environment Variables:")
print("   export LANGCHAIN_TRACING_V2=true")
print("   export LANGCHAIN_API_KEY='YOUR_LANGSMITH_API_KEY'")
print("   export LANGCHAIN_PROJECT='Your Project Name' # Optional")
print("Once set, subsequent LangChain runs will automatically be traced.")
print("----------------------------------------\n")




# --- 6. Evaluation (Simple QAEvalChain) ---
print("--- 6. Evaluation (Simple QAEvalChain) ---")

# Create example evaluation data (query, ground truth answer)
# Renaming 'ground_truth' to 'answer' to match QAEvalChain expectation
eval_examples = [
    {"query": "What is the main topic of the document?", "answer": "LangChain advanced features"},
    {"query": "Explain ConversationSummaryMemory.", "answer": "A memory type that summarizes the conversation history."},
]

# Create predictions (query, actual result from a chain)
# In a real scenario, you'd run your chain (e.g., qa_chain from Section 1)
# on the queries from eval_examples to get these predictions.
# Here, we'll use dummy predictions for demonstration.
predictions = [
    {"query": "What is the main topic of the document?", "result": "The document covers advanced features of LangChain like specialized chains, memory, callbacks, and evaluation."},
    {"query": "Explain ConversationSummaryMemory.", "result": "It's a memory system that keeps a summary of the chat."},
]

# Initialize the evaluation chain
eval_chain = QAEvalChain.from_llm(llm=llm)

# Evaluate the predictions against the examples
print("Running Evaluation...")
eval_results = eval_chain.evaluate(
    eval_examples,
    predictions,
    question_key="query",
    answer_key="answer", # Updated key
    prediction_key="result"
)

print("\nEvaluation Results:")
print(eval_results)
print("(Uses LLM-as-judge. Results interpretation depends on criteria used by the LLM.)")
print("----------------------------------------\n")

print("Advanced LangChain examples finished.")

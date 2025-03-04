# Retrieval-Augmented Generation (RAG) - Enhancing Chatbots with Knowledge Retrieval

## 1. Concept Explanation

### 📚 For a 10-year-old

Imagine you have a robot friend who is very smart but doesn’t remember everything. However, your robot friend has a magic book. Whenever you ask a question, the robot first looks into the magic book for information and then answers you! This way, the robot always gives better answers.

This is how RAG (Retrieval-Augmented Generation) works! The chatbot first searches for relevant knowledge before answering your question, making it smarter.

### 📚 For a 25-year-old

Retrieval-Augmented Generation (RAG) combines information retrieval with generative AI. Instead of relying solely on pre-trained knowledge, an RAG-based chatbot first searches a database (vector store) for relevant information and then generates a response based on the retrieved data. This makes it ideal for answering domain-specific or up-to-date questions.

In this exercise, we build a simple RAG chatbot using FAISS for vector storage and OpenAI’s GPT model for response generation.

---

## 2. Code Breakdown

Below is the code for our RAG-based chatbot:

### 📌 Importing Necessary Libraries

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai
import os
```
- `OpenAIEmbeddings`: Converts text into vector embeddings.
- `FAISS`: A lightweight and efficient vector store.
- `CharacterTextSplitter`: Splits documents into smaller chunks for better retrieval.
- `RetrievalQA`: Creates a retrieval-based chatbot.
- `ChatOpenAI`: The language model for generating responses.
- `openai`: Handles API requests to OpenAI.
- `os`: Used to access environment variables.

### 📌 Setting Up API Key

```python
openai.api_key = os.getenv("OPENAI_API_KEY")
```
- Retrieves the OpenAI API key securely from environment variables.

### 📌 Creating a Sample Knowledge Base

```python
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is in Paris.",
    "Python is widely used for AI development.",
    "The Great Wall of China is a famous landmark."
]
```
- A list of facts to act as our chatbot’s knowledge base.

### 📌 Converting Text into Vector Embeddings

```python
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.create_documents(documents)
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(split_docs, embeddings)
```
- Splits text into smaller chunks.
- Converts text into numerical representations (embeddings).
- Stores embeddings in a FAISS vector database.

### 📌 Setting Up the RAG Chatbot

```python
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=retriever)
```
- Converts the FAISS database into a retriever.
- Uses `RetrievalQA` to integrate retrieval and language generation.

### 📌 Running the Chatbot

```python
print("\n🚀 AI Chatbot with RAG! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run(user_input)
    print(f"AI: {response}\n")
```
- Runs an interactive chatbot session.
- Retrieves relevant facts before generating responses.
- Exits when the user types `exit` or `quit`.

---

### 📌 Testing

`You`: Python is great for? <br />
`AI`: Python is great for AI development.


`You`: The Great Wall of China is ?<br />
`AI`: The Great Wall of China is a famous landmark.

`You`: That country has what other landmark ?<br />
`AI`: I don't know, could you please provide more information or specify which country you are referring to?


### Demo
<a href="https://huggingface.co/spaces/Ganesh-Kunnamkumarath/RAG-Chatbot-Basics" target="_blank"> Demo link </a><br />
Note: API key would have updated.

## 3. Mini-Exercise & Practical Deployment

### 🔹 Exercise:
1. Add more facts to the knowledge base and test if the chatbot retrieves the right information.
2. Modify the chatbot to use a different embedding model and observe the impact on responses.

### 🚀 Deployment:
To make this chatbot accessible online, we will deploy it on Hugging Face Spaces. Stay tuned for the next steps!

# run below line to install necessary depencenies
# pip install faiss-cpu gradio openai langchain langchain_openai langchain_community


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai
import os

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sample Knowledge Base
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is in Paris.",
    "Python is widely used for AI development.",
    "The Great Wall of China is a famous landmark."
]

# Convert text into vector embeddings
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.create_documents(documents)
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(split_docs, embeddings)

# Create RAG-based chatbot
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=retriever)

print("\n🚀 AI Chatbot with RAG! Type 'exit' to quit.\n")
# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run(user_input)
    print(f"AI: {response}\n")



#Testing

# python 3.rag_chatbot.py

# You: Python is great for?
# AI: Python is great for AI development.

# You: The Great Wall of China is ?
# AI: The Great Wall of China is a famous landmark.

# You: That country has what other landmark ?
# AI: I don't know, could you please provide more information or specify which country you are referring to?
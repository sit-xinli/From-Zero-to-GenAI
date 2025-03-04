import gradio as gr
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
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

def chatbot_response(user_input):
    return qa_chain.run(user_input)

# Gradio UI
iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="RAG Chatbot")

if __name__ == "__main__":
    iface.launch()

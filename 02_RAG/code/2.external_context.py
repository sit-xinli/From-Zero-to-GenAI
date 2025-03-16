# pip install fitz && python 2.external_context.py
# pip install PyMuPDF && python 2.external_context.py


import fitz  # PyMuPDF for PDF extraction
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Function to extract text from a PDF
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# Function to load text from a .txt file
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Load data from a file (Update with your actual file path)
file_path = "data/sample.pdf"  # 2023 ICC World cup

if file_path.endswith(".pdf"):
    document_text = load_pdf(file_path)
elif file_path.endswith(".txt"):
    document_text = load_text(file_path)
else:
    raise ValueError("Unsupported file format. Use a .pdf or .txt file.")


# Same as before
# Convert text into vector embeddings
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.create_documents([document_text])
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(split_docs, embeddings)

# Create RAG-based chatbot
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=retriever)

print("\n🚀 AI Chatbot with RAG! Type 'exit' to quit.\n")
# Enter an infinite loop to interact with the user
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run(user_input)
    print(f"AI: {response}\n")


'''
Testing

You: Python is widely used for what?
AI: Python is widely used in various fields such as data science, machine learning, web development, automation, scientific computing, and more.

🔥 🔥 🔥
You: Who won most matched in Group table
AI: In the group table provided, India won the most matches, winning all 9 matches they played.

🔥 🔥 🔥 🔥 🔥 🔥
You: which players from India did great
AI: Some standout players from India in the ICC Men's Cricket World Cup 2023 were:
- Virat Kohli: Player of the Tournament with 765 runs
- Rohit Sharma: Contributed with 597 runs
- Mohammed Shami: Impressive bowling performances with 24 wickets

These players had great performances throughout the tournament.

'''

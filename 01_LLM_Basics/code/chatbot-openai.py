# run below line to install necessary depencenies
# pip install openai


import openai
import os

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=api_key)

# Function to chat with GPT
def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Change to "gpt-3.5-turbo" if needed // "gpt-4" is the latest model
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

print("\n🚀 AI Chatbot with OPENAI API! Type 'exit' to quit.\n")
# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat_with_gpt(user_input)
    print(f"AI: {response}\n")


#Testing

# python 1.chatbot-openai.py

# You: What is Generative AI?
# AI: Generative AI refers to a type of artificial intelligence that focuses on generating new cont...
import openai
import os
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client (New API)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define zero-shot prompts
prompts = [
    "Translate 'Hello, how are you?' into French.",
    "What is the capital of Japan?",
    "Summarize this paragraph: Artificial Intelligence is transforming industries by automating tasks, improving decision-making, and enhancing user experiences.",
    "Classify the sentiment of this sentence: 'I absolutely love this product!'",
    "Extract names from this text: 'Alice and Bob went to the market and met Charlie.'"
]

# Function to call OpenAI API (Updated syntax)
def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Change to "gpt-4" if available
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Run all prompts
for i, prompt in enumerate(prompts, 1):
    print(f"Prompt {i}: {prompt}")
    print("Response:", get_response(prompt))
    print("-" * 50)

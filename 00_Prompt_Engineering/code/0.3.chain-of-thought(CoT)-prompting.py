import openai
import os
from dotenv import load_dotenv

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=api_key)

# Chain-of-Thought (CoT) Prompt
prompt1 = """
Solve the following math problem step by step.

Problem: If a farmer has 3 fields, and each field produces 250 apples, how many apples does the farmer have in total?

Think step by step before answering.
"""

prompt2 = """
John, Mary, and Alex are in a race.  
- John finishes before Alex.  
- Mary finishes after John but before Alex.  

Who finishes last?

Think step by step before answering.
"""

# Get AI Responses
response1 = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Use "gpt-4o-mini" for better CoT reasoning
    messages=[{"role": "user", "content": prompt1}]
)
response2 = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Use "gpt-4o-mini" for better CoT reasoning
    messages=[{"role": "user", "content": prompt2}]
)

print("---------------------")
# Print AI Response
print(prompt1)
print("AI Response:", response1.choices[0].message.content)
print("\n---------------------\n")

print("\n---------------------")

print(prompt2)
print("AI Response:", response2.choices[0].message.content)
print("\n---------------------\n")

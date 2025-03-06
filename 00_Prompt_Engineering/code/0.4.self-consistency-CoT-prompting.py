import openai
import os
from collections import Counter
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Self-Consistency CoT Prompt
prompt = """
Solve the following math problem step by step.

Problem: A bakery sells 12 cupcakes per box. If a customer buys 5 boxes, how many cupcakes does the customer have in total?

Think step by step before answering.
"""

# Generate multiple responses
num_samples = 3  # Number of independent answers
answers = []

for _ in range(num_samples):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use "gpt-4" for better CoT
        messages=[{"role": "user", "content": prompt}]
    )
    answers.append(response.choices[0].message.content)

# Determine most common answer
most_common_answer = Counter(answers).most_common(1)[0][0]

# Print all responses and the final selected answer
print("All Responses:", answers)
print("\n Final Answer (Most Consistent):", most_common_answer)

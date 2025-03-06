import openai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Few-shot Prompt with Examples
prompt1 = """
Classify the sentiment of the following movie reviews as 'Positive' or 'Negative'.

Example 1:
Review: "The movie was an absolute masterpiece. The storytelling was gripping, and the characters were unforgettable."
Sentiment: Positive

Example 2:
Review: "I regret watching this movie. It was too long, boring, and the acting was terrible."
Sentiment: Negative

Example 3:
Review: "The cinematography was stunning, but the plot was weak and predictable."
Sentiment:
"""

prompt2 = """
Translate the following sentences from English to French.

Example 1:
English: "Good morning!"
French: "Bonjour!"

Example 2:
English: "How are you?"
French: "Comment ça va?"

Example 3:
English: "Where is the nearest train station?"
French:
"""

# Get AI Response
response1 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt1}]
)

# Get AI Response
response2 = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Change to "gpt-4" if available
    messages=[{"role": "user", "content": prompt2}]
)


# Print AI Responses
print("AI Response for Prompt 1:", response1.choices[0].message.content)

print("AI Response for Prompt 2:", response2.choices[0].message.content)
# LLM Basics - Building a Chatbot with OpenAI API

## 1. Concept Explanation

### 📖 For a 10-year-old

Imagine you have a magic book that knows everything! You ask it a question, and it gives you an answer. This magic book is like an AI chatbot. We use a special key (API key) to talk to it, and it responds based on what it has learned. The chatbot understands our words and gives helpful answers, just like a very smart friend!

### 📖 For a 25-year-old

Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like responses. OpenAI’s API provides access to models like GPT-4o-mini, allowing developers to create AI-powered applications. In this exercise, we build a simple chatbot using Python and OpenAI’s API, where user inputs get processed by the model to generate intelligent responses.

---

## 2. Code Breakdown

Below is the code for a simple chatbot using OpenAI’s API:

### 📌 Importing Necessary Libraries

```python
import openai
import os
```

- `openai`: This library allows us to interact with OpenAI’s models.
- `os`: Used to access environment variables securely.

### 📌 Retrieving the API Key

```python
api_key = os.getenv("OPENAI_API_KEY")
```

- The API key is stored in an environment variable for security reasons.
- Instead of hardcoding the key in the script, we retrieve it dynamically.

### 📌 Initializing the OpenAI Client

```python
client = openai.OpenAI(api_key=api_key)
```

- This line creates an instance of the OpenAI client using the API key, allowing us to send requests to OpenAI’s servers.

### 📌 Function to Interact with GPT

```python
def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Change to "gpt-3.5-turbo" if needed
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

- This function takes user input (`prompt`) and sends it to OpenAI’s chatbot.
- The model processes the input and generates a response.
- The response is then returned as text.

### 📌 Running the Chatbot in a Loop

```python
print("\n🚀 AI Chatbot with OPENAI API! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat_with_gpt(user_input)
    print(f"AI: {response}\n")
```

- The chatbot runs in a loop, continuously accepting user input.
- If the user types `exit` or `quit`, the chatbot stops.
- Otherwise, it processes the user’s message and prints the AI’s response.

### Demo link
<a href="https://huggingface.co/spaces/Ganesh-Kunnamkumarath/01_LLM_Basics" target="_blank">1_LLM_Basics</a> <br />
Note: API key would have been updated.


### 📌 Testing

`You`: What is Generative AI?<br />
`AI`: Generative AI refers to a type of artificial intelligence that focuses on generating new content, data, or information by learning patterns from existing datasets. Unlike traditional AI, which typically analyzes and makes predictions based on data, generative AI is capable of creating original outputs, such as text, images, music, and even video.....

---

## 3. Mini-Exercise & Practical Deployment

### 🔹 Exercise:

1. Change the chatbot to use `"gpt-3.5-turbo"` instead of `"gpt-4o-mini"` and observe the responses.
2. Modify the chatbot to remember past conversations (Hint: Maintain a message history in a list).

### 🚀 Deployment:

To make this chatbot accessible online, we will deploy it on Hugging Face Spaces. Stay tuned for the next steps!

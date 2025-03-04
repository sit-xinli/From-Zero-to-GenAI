import openai
import os
import gradio as gr

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=api_key)

# Function to chat with GPT
def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Gradio UI
def chatbot(input_text):
    return chat_with_gpt(input_text)

interface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="AI Chatbot")

# Launch app
if __name__ == "__main__":
    interface.launch()

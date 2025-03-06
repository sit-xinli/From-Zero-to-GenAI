import gradio as gr
import openai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Prompt templates
def zero_shot_prompt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def few_shot_prompt(example, prompt):
    full_prompt = example + "\n" + prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content


def chain_of_thought_prompt(prompt):
    full_prompt = f"Think step by step before answering.\n{prompt}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content


def self_consistency_prompt(prompt, num_samples=3):
    from collections import Counter
    answers = []
    for _ in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answers.append(response.choices[0].message.content)
    
    most_common_answer = Counter(answers).most_common(1)[0][0]
    return most_common_answer


def generate_response(technique, prompt, example=""):
    if technique == "Zero-Shot Prompting":
        return zero_shot_prompt(prompt)
    elif technique == "Few-Shot Prompting":
        return few_shot_prompt(example, prompt)
    elif technique == "Chain-of-Thought (CoT) Prompting":
        return chain_of_thought_prompt(prompt)
    elif technique == "Self-Consistency CoT Prompting":
        return self_consistency_prompt(prompt)
    return "Invalid technique selected."


# Function to clear inputs when switching techniques
def clear_inputs(technique):
    return gr.update(value=""), gr.update(value=""), gr.update(value="")


def ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🚀 Prompt Engineering Techniques")
        
        technique = gr.Radio(
            ["Zero-Shot Prompting", "Few-Shot Prompting", "Chain-of-Thought (CoT) Prompting", "Self-Consistency CoT Prompting"],
            label="Select Prompting Technique"
        )
        
        example = gr.Textbox(label="Few-Shot Examples (Only needed for Few-Shot)", placeholder="Provide example cases here")
        prompt = gr.Textbox(label="Enter your prompt")
        output = gr.Textbox(label="AI Response", interactive=False)
        button = gr.Button("Generate Response")
        
        # When the button is clicked, generate the response
        button.click(generate_response, inputs=[technique, prompt, example], outputs=output)

        # When the technique is changed, clear inputs and output
        technique.change(clear_inputs, inputs=[technique], outputs=[example, prompt, output])

    return demo


if __name__ == "__main__":
    ui().launch()

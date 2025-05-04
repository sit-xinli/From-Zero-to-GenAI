from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
import time

# ───────────────────────────────────────────
# 🚀 **Initializing ChatOpenAI LLM**
# ───────────────────────────────────────────
# - Model: gpt-3.5-turbo
# - Temperature: 0.7 (balanced creativity)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

print("------------------------------------START------------------------------------\n")


# ───────────────────────────────────────────
# 🖼️✨ **Multi-Modal - Image Description**
# ───────────────────────────────────────────
print("------ 🤖🔧 **Multi-Modal - Image Description** ------\n")
# Simulating an image description task.
image_description = "A picture of a cat sitting on a sofa."
text_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["description"], template="Describe this image: {description}"))
image_analysis = text_chain.run(image_description)
print(f"Image Description: {image_analysis}\n")
print("------ 🤖🔧 **Multi-Modal - Image Description** ------\n")




# ───────────────────────────────────────────
# 🌐🔍 **LLM Integration - Search**
# ───────────────────────────────────────────
print("------ 🤖🔧 **LLM Integration - Search** ------\n")
# Using DuckDuckGo to fetch search results.
search = DuckDuckGoSearchResults()
search_results = search.run("LangChain documentation")
print(f"Search Results: {search_results}\n")
print("------ 🤖🔧 **LLM Integration - Search** ------\n")




# ───────────────────────────────────────────
# 🔄📄 **Data Augmentation - Text Variations**
# ───────────────────────────────────────────
print("------ 🤖🔧 **Data Augmentation - Text Variations** ------\n")
# Generating paraphrased variations of sentences.
original_sentences = ["The weather is great today.", "It's a sunny day."]
augmented_data = [
    llm.invoke([{"role": "user", "content": f"Paraphrase: {sentence}"}]).content
    for sentence in original_sentences
]
print(f"Augmented Data: {augmented_data}\n")
print("------ 🤖🔧 **Data Augmentation - Text Variations** ------\n")




# ───────────────────────────────────────────
# 🚨⚙️ **Webhooks and Event Handling**
# ───────────────────────────────────────────
print("------ 🤖🔧 **Webhooks and Event Handling** ------\n")
# Handling event-based tasks.
def trigger_event(event):
    if event == "new_query":
        return "Event: A new query has been received, processing..."
    return "Event: Unknown."

# Simulating an event trigger
event_response = trigger_event("new_query")
print(f"Event Response: {event_response}\n")
print("------ 🤖🔧 **Webhooks and Event Handling** ------\n")





# ───────────────────────────────────────────
# 🎤🎶 **Streaming - Real-time Model Output**
# ───────────────────────────────────────────

# Start of streaming the model’s response token by token.
print("Streaming Model Response: ", end="", flush=True)
for token in llm.stream([{"role": "user", "content": "Tell me a joke"}]):
    print(f"{token.content}", end="", flush=True)  # Continue printing tokens on the same line
    time.sleep(0.5)

print("\n\n------ 🤖🔧 **End of Streaming - Real-time Model Output** ------\n")



print("------------------------------------END------------------------------------")



# Group 3:
# Multi-Modal: Integrating text and image data.
# LLM Integrations: Combining LLMs with external tools (e.g., search engines).
# Data Augmentation: Generating variations of input text.
# Evaluation: Assessing the quality of generated output.
# Webhooks: Handling event-driven tasks with webhooks.
# Streaming: Streaming tokens in real-time from LLMs

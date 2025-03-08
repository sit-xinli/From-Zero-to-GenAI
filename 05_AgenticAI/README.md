# Agentic AI Chatbot

## 🤖 What is Agentic AI?

### 🧒 For a 10-year-old:
Imagine you have a super-smart robot friend. Instead of just answering questions, this robot can think, plan, and use different tools to find answers for you. If you ask about dinosaurs, it might search online, check Wikipedia, and even do some math if needed—all on its own!

### 🧑 For a 25-year-old:
Agentic AI refers to AI systems that go beyond passive response generation. These agents actively **reason, plan, and use external tools** to solve problems dynamically. Unlike traditional chatbots, agentic AI can:
- Retrieve live information from the web
- Perform calculations
- Run code dynamically
- Utilize memory for contextual conversations

## 🌍 Real-World Use Cases of Agentic AI
Agentic AI is widely used across industries:
1. **Financial Analysis** 📊 → AI agents analyze market trends, calculate investment risks, and fetch real-time stock prices.
2. **Customer Support** 💬 → Virtual assistants provide personalized help by searching databases and FAQs.
3. **Medical Diagnosis** 🏥 → AI agents process symptoms, fetch medical references, and recommend possible treatments.
4. **Research Assistance** 📚 → AI tools scan academic papers, summarize findings, and perform calculations.

## 📝 About the Code (app.py)
This project builds an **Agentic AI chatbot** using **LangChain** and **GPT-4o-mini**. It integrates four powerful tools:

1️⃣ **DuckDuckGo Search** → Fetches live web search results.  
2️⃣ **Wikipedia Query** → Retrieves structured knowledge from Wikipedia.  
3️⃣ **Math Calculator** → Performs safe mathematical evaluations.  
4️⃣ **Python Code Executor** → Runs small Python scripts dynamically.  

The chatbot is memory-enabled, meaning it can remember past interactions within a session.

## 🛠️ Installation & Setup
### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Set Up API Keys
You need an OpenAI API key. Set it as an environment variable:
```sh
export OPENAI_API_KEY="your_api_key_here"
```

### 3️⃣ Run the Application
```sh
streamlit run app.py  # If using Streamlit
gradio app.py          # If using Gradio
```

## 🚀 How It Works
1. You enter a query (e.g., "Who won the FIFA World Cup in 2018?").
2. The agent decides which tool to use (Wikipedia or Web Search).
3. The tool fetches the answer.
4. The agent returns the final response with context.

This makes the chatbot **more autonomous and intelligent** than simple AI models. 🎯


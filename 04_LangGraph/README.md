# 🚀 AI Chatbot with LangGraph

## 📌 Introduction
This project demonstrates an AI chatbot built using **LangChain** and **LangGraph**. The chatbot can answer general questions and provide predefined responses for specific topics like weather and news. The chatbot is deployed as a **Streamlit** web app.

---

## 🧠 Understanding LangGraph

LangGraph is a powerful framework that helps design structured workflows for AI-powered applications. It allows us to define step-by-step logic for processing user inputs, deciding what response to give, and handling different types of interactions.

### 🏫 Explain Like I'm 10
Imagine you're playing a game where you ask a robot questions. This robot has different "paths" it can follow based on what you ask. If you ask about the weather, it gives you a weather message. If you ask about news, it tells you where to find news. LangGraph helps the robot **decide which path to take** based on what you ask!

### 🎓 Explain Like I'm 25
LangGraph is a **graph-based computation framework** that structures AI interactions into **nodes** (tasks) and **edges** (decision paths). It enables AI workflows by guiding inputs through a structured pipeline. This is particularly useful in chatbots, recommendation systems, and AI-powered assistants where logic-driven responses are necessary.

---

## 🌍 Real-World Use Cases of LangGraph
LangGraph is useful in various applications, including:
1. **Customer Support Bots**: Directs users to different support sections based on their queries.
2. **Automated Medical Assistants**: Routes patient inquiries to different AI models based on symptoms.
3. **Interactive Storytelling**: Creates AI-driven stories where the plot changes based on user choices.
4. **AI Tutors**: Helps students by selecting different teaching strategies depending on their questions.

---

## 🛠️ Installation Guide
To run this chatbot locally, follow these steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/langgraph-chatbot.git
cd langgraph-chatbot
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up OpenAI API Key
Export your API key (replace `your-api-key` with an actual key):
```bash
export OPENAI_API_KEY='your-api-key'
```
For Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=your-api-key
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 📜 Code Explanation
The chatbot uses **LangGraph** to create an AI-powered decision-making system. Here’s how it works:

### ✅ **Step 1: Define the State**
- `ChatState` stores the conversation history.

### ✅ **Step 2: Define Nodes (Chatbot Logic)**
- `ask_llm`: Calls OpenAI’s GPT-4 to generate responses.
- `check_topic`: Determines if the query is about **weather, news, or general topics**.
- `weather_response`: Returns a predefined response for weather queries.
- `news_response`: Returns a predefined response for news queries.

### ✅ **Step 3: Connect the Workflow**
- The chatbot decides the response path using `check_topic`.
- If it's a general query, it calls `ask_llm`.
- If it's about weather/news, it provides a predefined response.

### ✅ **Step 4: Build the Streamlit UI**
- Accepts user input via a text box.
- Displays conversation history with labels for **You, AI, and Predefined Response**.
- Uses a "Send" button for input submission.

---

## 🚀 Running the Chatbot
Once the app is running, you can enter queries and interact with the AI chatbot.

Example:
```plaintext
You: What's the weather today?
Predefined Response: I can't fetch live weather, but check a weather website!
```

---

## 📚 References
See **references.md** for detailed learning resources on LangChain and LangGraph.

---

Now you're ready to explore LangGraph-powered AI chatbots! 🚀


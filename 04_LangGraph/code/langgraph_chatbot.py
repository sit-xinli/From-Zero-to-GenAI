import openai
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

# ✅ Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Define State Schema for chatbot
class ChatState(TypedDict):
    messages: List[BaseMessage]

# ✅ Chatbot logic
def ask_llm(state: ChatState):
    """Generates AI responses based on user input."""
    model = ChatOpenAI(model="gpt-4")

    # Ensure correct message format
    valid_messages = [msg if isinstance(msg, BaseMessage) else HumanMessage(content=msg["content"]) for msg in state["messages"]]

    # Get AI response
    response = model.invoke(valid_messages)

    # Convert AI response into AIMessage
    print("\n🚀 n🚀 n🚀 n🚀 \\n")
    print(response.content)
    ai_message = AIMessage(content=response.content)

    # ✅ Append response correctly
    return {"messages": state["messages"] + [ai_message]}

def check_topic(state: ChatState):
    """Routes conversation based on topic."""
    user_input = state["messages"][-1].content.lower()

    if "weather" in user_input:
        return "weather"
    elif "news" in user_input:
        return "news"
    else:
        return "general"

def weather_response(state: ChatState):
    """Custom response for weather-related questions."""
    ai_message = AIMessage(content="I can't fetch live weather, but check a weather website!")
    return {"messages": state["messages"] + [ai_message]}

def news_response(state: ChatState):
    """Custom response for news-related questions."""
    ai_message = AIMessage(content="I can't fetch live news, but check a news website!")
    return {"messages": state["messages"] + [ai_message]}

# ✅ Create structured AI workflow
workflow = StateGraph(ChatState)  # ✅ Fix applied

# ✅ Define chatbot flow
workflow.add_node("chat", ask_llm)
workflow.add_node("weather", weather_response)
workflow.add_node("news", news_response)

# ✅ Define decision logic
workflow.add_conditional_edges("chat", check_topic, {"weather": "weather", "news": "news", "general": END})
workflow.add_edge("weather", END)
workflow.add_edge("news", END)

# ✅ Set start node & finalize graph
workflow.set_entry_point("chat")
app = workflow.compile()

# ✅ Interactive Chat
print("\n🚀 AI Chatbot with Decision Flow! Type 'exit' to quit.\n")
import streamlit as st

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are an AI assistant.")]

st.title("🚀 AI Chatbot with Decision Flow!")
user_input = st.text_input("You: ")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    response = app.invoke({"messages": st.session_state.chat_history})
    st.session_state.chat_history.extend(response["messages"])

    st.write(f"AI: {response['messages'][-1].content}")


# Testing

# Q1. How is the weather today in Bnagalore?
# Q2. What is the latest news on AI?
# Q3. What is 233 * 444
# Q4. What is General AI?

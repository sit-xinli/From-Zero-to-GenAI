import openai
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
import streamlit as st

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
    ai_message = AIMessage(content="[Predefined Response] I can't fetch live weather, but check a weather website!")
    return {"messages": state["messages"] + [ai_message]}

def news_response(state: ChatState):
    """Custom response for news-related questions."""
    ai_message = AIMessage(content="[Predefined Response] I can't fetch live news, but check a news website!")
    return {"messages": state["messages"] + [ai_message]}

# ✅ Create structured AI workflow
workflow = StateGraph(ChatState)

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

# ✅ Streamlit UI
st.title("🚀 AI Chatbot with Decision Flow!")


st.write("Sample Queries: `How is the weather in Bangalore today`, `Whats the latest news on technology?`")
st.write("Sample Queries: `What is the capital of France?`, `Tell me a joke!`")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are an AI assistant.")]

user_input = st.text_input("You:", key="user_input")
if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        response = app.invoke({"messages": st.session_state.chat_history})
        st.session_state.chat_history.extend(response["messages"])

        # Display user input
        st.write(f"**You:** {user_input}")
        st.write("-----------------------------------------------------------------")
        # Display AI and predefined responses with labels
        for msg in response["messages"]:
            if isinstance(msg, AIMessage):
                if msg.content.startswith("[Predefined Response]"):
                    st.write(f"**Predefined Response:** {msg.content.replace('[Predefined Response] ', '')}")
                else:
                    st.write(f"**AI:** {msg.content}")
        st.write("-----------------------------------------------------------------")

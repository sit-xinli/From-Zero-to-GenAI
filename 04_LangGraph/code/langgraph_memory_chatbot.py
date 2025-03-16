import openai
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.memory import ConversationBufferMemory

# ✅ Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Define State Schema with Memory
class ChatState(TypedDict):
    messages: List[BaseMessage]
    memory: ConversationBufferMemory

# ✅ Initialize Conversation Memory
memory = ConversationBufferMemory(return_messages=True)

# ✅ Chatbot logic with memory
def ask_llm(state: ChatState):
    """Generates AI responses using memory."""
    model = ChatOpenAI(model="gpt-4o-mini")

    # Retrieve past messages from memory
    conversation_history = state["memory"].load_memory_variables({})["history"]

    # Ensure correct format
    valid_messages = conversation_history + state["messages"]

    # Get AI response
    response = model.invoke(valid_messages)

    # Store latest message in memory
    ai_message = AIMessage(content=response.content)
    state["memory"].save_context({"input": state["messages"][-1].content}, {"output": ai_message.content})

    # ✅ Append response correctly
    return {"messages": state["messages"] + [ai_message], "memory": state["memory"]}

# ✅ Decision-making logic
def check_topic(state: ChatState):
    """Routes conversation based on past messages and topic."""
    user_input = state["messages"][-1].content.lower()

    # Recall past messages from memory
    past_conversations = state["memory"].load_memory_variables({})["history"]

    if "weather" in user_input:
        return "weather"
    elif "news" in user_input:
        return "news"
    elif "recall" in user_input:  # Custom command to recall memory
        return "recall"
    else:
        return "general"

# ✅ Custom Responses
def weather_response(state: ChatState):
    """Custom response for weather-related questions."""
    ai_message = AIMessage(content="I can't fetch live weather, but check a weather website!")
    return {"messages": state["messages"] + [ai_message]}

def news_response(state: ChatState):
    """Custom response for news-related questions."""
    ai_message = AIMessage(content="I can't fetch live news, but check a news website!")
    return {"messages": state["messages"] + [ai_message]}

# ✅ Recall past conversations
def recall_memory(state: ChatState):
    """Retrieves and summarizes past conversations."""
    past_conversations = state["memory"].load_memory_variables({})["history"]

    if not past_conversations:
        return {"messages": state["messages"] + [AIMessage(content="I don't remember anything yet!")]}

    # Format past conversations for display
    recall_text = "\n".join([msg.content for msg in past_conversations[-5:]])  # Show last 5 messages
    return {"messages": state["messages"] + [AIMessage(content=f"Here's what I remember:\n{recall_text}")]}

# ✅ Create structured AI workflow with memory
workflow = StateGraph(ChatState)

# ✅ Define chatbot flow with memory
workflow.add_node("chat", ask_llm)
workflow.add_node("weather", weather_response)
workflow.add_node("news", news_response)
workflow.add_node("recall", recall_memory)  # ✅ New recall function

# ✅ Define decision logic with memory
workflow.add_conditional_edges("chat", check_topic, {"weather": "weather", "news": "news", "recall": "recall", "general": END})
workflow.add_edge("weather", END)
workflow.add_edge("news", END)
workflow.add_edge("recall", END)

# ✅ Set start node & finalize graph
workflow.set_entry_point("chat")
app = workflow.compile()

# ✅ Interactive Chat
print("\n🚀 AI Chatbot with Memory + LangGraph! Type 'exit' to quit.\n")
chat_history = [SystemMessage(content="You are an AI assistant.")]
state = {"messages": chat_history, "memory": memory}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    if user_input.lower() in ["recall", "memory", "history"]:
        for message in state["messages"]:
            print(f"{message.__class__.__name__}: {message.content}")
        break
    state["messages"].append(HumanMessage(content=user_input))
    response = app.invoke(state)
    state["messages"].extend(response["messages"])

    print(f"AI: {response['messages'][-1].content}\n")

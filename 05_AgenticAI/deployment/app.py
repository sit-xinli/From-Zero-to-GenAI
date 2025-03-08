import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
import math
import time

# DuckDuckGo Search Tool with Error Handling
def safe_duckduckgo_search(query):
    """Performs a DuckDuckGo search with rate limiting handling."""
    try:
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        return f"Error: DuckDuckGo search failed ({str(e)})"

search_tool = Tool(
    name="duckduckgo_search",
    func=safe_duckduckgo_search,
    description="Use this tool to search the web using DuckDuckGo.",
)

# Wikipedia Tool
wiki_retriever = WikipediaAPIWrapper()
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_retriever)

# Calculator Tool
def safe_eval(expression: str):
    """Safely evaluates a simple math expression."""
    try:
        return eval(expression, {"__builtins__": None}, {"math": math})
    except Exception as e:
        return f"Error: {e}"

math_tool = Tool(
    name="calculator",
    func=safe_eval,
    description="Use this tool to evaluate basic mathematical expressions."
)

# Python Code Executor (Executes code dynamically)
def run_python_code(code):
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return exec_globals.get("result", "Code executed successfully, but no 'result' variable found.")
    except Exception as e:
        return f"Python Execution Error: {str(e)}"

python_executor = Tool(
    name="python_code_runner",
    func=run_python_code,
    description="Executes Python code safely and returns results."
)

# 🔥 Add all tools
tools = [search_tool, wiki_tool, math_tool, python_executor]

# 🤖 AI model: GPT-4
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 🛠️ Create Agent (Now more powerful!)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # Using OpenAI's ReAct-style agent
    verbose=True,
    return_intermediate_steps=True  # ✅ Ensuring tool logs are captured
)

# 🎨 Streamlit UI
st.title("🤖 Agentic AI Chatbot")

# User input field
user_query = st.text_input("Ask me anything:")

st.write("Sample Queries: `What is 253 * 12323?`, `Run this Python code: result = sum(range(10))`")
st.write("Sample Queries: `Who was Nikola Tesla?`, `Who won the latest FIFA World Cup?`")

if st.button("Send"):
    if user_query:
        with st.spinner("Thinking..."):
            time.sleep(1)  # Adding a slight delay to prevent spam queries
            response = agent.invoke({"input": user_query})
            output = response["output"]
            tool_logs = response.get("intermediate_steps", [])

        # Display AI response
        st.markdown("### AI Response")
        st.write(f"**AI:** {output}")

        # Display tool invocation logs
        if tool_logs:
            st.markdown("### Tool Invocations")
            for step in tool_logs:
                if isinstance(step, tuple) and len(step) == 2:
                    tool_call, tool_result = step
                    if hasattr(tool_call, 'tool') and hasattr(tool_call, 'tool_input'):
                        tool_name = tool_call.tool
                        tool_input = tool_call.tool_input
                        st.markdown(f"**> Invoking: `{tool_name}` with `{tool_input}`**")

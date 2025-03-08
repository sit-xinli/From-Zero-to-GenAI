from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
import math

# DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchRun()

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

# 💾 Memory for conversation history
memory = ConversationBufferMemory(return_messages=True)

# 🛠️ Create Agent (Now more powerful!)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # Using OpenAI's ReAct-style agent
    memory=memory,
    verbose=True
)

# 🏃 Run the chatbot loop
while True:
    query = input("Ask me anything: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = agent.invoke({"input": query})
    print("\n🤖 AI:", response["output"])


"""

Math Calculation
👤: What is 253 * 12323?
> Entering new AgentExecutor chain...
> Invoking: `calculator` with `253 * 12323`
> 3117719   The result of 253 multiplied by 12323 is 3,117,719.
> Finished chain.


💡 Python Execution
👤: Run this Python code: result = sum(range(10))
> Entering new AgentExecutor chain...
> Invoking: `python_code_runner` with `result = sum(range(10))`
> 45    The result of the Python code is 45.
> Finished chain.


💡 Wikipedia Retrieval
👤: Who was Nikola Tesla?
> Entering new AgentExecutor chain...
> Invoking: `wikipedia` with `{'query': 'Nikola Tesla'}`
Page: Nikola Tesla
Summary: Nikola Tesla (; Serbian Cyrillic: Никола Тесла [nǐkola têsla]...

Page: Belgrade Nikola Tesla Airport
Summary: Belgrade Nikola Tesla Airport (Serbian: Аеродром Никола...

> Finished chain.


💡 Live Web Search
👤: Who won the latest FIFA World Cup?
> Entering new AgentExecutor chain...
> Invoking: `duckduckgo_search` with `{'query': 'latest FIFA World Cup winner'}`
This is a complete list of all the FIFA World Cup winners since its start...
> Finished chain.
"""

# Invoking is the main word to look for in the output. It shows the tool being used to answer the query.

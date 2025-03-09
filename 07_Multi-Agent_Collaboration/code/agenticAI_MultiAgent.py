from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# 📌 Define System State (How information is passed between agents)
class ResearchState(TypedDict):
    query: str
    research_results: List[str]
    analysis_summary: str
    fact_check_feedback: str
    final_report: str

# 🛠️ Initialize Tools
search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# 🤖 Model: GPT-4o
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 💾 Memory
memory = ConversationBufferMemory(return_messages=True)

# 🔹 Research Agent
def research_agent(state: ResearchState):
    print("Doing Research...")
    query = state["query"]
    wiki_result = wiki_tool.run(query)
    web_result = search_tool.run(query)
    return {"research_results": [wiki_result, web_result]}

# 🔹 Analysis Agent
def analysis_agent(state: ResearchState):
    print("Doing Analysis...")
    research_data = "\n".join(state["research_results"])
    prompt = f"Summarize the key insights from the following research:\n\n{research_data}"
    summary = llm.invoke(prompt).content
    return {"analysis_summary": summary}

# 🔹 Fact-Checker Agent
def fact_checker_agent(state: ResearchState):
    print("Doing Fact-Checking...")
    summary = state["analysis_summary"]
    prompt = f"Fact-check the following summary and highlight any inconsistencies:\n\n{summary}"
    fact_check_result = llm.invoke(prompt).content
    return {"fact_check_feedback": fact_check_result}

# 🔹 Report Generator Agent
def report_generator(state: ResearchState):
    print("Generating Final Report...")
    summary = state["analysis_summary"]
    fact_check = state["fact_check_feedback"]
    prompt = f"Generate a well-structured final report with fact-check notes:\n\nSummary:\n{summary}\n\nFact-Check Notes:\n{fact_check}"
    final_report = llm.invoke(prompt).content
    return {"final_report": final_report}

# 📌 Define Multi-Agent Graph
workflow = StateGraph(ResearchState)
workflow.add_node("Research", research_agent)
workflow.add_node("Analysis", analysis_agent)
workflow.add_node("Fact-Check", fact_checker_agent)
workflow.add_node("Report", report_generator)

# Define execution flow
workflow.add_edge("Research", "Analysis")
workflow.add_edge("Analysis", "Fact-Check")
workflow.add_edge("Fact-Check", "Report")
workflow.add_edge("Report", END)

# Set entry point
workflow.set_entry_point("Research")

# 🚀 Compile Graph
research_graph = workflow.compile()

# 🏃 Run the Multi-Agent System
query = input("Enter your research topic: ")
state = {"query": query}
output = research_graph.invoke(state)

# 📌 Print Final Report
print("\n📝 Final Report:\n", output["final_report"])

"""

> Enter your research topic: Elon Musk

Doing Research...
Doing Analysis...
Doing Fact-Checking...
Generating Final Report...

📝 Final Report:
 **Final Report: Elon Musk - Career, Wealth, and Controversies**

**Summary:**
This report provides a detailed analysis of Elon Musk's career, wealth, and controversies, highlighting his significant impact on technology and business, as well as his polarizing public persona.

1. **Career and Business Ventures**: 
   - Elon Musk is ...
2. **Wealth**: 
   - Musk is one ...
... 

**Fact-Check Notes:**
- The report ...

**Conclusion:**
Elon Musk remains ...

"""
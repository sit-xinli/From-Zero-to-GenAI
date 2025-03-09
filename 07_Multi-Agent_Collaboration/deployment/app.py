import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from duckduckgo_search.exceptions import DuckDuckGoSearchException
import time

# Define System State
class ResearchState(TypedDict):
    query: str
    research_results: List[str]
    analysis_summary: str
    fact_check_feedback: str
    final_report: str

# Initialize Tools
search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Model: GPT-4o
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Research Agent
def research_agent(state: ResearchState):
    print("[LOG] Running Research Agent...")
    query = state["query"]
    
    # Wikipedia Search
    wiki_result = wiki_tool.run(query)
    
    # DuckDuckGo Search with Retry Logic
    web_result = ""
    for _ in range(5):  # Retry up to 5 times
        try:
            web_result = search_tool.run(query)
            break  # Exit loop if successful
        except DuckDuckGoSearchException:
            print("[LOG] Rate limit hit. Retrying in 5 seconds...")
            time.sleep(5)  # Wait and retry

    return {"research_results": [wiki_result, web_result]}


# Analysis Agent
def analysis_agent(state: ResearchState):
    print("[LOG] Running Analysis Agent...")
    research_data = "\n".join(state["research_results"])
    prompt = f"Summarize the key insights from the following research:\n\n{research_data}"
    summary = llm.invoke(prompt).content
    return {"analysis_summary": summary}

# Fact-Checker Agent
def fact_checker_agent(state: ResearchState):
    print("[LOG] Running Fact-Checker Agent...")
    summary = state["analysis_summary"]
    prompt = f"Fact-check the following summary and highlight any inconsistencies:\n\n{summary}"
    fact_check_result = llm.invoke(prompt).content
    return {"fact_check_feedback": fact_check_result}

# Report Generator Agent
def report_generator(state: ResearchState):
    print("[LOG] Running Report Generator Agent...")
    summary = state["analysis_summary"]
    fact_check = state["fact_check_feedback"]
    prompt = f"Generate a well-structured final report with fact-check notes:\n\nSummary:\n{summary}\n\nFact-Check Notes:\n{fact_check}"
    final_report = llm.invoke(prompt).content
    return {"final_report": final_report}

# Define Multi-Agent Graph
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
workflow.set_entry_point("Research")
research_graph = workflow.compile()

# Gradio Interface
def generate_report(topic):
    print("[LOG] Generating Report...")
    state = {"query": topic}
    output = research_graph.invoke(state)
    print("[LOG] Report Generation Complete!")
    return output["final_report"]

demo = gr.Interface(
    fn=generate_report,
    inputs=gr.Textbox(label="Enter Research Topic"),
    outputs=gr.Textbox(label="Generated Report", lines=20),
    title="Multi-Agent Research System",
    description="Enter a topic, and the system will generate a research report with analysis and fact-checking."
)

demo.launch()

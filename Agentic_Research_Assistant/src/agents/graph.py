from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from src.state import ResearchState
from src.agents.planner import planner_agent_node
from src.agents.writer import writer_agent_node
from src.agents.critic import critic_agent_node
from src.tools.web_search import search_web
from src.tools.vector_store import LocalVectorStore
from config import MAX_REVISIONS

def research_agent_node(state: dict) -> dict:
    """
    Research Node: Executes parallel DuckDuckGo web search and ChromaDB vector retrieval.
    """
    search_queries = state.get("search_queries", [])
    status_log = state.get("status_log", [])
    
    status_log.append(f"🌐 Research Agent: Searching live web & local vector store for queries: {search_queries}...")

    # 1. Live Web Search
    web_results = search_web(search_queries)

    # 2. Local ChromaDB Vector Search
    vector_store = LocalVectorStore()
    retrieved_docs = vector_store.search(search_queries)

    return {
        "web_results": web_results,
        "retrieved_docs": retrieved_docs,
        "status_log": status_log
    }

def increment_revision_node(state: dict) -> dict:
    """Helper node to increment revision counter on reflection loop back."""
    current_revisions = state.get("revision_count", 0)
    status_log = state.get("status_log", [])
    status_log.append(f"🔄 Reflection Loop: Revision #{current_revisions + 1} initiated based on Critic Feedback...")
    return {
        "revision_count": current_revisions + 1,
        "status_log": status_log
    }

def finalizer_node(state: dict) -> dict:
    """Finalizer Node: Finalizes report output and appends completion metrics."""
    draft_report = state.get("draft_report", "")
    critic_score = state.get("critic_score", 1.0)
    status_log = state.get("status_log", [])
    
    status_log.append("✅ Finalizer Node: Research completed successfully!")

    final_report = draft_report
    if critic_score < 0.8:
        final_report += f"\n\n---\n> ⚠️ **Quality Note:** Final report generated after maximum reflection iterations. Critic Quality Score: `{critic_score:.2f}/1.00`."

    return {
        "final_report": final_report,
        "status_log": status_log
    }

def should_continue(state: dict) -> Literal["continue_revision", "finalize"]:
    """
    Conditional Routing Edge:
    Checks if Critic Score is sufficient OR if maximum revisions have been reached.
    """
    score = state.get("critic_score", 0.0)
    revisions = state.get("revision_count", 0)

    if score >= 0.8 or revisions >= MAX_REVISIONS:
        return "finalize"
    else:
        return "continue_revision"

def create_research_graph() -> StateGraph:
    """
    Builds and compiles the Multi-Agent Research Assistant StateGraph using LangGraph.
    """
    builder = StateGraph(ResearchState)

    # Add Nodes
    builder.add_node("planner", planner_agent_node)
    builder.add_node("research", research_agent_node)
    builder.add_node("writer", writer_agent_node)
    builder.add_node("critic", critic_agent_node)
    builder.add_node("increment_revision", increment_revision_node)
    builder.add_node("finalizer", finalizer_node)

    # Define Linear Flow Edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "research")
    builder.add_edge("research", "writer")
    builder.add_edge("writer", "critic")

    # Add Conditional Reflection Loop Routing
    builder.add_conditional_edges(
        "critic",
        should_continue,
        {
            "continue_revision": "increment_revision",
            "finalize": "finalizer"
        }
    )
    
    # Connect revision loop back to research
    builder.add_edge("increment_revision", "research")
    builder.add_edge("finalizer", END)

    return builder.compile()

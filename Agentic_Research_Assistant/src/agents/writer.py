import time
from config import get_llm

def writer_agent_node(state: dict) -> dict:
    """
    Writer / Synthesizer Agent Node:
    Aggregates web and vector search results into a comprehensive draft research report with inline citations.
    """
    topic = state["topic"]
    sub_questions = state.get("sub_questions", [])
    web_results = state.get("web_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    critic_feedback = state.get("critic_feedback", "")
    status_log = state.get("status_log", [])
    
    status_log.append("✍️ Writer Agent: Synthesizing facts into a structured research draft...")

    time.sleep(1)  # Pacing for rate limit safety

    # Format context sources with numerical citations
    sources_text = ""
    source_index = 1
    citations_map = []

    for item in web_results:
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        title = item.get("title", "Web Source")
        if snippet:
            sources_text += f"\n[{source_index}] Title: {title}\nURL: {url}\nContent: {snippet}\n"
            citations_map.append(f"[{source_index}] {title} - {url}")
            source_index += 1

    for doc in retrieved_docs:
        content = doc.get("content", "")
        source_name = doc.get("source", "Uploaded Document")
        if content:
            sources_text += f"\n[{source_index}] Source: {source_name}\nContent: {content}\n"
            citations_map.append(f"[{source_index}] Local Document: {source_name}")
            source_index += 1

    feedback_prompt = ""
    if critic_feedback:
        feedback_prompt = f"\nCRITICAL REVIEW FEEDBACK FROM PREVIOUS DRAFT:\n{critic_feedback}\nMake sure to address all noted gaps or errors above.\n"

    prompt = f"""You are a Lead AI Technical Writer.
Your task is to synthesize retrieved facts into a comprehensive, well-structured Markdown Research Report.

Topic: {topic}

Sub-Questions Addressed:
{chr(10).join(f"- {q}" for q in sub_questions)}
{feedback_prompt}
RETRIEVED EVIDENCE & SOURCES:
{sources_text if sources_text else "No external evidence retrieved. Use high-level knowledge base."}

Report Guidelines:
1. Include a clear Title, Executive Summary, Detailed Analysis by Sub-Question, and Conclusion.
2. Use inline numerical citations like [1], [2] corresponding EXACTLY to the source numbers provided above whenever presenting facts.
3. Keep the tone academic, objective, and accurate. Avoid unsupported claims.
4. Conclude with a 'References' section listing all cited sources.
"""

    llm = get_llm(temperature=0.3)
    try:
        response = llm.invoke(prompt)
        draft_report = response.content
    except Exception as e:
        draft_report = f"# Research Report: {topic}\n\n*Error generating draft report: {str(e)}*"

    return {
        "draft_report": draft_report,
        "status_log": status_log
    }

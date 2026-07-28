from config import get_llm

def writer_agent_node(state: dict) -> dict:
    """
    Writer / Synthesizer Agent Node:
    Aggregates web & vector context into a rich Markdown Research Report with inline citations and references table.
    """
    topic = state["topic"]
    sub_questions = state.get("sub_questions", [])
    web_results = state.get("web_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    critic_feedback = state.get("critic_feedback", "")
    status_log = state.get("status_log", [])
    
    status_log.append("✍️ Writer Agent: Synthesizing facts into a structured research report...")

    # Build structured sources index
    sources_text = ""
    source_index = 1
    references_table = "| ID | Source Title | Web Link / Source |\n| :--- | :--- | :--- |\n"

    for item in web_results:
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        title = item.get("title", "Web Source")
        if snippet:
            sources_text += f"\n[{source_index}] Title: {title}\nURL: {url}\nSnippet: {snippet}\n"
            references_table += f"| [{source_index}] | **{title}** | [{url}]({url}) |\n"
            source_index += 1

    for doc in retrieved_docs:
        content = doc.get("content", "")
        source_name = doc.get("source", "Uploaded Document")
        if content:
            sources_text += f"\n[{source_index}] Local Document: {source_name}\nSnippet: {content}\n"
            references_table += f"| [{source_index}] | Local Document: `{source_name}` | Internal Vector Store |\n"
            source_index += 1

    feedback_prompt = ""
    if critic_feedback and "meets" not in critic_feedback.lower():
        feedback_prompt = f"\nCRITICAL REVIEW FEEDBACK FROM PREVIOUS DRAFT:\n{critic_feedback}\nMake sure to address all noted gaps above.\n"

    prompt = f"""You are a Lead AI Systems Technical Writer.
Your goal is to synthesize the retrieved evidence into a comprehensive, publication-grade Markdown Research Report.

Topic: {topic}

Sub-Questions Analyzed:
{chr(10).join(f"- {q}" for q in sub_questions)}
{feedback_prompt}
RETRIEVED SOURCES & EVIDENCE:
{sources_text if sources_text else "No external evidence retrieved. Use general domain knowledge base."}

Format Guidelines:
# [Clear Descriptive Report Title]

## 📌 Executive Summary
Provide a high-level 2-3 paragraph executive summary of key findings.

## 🚀 Key Takeaways
- Highlight 4-5 major technical takeaways using bold key terms.

## 🔬 Detailed Technical Analysis
Break down the research by sub-question headers. Write detailed, academic paragraphs.
Include inline numerical citations like [1], [2] corresponding EXACTLY to the source IDs provided above whenever presenting factual statements.

## 📊 Summary Comparison
Include a Markdown comparison table summarizing key aspects, trade-offs, or trends.

## 🏁 Conclusion
Provide a strategic conclusion and future outlook.

## 📖 References
{references_table if source_index > 1 else "*No external sources cited.*"}
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

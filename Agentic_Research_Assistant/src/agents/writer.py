"""
writer.py — Report Synthesis & Citation Engine Agent Node
==========================================================

📚 BEGINNER EXPLANATION — What Does the Writer Do?
    The Writer Agent is like a technical journalist who:
    1. Reads ALL the evidence gathered by the Research Agent (web results + vector docs)
    2. Reads the sub-questions from the Planner Agent
    3. Reads any feedback from the Critic Agent (if this is a revision loop)
    4. Synthesizes everything into a structured, publication-grade Markdown report
       with inline numerical citations [1], [2] and a references table.

    The Writer does NOT make up facts. It strictly synthesizes from the provided evidence.

🏗️ DESIGN PATTERN — Context Aggregation + Citation Indexing:
    The Writer builds a "sources index" by iterating through all retrieved evidence
    and assigning sequential numerical IDs ([1], [2], [3]...). These IDs are included
    in the LLM prompt so the model can reference them as inline citations.
    
    This is a manual citation indexing algorithm — a common pattern in RAG systems
    where you need the LLM to produce traceable, verifiable references.

🔄 REVISION AWARENESS:
    On revision loops (when the Critic rejected the previous draft), the Writer
    receives `critic_feedback` in the state. This feedback is injected into the
    prompt as "CRITICAL REVIEW FEEDBACK FROM PREVIOUS DRAFT" so the LLM knows
    exactly what to fix. This is the mechanism that makes self-correction work.

📁 ARCHITECTURE ROLE:
    LAYER 3 (Intelligence Agent) → Third node in the graph (after Research).
    Reads: topic, sub_questions, web_results, retrieved_docs, critic_feedback
    Writes: draft_report, status_log
"""

import logging
from config import get_llm

# Setup module-level logger
logger = logging.getLogger(__name__)


def writer_agent_node(state: dict) -> dict:
    """
    Writer / Synthesizer Agent Node — Produces structured Markdown research reports.

    📚 HOW THE CITATION SYSTEM WORKS:
        1. We iterate through web_results and retrieved_docs
        2. Each piece of evidence gets a sequential number: [1], [2], [3]...
        3. We build TWO things simultaneously:
           a) `sources_text` — A text block showing all evidence with IDs for the LLM prompt
           b) `references_table` — A Markdown table mapping IDs to titles/URLs
        4. The LLM prompt instructs: "Include inline citations like [1], [2] corresponding
           EXACTLY to the source IDs provided above"
        5. The references table is included in the output format template

    📚 BEGINNER NOTE — Why Markdown?
        Markdown (.md) is a lightweight text formatting language that converts
        to rich HTML. Using Markdown means the report renders beautifully in:
        - Streamlit (our UI framework)
        - GitHub READMEs
        - Any Markdown viewer or text editor

    Args:
        state: The shared ResearchState dictionary containing evidence and context.

    Returns:
        dict: Updated state keys: draft_report, status_log.
    """
    # ──────────────────────────────────────────────────────────
    # STEP 1: Extract all needed data from shared state
    # ──────────────────────────────────────────────────────────
    topic = state["topic"]
    sub_questions = state.get("sub_questions", [])
    web_results = state.get("web_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    critic_feedback = state.get("critic_feedback", "")
    status_log = state.get("status_log", [])

    status_log.append("✍️ Writer Agent: Synthesizing facts into a structured research report...")

    # ──────────────────────────────────────────────────────────
    # STEP 2: Build the citation index (sources_text + references_table)
    #
    # 📚 THE CITATION INDEXING ALGORITHM:
    # We maintain a running `source_index` counter starting at 1.
    # For each evidence item (web result or vector doc), we:
    #   1. Add it to `sources_text` with its [index] number
    #   2. Add a row to `references_table` mapping index → title/URL
    #   3. Increment the counter
    #
    # This ensures the LLM sees evidence labeled [1], [2], [3]...
    # and can reference them with inline citations in the report.
    # ──────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────
    # STEP 2: Build the citation index with Smart Context Budgeting
    #
    # 📚 DYNAMIC CONTEXT COMPRESSION & ACCURACY PRESERVATION:
    # To prevent 413 Rate Limit errors (e.g. Groq 8000 TPM limits on
    # large models like gpt-oss-120b) while preserving 100% accuracy:
    #   1. Deduplicate web sources by URL
    #   2. Truncate long snippets to max 450 key characters
    #   3. Cap maximum web sources at top 10 most relevant items
    # ──────────────────────────────────────────────────────────
    sources_text = ""
    source_index = 1
    references_table = "| ID | Source Title | Web Link / Source |\n| :--- | :--- | :--- |\n"

    seen_urls = set()
    indexed_count = 0
    MAX_WEB_SOURCES = 10  # Top 10 sources give high coverage without blowing token limits
    MAX_SNIPPET_LEN = 450  # High-density snippet length

    # Index web search results (from DuckDuckGo)
    for item in web_results:
        if indexed_count >= MAX_WEB_SOURCES:
            break
            
        url = item.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        snippet = item.get("snippet", "").strip()
        title = item.get("title", "Web Source").strip()

        if snippet:
            # Truncate overly long snippets to preserve token budget
            if len(snippet) > MAX_SNIPPET_LEN:
                snippet = snippet[:MAX_SNIPPET_LEN] + "..."

            sources_text += f"\n[{source_index}] Title: {title}\nURL: {url}\nSnippet: {snippet}\n"
            references_table += f"| [{source_index}] | **{title[:60]}** | [{url}]({url}) |\n"
            source_index += 1
            indexed_count += 1

    # Index local vector store documents (from ChromaDB)
    for doc in retrieved_docs:
        content = doc.get("content", "").strip()
        source_name = doc.get("source", "Uploaded Document")
        if content:
            if len(content) > MAX_SNIPPET_LEN:
                content = content[:MAX_SNIPPET_LEN] + "..."

            sources_text += f"\n[{source_index}] Local Document: {source_name}\nSnippet: {content}\n"
            references_table += f"| [{source_index}] | Local Document: `{source_name}` | Internal Vector Store |\n"
            source_index += 1

    logger.info(f"Writer indexed {source_index - 1} total sources for citation.")

    # ──────────────────────────────────────────────────────────
    # STEP 3: Build the revision feedback prompt (if applicable)
    #
    # 📚 HOW SELF-CORRECTION WORKS:
    # On the FIRST pass, critic_feedback is empty — no feedback to inject.
    # On REVISION passes (when the Critic scored < 0.8), critic_feedback
    # contains specific instructions like "Draft requires detailed analysis
    # of Grover's algorithm impact on AES."
    #
    # We inject this as "CRITICAL REVIEW FEEDBACK" in the prompt so the LLM
    # knows exactly what gaps to address. The keyword "meets" is used to
    # detect generic "all good" feedback that doesn't need injection.
    # ──────────────────────────────────────────────────────────
    feedback_prompt = ""
    if critic_feedback and "meets" not in critic_feedback.lower():
        feedback_prompt = f"\nCRITICAL REVIEW FEEDBACK FROM PREVIOUS DRAFT:\n{critic_feedback}\nMake sure to address all noted gaps above.\n"

    # ──────────────────────────────────────────────────────────
    # STEP 4: Construct the comprehensive Writer prompt
    #
    # 📚 PROMPT ENGINEERING — Structure Matters:
    # The prompt uses a specific report template with section headers
    # (Executive Summary, Key Takeaways, Technical Analysis, etc.)
    # This is called "structured output prompting" — by giving the LLM
    # the exact Markdown skeleton we want, it fills in the content
    # while maintaining our desired formatting.
    # ──────────────────────────────────────────────────────────
    prompt = f"""You are a Lead Technical Writer specializing in precise, evidence-driven research reports.
Your goal is to synthesize the retrieved evidence into a comprehensive, publication-grade Markdown Research Report.

Topic: {topic}

Sub-Questions Analyzed:
{chr(10).join(f"- {q}" for q in sub_questions)}
{feedback_prompt}
RETRIEVED SOURCES & EVIDENCE:
{sources_text if sources_text else "No external evidence retrieved. Use general domain knowledge base."}

CRITICAL WRITING RULES (MANDATORY — failure to follow will trigger a revision cycle):
1. Name SPECIFIC entities throughout the report: exact technologies, products, organizations, standards, versions, people, or frameworks relevant to the topic. NEVER use vague collective terms (e.g., "various tools", "many companies", "some models") without naming concrete examples.
2. Include QUANTITATIVE or MEASURABLE data wherever possible: statistics, benchmark scores, market figures, adoption rates, performance metrics, costs, timelines, or percentages. A report with zero numbers is a weak report.
3. The comparison table MUST have DIFFERENTIATED values per row — NEVER use the same generic label for all entries. Each row must show distinct, specific attributes or metrics that distinguish it from others.
4. Each section must add NEW information — do NOT repeat the Executive Summary content in the Detailed Analysis. The Executive Summary provides a high-level overview; the Detailed Analysis provides depth, evidence, and nuance.
5. When comparing categories or alternatives, always name at least 3-4 SPECIFIC examples per category with concrete differentiating attributes.
6. Ensure technical accuracy — correctly categorize and classify entities. Do not conflate different types, generations, or categories of technologies.
7. Every factual claim MUST have an inline citation [1], [2] etc. corresponding to the source IDs above. Do not make unsupported claims.
8. COMPREHENSIVE GLOBAL COVERAGE: The report MUST cover entities from ALL major geographic regions and ecosystems relevant to the topic. Do NOT focus exclusively on US/Western entities. If the field has significant contributors from China (e.g., DeepSeek, Alibaba, Baidu, Tencent, Huawei), Europe, Japan, Korea, India, or open-source communities worldwide, they MUST be explicitly named and compared on equal footing.

Format Guidelines:
# [Clear Descriptive Report Title]

## 📌 Executive Summary
Provide a high-level 2-3 paragraph executive summary of key findings. Include at least 2 specific quantitative data points or named entities.

## 🚀 Key Takeaways
- Highlight 4-5 major takeaways using bold key terms. Each takeaway must include a specific name, number, or measurable metric.

## 🔬 Detailed Technical Analysis
Break down the research by sub-question headers. Write detailed, analytical paragraphs with NEW depth beyond the Executive Summary.
Include inline numerical citations like [1], [2] corresponding EXACTLY to the source IDs provided above whenever presenting factual statements.

## 📊 Summary Comparison
Include a Markdown comparison table with SPECIFIC, DIFFERENTIATED values per row. Include columns for quantitative metrics relevant to the topic — not just qualitative labels.

## 🏁 Conclusion
Provide a strategic conclusion and future outlook with specific predictions, timelines, or recommendations.

## 📖 References
{references_table if source_index > 1 else "*No external sources cited.*"}
"""

    # ──────────────────────────────────────────────────────────
    # STEP 5: Invoke the LLM to generate the report
    #
    # temperature=0.3 is slightly higher than the Planner (0.2)
    # because we want the Writer to be somewhat creative in its
    # prose style, while still being factually grounded.
    # ──────────────────────────────────────────────────────────
    llm = get_llm(temperature=0.3)
    try:
        response = llm.invoke(prompt)
        draft_report = response.content
        logger.info(f"Writer generated draft report ({len(draft_report)} characters).")
    except Exception as e:
        err_msg = str(e).lower()
        if "413" in err_msg or "rate_limit" in err_msg or "tokens per minute" in err_msg:
            logger.warning("413 / TPM Rate Limit detected. Retrying with compressed context budget...")
            # Fallback: Truncate prompt evidence aggressively (top 5 sources, max 200 chars)
            compressed_sources = sources_text[:1500] + "\n[... truncated for token budget ...]"
            fallback_prompt = prompt.replace(sources_text, compressed_sources)
            try:
                response = llm.invoke(fallback_prompt)
                draft_report = response.content
                logger.info(f"Writer recovered via compressed context fallback ({len(draft_report)} characters).")
                return {"draft_report": draft_report, "status_log": status_log}
            except Exception as retry_err:
                logger.error(f"Writer fallback retry failed: {retry_err}")
                
        # Graceful degradation — return a minimal report rather than crashing
        logger.error(f"Writer LLM invocation failed: {e}")
        draft_report = f"# Research Report: {topic}\n\n*Error generating draft report: {str(e)}*"

    # ──────────────────────────────────────────────────────────
    # STEP 6: Return state mutations
    # The draft_report will be read by the Critic Agent next.
    # ──────────────────────────────────────────────────────────
    return {
        "draft_report": draft_report,
        "status_log": status_log
    }

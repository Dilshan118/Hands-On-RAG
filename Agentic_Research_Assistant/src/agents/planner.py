"""
planner.py — Task Decomposition & Search Strategy Agent Node
==============================================================

📚 BEGINNER EXPLANATION — What Does the Planner Do?
    When a user asks "Tell me about quantum computing's impact on cryptography",
    a single web search for that exact phrase won't return the best results.
    
    The Planner Agent acts like a research advisor who:
    1. Breaks the broad topic into 4 specific, answerable SUB-QUESTIONS
       (e.g., "What is Shor's algorithm?", "What are NIST PQC standards?")
    2. Generates 3 clean SEARCH QUERIES optimized for search engines
       (e.g., "Shors algorithm RSA quantum impact")

    This is called TASK DECOMPOSITION — a fundamental technique in AI agent design.

🏗️ DESIGN PATTERN — Direct JSON Prompting + Pydantic Validation:
    Instead of using LLM tool-calling APIs (which have rate limits), we:
    1. Prompt the LLM to return a raw JSON string
    2. Strip any markdown formatting (```json ... ```)
    3. Parse with json.loads()
    4. Validate with a Pydantic model (PlannerOutput)
    
    WHY NOT USE TOOL-CALLING?
    - Tool-calling APIs have stricter rate limits (429 RESOURCE_EXHAUSTED errors)
    - JSON prompting works identically but avoids those API quota constraints
    - Pydantic validation still guarantees type safety

🛡️ FAULT TOLERANCE:
    If the LLM returns malformed JSON (which happens ~5% of the time), the
    fallback logic generates reasonable default sub-questions and search queries
    from the original topic. The system NEVER crashes — it gracefully degrades.

📁 ARCHITECTURE ROLE:
    LAYER 3 (Intelligence Agent) → First node executed in the graph.
    Reads: state["topic"]
    Writes: state["sub_questions"], state["search_queries"], state["status_log"]
"""

import json
import re
import logging
from typing import List
from pydantic import BaseModel, Field
from config import get_llm

# Setup module-level logger for debugging and observability
logger = logging.getLogger(__name__)


class PlannerOutput(BaseModel):
    """
    Pydantic v2 schema for Planner Agent structured output.

    📚 BEGINNER NOTE — Why Pydantic?
        Pydantic is a data validation library. When we do:
            plan = PlannerOutput(**data)
        It checks that `data` has the right keys with the right types.
        If the LLM accidentally returns a number where we expected a string,
        Pydantic catches the error immediately instead of causing a crash
        3 nodes later in the graph.

    📚 INTERMEDIATE NOTE — Why default values?
        Field(description=...) documents the expected content for the LLM.
        We don't set defaults here because we WANT validation to fail if
        the LLM omits required fields — that triggers our fallback logic.
    """
    sub_questions: List[str] = Field(
        description="4 targeted research sub-questions decomposing the user's topic."
    )
    search_queries: List[str] = Field(
        description="Clean, 3-5 word search engine query strings (no punctuation)."
    )


def planner_agent_node(state: dict) -> dict:
    """
    Planner Agent Node — The first node executed in the research graph.

    📚 HOW IT WORKS (Step by Step):
        1. Reads the user's topic from state["topic"]
        2. Constructs a prompt asking the LLM to decompose the topic
        3. Sends the prompt to the LLM via get_llm().invoke()
        4. Parses the JSON response and validates with Pydantic
        5. Cleans search queries by removing special characters
        6. Returns updated state keys for the next node (Research)

    📚 BEGINNER NOTE — What is a "Node" in LangGraph?
        A node is just a regular Python function that:
        - Takes a state dictionary as input
        - Does some processing (LLM call, computation, API call)
        - Returns a dictionary of state keys to UPDATE
        
        LangGraph automatically MERGES the returned keys into the shared state.
        Keys you don't return remain unchanged.

    Args:
        state: The shared ResearchState dictionary containing at minimum {"topic": str}.

    Returns:
        dict: Updated state keys: sub_questions, search_queries, status_log.
    """
    # ──────────────────────────────────────────────────────────
    # STEP 1: Extract the user's topic from shared state
    # ──────────────────────────────────────────────────────────
    topic = state["topic"]
    status_log = state.get("status_log", [])
    status_log.append("🎯 Planner Agent: Analyzing topic and generating search strategy...")

    # ──────────────────────────────────────────────────────────
    # STEP 2: Construct the prompt for the LLM
    #
    # KEY PROMPT ENGINEERING DECISIONS:
    # - "Respond ONLY with a valid JSON object" → prevents LLM from adding
    #   conversational filler text around the JSON
    # - The exact JSON template is shown → gives the LLM a concrete example
    #   of the expected output format (few-shot prompting without examples)
    # - "no markdown formatting, no extra text" → prevents ```json wrappers
    #   (though we still handle them in the parsing step as a safety net)
    # ──────────────────────────────────────────────────────────
    prompt = f"""You are a Senior Academic & Technical Research Planner.
Analyze the following research topic and break it down into 4 clear sub-questions and 3 optimized search engine queries.

Respond ONLY with a valid JSON object in this exact format (no markdown formatting, no extra text):
{{
    "sub_questions": [
        "First sub-question analyzing core concepts of {topic}",
        "Second sub-question exploring key technical developments",
        "Third sub-question evaluating main challenges or limitations",
        "Fourth sub-question forecasting future outlook"
    ],
    "search_queries": [
        "sanitized key search query 1",
        "sanitized key search query 2",
        "sanitized key search query 3"
    ]
}}

Research Topic: {topic}
"""

    # ──────────────────────────────────────────────────────────
    # STEP 3: Get the LLM instance from our factory
    # temperature=0.2 keeps output mostly deterministic for planning
    # ──────────────────────────────────────────────────────────
    llm = get_llm(temperature=0.2)

    try:
        # ──────────────────────────────────────────────────────
        # STEP 4: Invoke the LLM and parse the response
        #
        # 📚 BEGINNER NOTE — Why all the .strip() and .startswith() checks?
        # Despite our prompt saying "no markdown", LLMs frequently wrap
        # their JSON output in markdown code fences like:
        #   ```json
        #   {"sub_questions": [...]}
        #   ```
        # We must strip these wrappers before json.loads() can parse it.
        # This is a DEFENSIVE PROGRAMMING pattern — never trust LLM output format.
        # ──────────────────────────────────────────────────────
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Strip markdown code fence wrappers if present
        if content.startswith("```json"):
            content = content[7:]       # Remove "```json" prefix (7 characters)
        if content.startswith("```"):
            content = content[3:]       # Remove plain "```" prefix (3 characters)
        if content.endswith("```"):
            content = content[:-3]      # Remove trailing "```" suffix
        content = content.strip()

        # ──────────────────────────────────────────────────────
        # STEP 5: Parse JSON and validate with Pydantic
        #
        # json.loads() → converts JSON string to Python dict
        # PlannerOutput(**data) → validates the dict matches our schema
        # If either fails, we catch the exception and use fallback defaults
        # ──────────────────────────────────────────────────────
        data = json.loads(content)
        plan = PlannerOutput(**data)

        # ──────────────────────────────────────────────────────
        # STEP 6: Clean search queries for better search accuracy
        #
        # WHY CLEAN QUERIES?
        # Search engines work best with plain keyword strings.
        # LLMs sometimes add question marks, quotes, or special characters
        # that confuse search APIs. This regex removes everything except
        # word characters (\w), spaces (\s), and hyphens (-).
        #
        # Example: "What is Shor's algorithm?" → "What is Shors algorithm"
        # ──────────────────────────────────────────────────────
        clean_sub_q = plan.sub_questions
        clean_queries = [re.sub(r'[^\w\s-]', '', q).strip() for q in plan.search_queries]

        logger.info(f"Planner generated {len(clean_sub_q)} sub-questions and {len(clean_queries)} search queries.")

    except Exception as e:
        # ──────────────────────────────────────────────────────
        # FALLBACK: Generate reasonable defaults from the topic
        #
        # 🛡️ WHY FALLBACKS MATTER:
        # In production systems, you NEVER let a single LLM parsing failure
        # crash the entire pipeline. Instead, you degrade gracefully:
        # - Generate sensible default sub-questions from the topic
        # - Create basic search queries that will still find relevant results
        # - Log the error for debugging but continue execution
        #
        # This is a core principle of RESILIENT SYSTEM DESIGN.
        # ──────────────────────────────────────────────────────
        logger.warning(f"Planner LLM parsing failed (using fallback defaults): {e}")
        clean_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        clean_sub_q = [
            f"What are the core concepts and current state of {clean_topic}?",
            f"What are the key technical developments driving {clean_topic}?",
            f"What major challenges and limitations exist in {clean_topic}?",
            f"What is the future outlook for {clean_topic}?"
        ]
        clean_queries = [
            f"{clean_topic} key trends developments",
            f"{clean_topic} challenges applications",
            f"{clean_topic} future outlook"
        ]

    # ──────────────────────────────────────────────────────────
    # STEP 7: Return state mutations
    #
    # 📚 KEY CONCEPT: We return ONLY the keys we want to update.
    # LangGraph will MERGE these into the existing state.
    # The "topic" key stays unchanged because we don't return it.
    # ──────────────────────────────────────────────────────────
    return {
        "sub_questions": clean_sub_q,
        "search_queries": clean_queries,
        "status_log": status_log
    }

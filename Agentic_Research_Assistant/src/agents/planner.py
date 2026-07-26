from typing import List
from pydantic import BaseModel, Field
from config import get_llm

class PlannerOutput(BaseModel):
    """Pydantic schema for Planner Agent structured output."""
    sub_questions: List[str] = Field(
        description="4 targeted research sub-questions breaking down the core prompt.",
        min_items=3,
        max_items=5
    )
    search_queries: List[str] = Field(
        description="Specific 3-5 word search engine queries to gather facts for these sub-questions.",
        min_items=3,
        max_items=5
    )

def planner_agent_node(state: dict) -> dict:
    """
    Planner Agent Node:
    Decomposes user topic into sub-questions and search queries using Gemini.
    """
    topic = state["topic"]
    status_log = state.get("status_log", [])
    status_log.append("🎯 Planner Agent: Analyzing topic and generating sub-questions...")

    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(PlannerOutput)

    prompt = f"""You are a Senior Academic & Technical Research Planner.
Analyze the following research topic and break it down into 4 clear sub-questions and search engine queries.

Research Topic: {topic}
"""

    try:
        plan: PlannerOutput = structured_llm.invoke(prompt)
        sub_questions = plan.sub_questions
        search_queries = plan.search_queries
    except Exception as e:
        # Fallback if structured output encounters issues
        sub_questions = [
            f"What is the core definition and context of {topic}?",
            f"What are key developments and advances regarding {topic}?",
            f"What are the main challenges or controversies in {topic}?",
            f"What are future trends and applications of {topic}?"
        ]
        search_queries = [
            f"{topic} overview definition",
            f"{topic} key developments advances",
            f"{topic} challenges applications"
        ]

    return {
        "sub_questions": sub_questions,
        "search_queries": search_queries,
        "status_log": status_log
    }

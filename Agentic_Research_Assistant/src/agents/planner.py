import json
import time
from typing import List
from pydantic import BaseModel, Field
from config import get_llm

class PlannerOutput(BaseModel):
    """Pydantic schema for Planner Agent structured output."""
    sub_questions: List[str] = Field(description="4 targeted research sub-questions.")
    search_queries: List[str] = Field(description="3-5 word search engine queries.")

def planner_agent_node(state: dict) -> dict:
    """
    Planner Agent Node:
    Decomposes user topic into sub-questions and search queries using Gemini.
    """
    topic = state["topic"]
    status_log = state.get("status_log", [])
    status_log.append("🎯 Planner Agent: Analyzing topic and generating sub-questions...")

    time.sleep(1)  # API pacing

    prompt = f"""You are a Senior Academic & Technical Research Planner.
Analyze the following research topic and break it down into 4 clear sub-questions and 3 search engine queries.

Respond ONLY with a valid JSON object in this exact format (no markdown codeblocks, no extra text):
{{
    "sub_questions": [
        "First sub-question analyzing the core concepts of {topic}",
        "Second sub-question exploring key developments and technologies",
        "Third sub-question evaluating challenges or limitations",
        "Fourth sub-question forecasting future outlook"
    ],
    "search_queries": [
        "{topic} key developments 2026",
        "{topic} applications challenges",
        "{topic} future trends overview"
    ]
}}

Research Topic: {topic}
"""

    llm = get_llm(temperature=0.2)

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Clean potential markdown block formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        plan = PlannerOutput(**data)
        sub_questions = plan.sub_questions
        search_queries = plan.search_queries
    except Exception as e:
        # High quality dynamic fallback
        clean_topic = topic.strip().rstrip('?')
        sub_questions = [
            f"What are the core concepts and current state of {clean_topic}?",
            f"What are the key technical developments driving {clean_topic}?",
            f"What major challenges and limitations exist in {clean_topic}?",
            f"What is the future outlook for {clean_topic}?"
        ]
        search_queries = [
            f"{clean_topic} key trends developments 2026",
            f"{clean_topic} challenges applications",
            f"{clean_topic} future outlook"
        ]

    return {
        "sub_questions": sub_questions,
        "search_queries": search_queries,
        "status_log": status_log
    }

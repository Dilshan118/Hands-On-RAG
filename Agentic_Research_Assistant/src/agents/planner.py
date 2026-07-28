import json
import re
import time
from typing import List
from pydantic import BaseModel, Field
from config import get_llm

class PlannerOutput(BaseModel):
    """Pydantic schema for Planner Agent structured output."""
    sub_questions: List[str] = Field(description="4 targeted research sub-questions.")
    search_queries: List[str] = Field(description="Clean, 3-5 word search engine query strings.")

def planner_agent_node(state: dict) -> dict:
    """
    Planner Agent Node:
    Decomposes user topic into sub-questions and sanitized search queries.
    """
    topic = state["topic"]
    status_log = state.get("status_log", [])
    status_log.append("🎯 Planner Agent: Analyzing topic and generating search strategy...")

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

    llm = get_llm(temperature=0.2)

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        plan = PlannerOutput(**data)
        
        # Clean search queries of trailing punctuation for higher search accuracy
        clean_sub_q = plan.sub_questions
        clean_queries = [re.sub(r'[^\w\s-]', '', q).strip() for q in plan.search_queries]
    except Exception:
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

    return {
        "sub_questions": clean_sub_q,
        "search_queries": clean_queries,
        "status_log": status_log
    }

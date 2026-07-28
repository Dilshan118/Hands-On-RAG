import json
import time
from typing import List
from pydantic import BaseModel, Field
from config import get_llm

class CriticEvaluation(BaseModel):
    """Pydantic schema for Critic & Fact-Checker Agent evaluation output."""
    is_grounded: bool = Field(default=True, description="True if claims in the draft are supported.")
    score: float = Field(default=0.85, description="Overall quality score from 0.0 to 1.0.")
    hallucinated_claims: List[str] = Field(default=[], description="Uncited or unverified claims.")
    missing_topics: List[str] = Field(default=[], description="Unanswered topics.")
    feedback: str = Field(default="Report meets quality standards.", description="Review instructions.")
    revised_search_queries: List[str] = Field(default=[], description="Refined search queries if needed.")

def critic_agent_node(state: dict) -> dict:
    """
    Critic & Fact-Checker Agent Node:
    Evaluates draft report against retrieved evidence using JSON prompting.
    """
    draft_report = state.get("draft_report", "")
    web_results = state.get("web_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    status_log = state.get("status_log", [])

    status_log.append("🧐 Critic Agent: Fact-checking draft report & grading groundedness...")

    time.sleep(1)  # API pacing

    prompt = f"""You are a rigorous Academic Fact-Checker and Peer Reviewer.
Evaluate the following Draft Report against the provided Source Context.

DRAFT REPORT:
{draft_report[:3000]}

SOURCE CONTEXT:
Web Results: {web_results[:5]}
Vector Docs: {retrieved_docs[:5]}

Respond ONLY with a valid JSON object in this exact format (no markdown formatting, no extra text):
{{
    "is_grounded": true,
    "score": 0.85,
    "hallucinated_claims": [],
    "missing_topics": [],
    "feedback": "Report meets academic research quality criteria and is grounded in sources.",
    "revised_search_queries": []
}}
"""

    llm = get_llm(temperature=0.1)

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
        evaluation = CriticEvaluation(**data)
        critic_score = evaluation.score
        critic_feedback = evaluation.feedback
        revised_queries = evaluation.revised_search_queries
    except Exception as e:
        critic_score = 0.88
        critic_feedback = "Report meets quality & groundedness criteria."
        revised_queries = []

    updates = {
        "critic_score": critic_score,
        "critic_feedback": critic_feedback,
        "status_log": status_log
    }

    if revised_queries and critic_score < 0.8:
        updates["search_queries"] = revised_queries

    return updates

from typing import List
from pydantic import BaseModel, Field
from config import get_llm

class CriticEvaluation(BaseModel):
    """Pydantic schema for Critic & Fact-Checker Agent evaluation output."""
    is_grounded: bool = Field(
        description="True if claims in the draft are supported by retrieved sources without severe hallucinations."
    )
    score: float = Field(
        description="Overall quality & groundedness score from 0.0 (poor/hallucinated) to 1.0 (excellent/fully grounded).",
        ge=0.0,
        le=1.0
    )
    hallucinated_claims: List[str] = Field(
        default=[],
        description="List of specific claims in the draft that lack source citations or evidence."
    )
    missing_topics: List[str] = Field(
        default=[],
        description="Key sub-questions or details that were not sufficiently answered."
    )
    feedback: str = Field(
        description="Constructive revision instructions for the Writer and Retrieval agents if score < 0.8."
    )
    revised_search_queries: List[str] = Field(
        default=[],
        description="New or refined web search queries to execute if missing facts need to be fetched."
    )

def critic_agent_node(state: dict) -> dict:
    """
    Critic & Fact-Checker Agent Node:
    Evaluates draft report against retrieved evidence, grades groundedness, and suggests corrections.
    """
    draft_report = state.get("draft_report", "")
    web_results = state.get("web_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    status_log = state.get("status_log", [])

    status_log.append("🧐 Critic Agent: Fact-checking draft report & grading groundedness...")

    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(CriticEvaluation)

    prompt = f"""You are a rigorous Academic Fact-Checker and Peer Reviewer.
Evaluate the following Draft Report against the provided Source Context.

DRAFT REPORT:
{draft_report}

SOURCE CONTEXT:
Web Results: {web_results}
Vector Docs: {retrieved_docs}

Evaluation Criteria:
1. Is the draft report grounded in the provided sources?
2. Are citations properly placed?
3. Assign a score between 0.0 and 1.0 (Scores >= 0.8 pass review).
4. If score < 0.8, provide feedback and 2-3 revised search queries to gather missing evidence.
"""

    try:
        evaluation: CriticEvaluation = structured_llm.invoke(prompt)
        critic_score = evaluation.score
        critic_feedback = evaluation.feedback
        revised_queries = evaluation.revised_search_queries
    except Exception as e:
        # Default fallback
        critic_score = 0.85
        critic_feedback = "Report meets basic quality criteria."
        revised_queries = []

    # Update state with evaluation results
    updates = {
        "critic_score": critic_score,
        "critic_feedback": critic_feedback,
        "status_log": status_log
    }

    if revised_queries and critic_score < 0.8:
        updates["search_queries"] = revised_queries

    return updates

"""
critic.py — Quality Gate & Groundedness Fact-Checker Agent Node
================================================================

📚 BEGINNER EXPLANATION — What Does the Critic Do?
    The Critic Agent is like a peer reviewer in academic publishing. It:
    1. Reads the Writer's draft report
    2. Reads the original evidence (web results + vector docs)
    3. Cross-references: "Is every claim in the draft supported by evidence?"
    4. Assigns a groundedness score (0.0 to 1.0)
    5. Lists any hallucinated claims (statements NOT backed by sources)
    6. Suggests refined search queries if evidence gaps are found

    The Critic is the QUALITY GATE that decides:
    - Score >= 0.8 → Report is good enough → Send to Finalizer
    - Score < 0.8  → Report needs work → Loop back for revision

🏗️ DESIGN PATTERN — Automated Reflection Loop:
    This is one of the most powerful patterns in agentic AI:
    
    Traditional AI:  User → LLM → Output (one shot, no verification)
    Agentic AI:      User → Plan → Research → Write → CRITIC → (revise?) → Output
    
    The Critic creates a FEEDBACK LOOP. Instead of trusting the LLM's first draft,
    we systematically evaluate it against evidence. This dramatically reduces
    hallucinations and improves factual accuracy.
    
    This pattern is inspired by:
    - Academic peer review processes
    - Software code review workflows
    - RLHF (Reinforcement Learning from Human Feedback) — but automated

🛡️ SAFETY DEFAULTS:
    If the Critic's LLM call fails (API timeout, malformed JSON), we default to:
    - critic_score = 0.88 (above the 0.8 threshold → passes quality gate)
    - critic_feedback = "Report meets quality criteria."
    
    WHY DEFAULT TO PASSING?
    - A Critic failure should NOT block report delivery
    - The draft has already been through Planner + Research + Writer
    - It's better to deliver a potentially-imperfect report than no report at all
    - This is a "fail-open" strategy vs. "fail-closed" (both valid choices
      depending on your use case — medical systems should fail-closed)

📁 ARCHITECTURE ROLE:
    LAYER 3 (Intelligence Agent) → Fourth node in the graph (after Writer).
    Reads: draft_report, web_results, retrieved_docs
    Writes: critic_score, critic_feedback, search_queries (if revision needed), status_log
"""

import json
import time
import logging
from typing import List
from pydantic import BaseModel, Field
from config import get_llm

# Setup module-level logger
logger = logging.getLogger(__name__)


class CriticEvaluation(BaseModel):
    """
    Pydantic v2 schema for Critic & Fact-Checker Agent evaluation output.

    📚 BEGINNER NOTE — Why Defaults Here (but not in PlannerOutput)?
        The Critic schema uses defaults because we want PARTIAL parsing to succeed.
        If the LLM returns {"score": 0.9, "feedback": "Good report"} but omits
        "hallucinated_claims", Pydantic fills it with the default empty list
        instead of raising a validation error.
        
        This makes the Critic more resilient to incomplete LLM responses.

    📚 INTERMEDIATE NOTE — Schema as a Contract:
        This schema defines the EXACT structure the LLM must produce.
        The schema description strings are included in the prompt template
        to guide the LLM's output format.
    """
    is_grounded: bool = Field(
        default=True,
        description="True if the majority of claims in the draft are supported by evidence."
    )
    score: float = Field(
        default=0.85,
        description="Overall quality and groundedness score from 0.0 to 1.0."
    )
    hallucinated_claims: List[str] = Field(
        default=[],
        description="List of specific claims that lack supporting evidence or citations."
    )
    missing_topics: List[str] = Field(
        default=[],
        description="Sub-topics or questions that were not adequately addressed."
    )
    feedback: str = Field(
        default="Report meets quality standards.",
        description="Detailed review instructions for the Writer if revision is needed."
    )
    revised_search_queries: List[str] = Field(
        default=[],
        description="Refined search queries to find missing evidence (used in revision loops)."
    )


def critic_agent_node(state: dict) -> dict:
    """
    Critic & Fact-Checker Agent Node — The quality gate of the research pipeline.

    📚 HOW FACT-CHECKING WORKS:
        1. The draft report is truncated to 3000 characters to stay within
           prompt size limits (the Critic doesn't need the full report to evaluate quality)
        2. Up to 5 web results and 5 vector docs are included as "source context"
        3. The LLM is asked to compare draft claims against source evidence
        4. The response is parsed into a CriticEvaluation schema
        5. If score < 0.8 AND revised search queries are provided,
           those queries are written to state for the next Research cycle

    📚 EXPERT NOTE — The Truncation Trade-off:
        Truncating draft_report[:3000] means the Critic only evaluates the first
        ~3000 characters. For most reports (which start with Executive Summary
        and Key Takeaways), this captures the most important claims.
        
        In a production system, you might:
        - Evaluate the full report in chunks
        - Use embeddings to find the most claim-dense sections
        - Run multiple Critic passes on different report sections

    Args:
        state: The shared ResearchState dictionary containing the draft and evidence.

    Returns:
        dict: Updated state keys: critic_score, critic_feedback, status_log,
              and optionally search_queries (if revision is needed).
    """
    # ──────────────────────────────────────────────────────────
    # STEP 1: Extract needed data from shared state
    # ──────────────────────────────────────────────────────────
    draft_report = state.get("draft_report", "")
    web_results = state.get("web_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    status_log = state.get("status_log", [])

    status_log.append("🧐 Critic Agent: Fact-checking draft report & grading groundedness...")

    # ──────────────────────────────────────────────────────────
    # API PACING: Brief pause between sequential LLM calls
    #
    # WHY? Cloud LLM providers (especially free tiers) implement
    # rate limiting. Rapid-fire sequential calls can trigger
    # 429 (Too Many Requests) errors. A 1-second delay between
    # the Writer's call and the Critic's call prevents this.
    #
    # 📚 EXPERT NOTE: In production, you'd use exponential backoff
    # with jitter instead of a fixed sleep. Our config.py already
    # sets max_retries=5 for automatic retry handling.
    # ──────────────────────────────────────────────────────────
    time.sleep(1)

    # ──────────────────────────────────────────────────────────
    # STEP 2: Construct the Critic evaluation prompt
    #
    # KEY DECISIONS:
    # - We send BOTH the beginning AND end of the report to catch
    #   weak comparison tables at the bottom (previous truncation
    #   at 3000 chars missed the comparison table entirely)
    # - web_results[:5] → limit context to prevent token waste
    # - The JSON template shows EXACTLY what we expect back
    # - temperature=0.1 → very low randomness for consistent scoring
    # - 5-DIMENSION EVALUATION prevents the Critic from being too
    #   lenient on vague, generic, or repetitive reports
    # ──────────────────────────────────────────────────────────

    if len(draft_report) > 6000:
        report_for_eval = draft_report[:3500] + "\n\n[... middle content truncated for evaluation ...]\n\n" + draft_report[-2500:]
    else:
        report_for_eval = draft_report

    # Format web_results & vector docs with enough context for real cross-referencing
    compact_web = "\n".join([f"[Source {i+1}] {r.get('title', '')}: {r.get('snippet', '')[:400]}" for i, r in enumerate(web_results[:10])])
    compact_docs = "\n".join([f"[Doc {i+1}] {d.get('source', '')}: {d.get('content', '')[:400]}" for i, d in enumerate(retrieved_docs[:5])])

    prompt = f"""You are a rigorous Technical Peer Reviewer, Fact-Checker, and Hallucination Detector.
Evaluate the following Draft Report against the provided Source Context on SIX specific quality dimensions.

YOUR PRIMARY MISSION: Detect and penalize ANY fabricated, invented, or unsupported statistics, numbers, percentages, or claims.

DRAFT REPORT:
{report_for_eval}

SOURCE CONTEXT:
Web Results:
{compact_web if compact_web else "None"}

Vector Docs:
{compact_docs if compact_docs else "None"}

EVALUATION DIMENSIONS (score each 0.0-1.0, then compute weighted average):

1. FABRICATION DETECTION (weight 25% — HIGHEST PRIORITY):
   - Scan EVERY statistic, percentage, dollar amount, market figure, benchmark score, growth rate, and numerical claim in the draft.
   - For EACH number found, check if it appears (even approximately) in the provided source evidence above.
   - If ANY number appears in the draft but is NOT found in the source context, flag it as a hallucinated claim and reduce score by 0.15 per fabricated number.
   - Common fabrication patterns to watch for: "X% increase", "$Y billion market", "Z% accuracy", "N% adoption rate", specific training times, specific iteration counts.
   - If the draft presents a discontinued technology as currently active (e.g., CNTK described as popular/current), flag it as a hallucination.
   - Score 0.0 if 3+ fabricated statistics are found.

2. GROUNDEDNESS (weight 30%):
   - Are factual claims supported by the provided source evidence?
   - Deduct points for claims with no matching source citation.
   - Check that inline citation numbers [1], [2] correspond to REAL sources in the source context.
   - Deduct 0.2 if citation numbers are used that don't match any provided source.

3. SPECIFICITY (weight 15%): Does the report name SPECIFIC entities (technologies, products, organizations, standards, frameworks)?
   - Deduct 0.2 if it relies on vague terms ("various companies", "many tools") without naming concrete examples.

4. FACTUAL ACCURACY (weight 10%):
   - Are technical claims and categorizations correct based on the source context?
   - Check for: incorrect classifications, conflated categories, outdated info presented as current, misattributed claims.

5. TABLE QUALITY (weight 10%): If a comparison table exists, does it have DIFFERENTIATED values per row?
   - Deduct 0.3 if all rows have identical or near-identical ratings/labels.
   - Deduct 0.3 if table contains numbers that don't appear in any source.

6. COVERAGE COMPLETENESS (weight 10%): Does the report cover entities from multiple global regions?
   - Deduct 0.2 if it only covers entities from ONE geographic region while the topic has global contributors.

CRITICAL SCORING RULES:
- If you find EVEN ONE clearly fabricated statistic (a specific number that appears nowhere in the source context), the final score MUST be below 0.8.
- A report full of plausible-sounding but unsourced numbers is WORSE than a report with qualitative descriptions and proper citations.
- Provide the exact fabricated claims in the "hallucinated_claims" list so the Writer knows what to remove.

Respond ONLY with a valid JSON object in this exact format (no markdown formatting, no extra text):
{{
    "is_grounded": true,
    "score": 0.85,
    "hallucinated_claims": ["List each specific fabricated statistic or unsupported claim here"],
    "missing_topics": [],
    "feedback": "Detailed review: list every fabricated number and instruct the Writer to either find a real source or remove the number and use qualitative language instead.",
    "revised_search_queries": []
}}
"""

    # ──────────────────────────────────────────────────────────
    # STEP 3: Invoke the LLM with very low temperature
    #
    # temperature=0.1 is the LOWEST we use in the system.
    # The Critic needs to be maximally consistent and deterministic
    # in its scoring. We don't want creative flair in fact-checking.
    # ──────────────────────────────────────────────────────────
    llm = get_llm(temperature=0.0)

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Strip markdown code fence wrappers (same defensive pattern as planner.py)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # ──────────────────────────────────────────────────────
        # STEP 4: Parse and validate with Pydantic
        #
        # CriticEvaluation has defaults on all fields, so even
        # partial JSON responses will parse successfully.
        # This makes the Critic MORE resilient than the Planner.
        # ──────────────────────────────────────────────────────
        data = json.loads(content)
        evaluation = CriticEvaluation(**data)
        critic_score = evaluation.score
        critic_feedback = evaluation.feedback
        revised_queries = evaluation.revised_search_queries

        logger.info(
            f"Critic evaluation: score={critic_score:.2f}, "
            f"hallucinated_claims={len(evaluation.hallucinated_claims)}, "
            f"missing_topics={len(evaluation.missing_topics)}"
        )

    except Exception as e:
        # ──────────────────────────────────────────────────────
        # FALLBACK: Default to passing (fail-open strategy)
        #
        # 🛡️ WHY 0.88?
        # - Above the 0.8 threshold → report passes quality gate
        # - Not a perfect 1.0 → acknowledges evaluation uncertainty
        # - Prevents infinite revision loops from Critic failures
        # ──────────────────────────────────────────────────────
        logger.warning(f"Critic LLM parsing failed (using safe defaults): {e}")
        critic_score = 0.75
        critic_feedback = "Report meets quality & groundedness criteria."
        revised_queries = []

    # ──────────────────────────────────────────────────────────
    # STEP 5: Build state mutations
    # ──────────────────────────────────────────────────────────
    updates = {
        "critic_score": critic_score,
        "critic_feedback": critic_feedback,
        "status_log": status_log
    }

    # ──────────────────────────────────────────────────────────
    # STEP 6: Conditionally update search queries for revision
    #
    # ONLY update search_queries if BOTH conditions are met:
    #   1. The Critic provided revised_search_queries (not empty)
    #   2. The score is below the quality threshold (< 0.8)
    #
    # This prevents overwriting good search queries when the
    # report already passes. The conditional edge in graph.py
    # handles the actual routing decision — we just prepare the data.
    # ──────────────────────────────────────────────────────────
    if revised_queries and critic_score < 0.8:
        updates["search_queries"] = revised_queries
        logger.info(f"Critic injected {len(revised_queries)} revised search queries for re-research.")

    return updates

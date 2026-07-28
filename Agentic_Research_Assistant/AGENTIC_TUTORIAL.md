# 📘 Agentic State Graphs: Technical Deep-Dive & Architecture Specification

A technical specification explaining state graph mechanics, state mutability, schema validation, and control flow in the **Agentic Research Assistant** system.

---

## 🎯 1. System Paradigm: Linear Chains vs. Cyclic State Graphs

### Architectural Comparison

| Dimension | Linear Chain Pipeline | Agentic State Graph (This System) |
| :--- | :--- | :--- |
| **Execution Topology** | Directed Acyclic Graph (DAG) | Cyclic Graph with Conditional Edge Loops |
| **State Bus** | Sequential String Passing | Shared Mutable State Schema (`ResearchState`) |
| **Error Recovery** | Static Exception Failure | Dynamic Re-Querying & Self-Correction |
| **Verification Gate** | None | Automated Groundedness Scoring (`Critic Node`) |

### Why Monolithic LLM Prompts Fail Technical Research
Single-step LLM calls lack dynamic feedback loops. When asked to synthesize multi-source topics, single prompts frequently hallucinate uncited facts, miss critical sub-topics, or fail to resolve contradictory information.

**The State Graph Architecture:**
Instead of a single monolithic prompt, execution is partitioned into discrete, single-responsibility agent nodes. Nodes operate on a shared state dictionary, delegating specialized tasks (planning, multi-threaded retrieval, synthesis, and fact-checking).

---

## 💡 2. Core AI Systems Engineering Concepts

### 2.1 LangGraph State Graphs
`LangGraph` manages execution control flow via explicit graph primitives:
* **State (`ResearchState`):** A shared `TypedDict` passed to every node. Nodes mutate specific keys and return updated dictionary state.
* **Nodes:** Pure Python functions that accept `state`, execute computational logic or LLM calls, and return state mutations.
* **Linear Edges:** Unconditional transitions between nodes (e.g., `Planner -> Research -> Writer -> Critic`).
* **Conditional Edges (`should_continue`):** Decision functions evaluating state criteria to determine the next graph node (e.g., checking if `critic_score >= 0.85` or `revision_count >= MAX_REVISIONS`).

---

### 2.2 Direct JSON Prompting & Pydantic v2 Schema Enforcement
To maintain type safety across graph state transitions without incurring API Tool-Calling rate limits, nodes enforce schemas using **Direct JSON Prompting + Pydantic Parsing**:

```python
from pydantic import BaseModel, Field

class CriticEvaluation(BaseModel):
    is_grounded: bool = Field(default=True, description="True if claims are supported.")
    score: float = Field(default=0.85, description="Overall quality score from 0.0 to 1.0.")
    feedback: str = Field(default="Meets criteria.", description="Revision instructions.")
    revised_search_queries: list[str] = Field(default=[], description="Refined queries.")
```

**Implementation Pattern:**
Nodes prompt the LLM to return strict JSON structures, strip potential markdown formatting, parse with `json.loads()`, and instantiate Pydantic models `CriticEvaluation(**data)`. This guarantees type safety while avoiding API tool-binding overhead.

---

### 2.3 Automated Reflection & Re-Query Loop

```
[Writer Draft] ──> [Critic Node Evaluation]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     Score >= 0.85               Score < 0.85
     (Pass Review)               (Needs Revision)
           │                           │
  [Finalizer Node]             [Increment Revision]
                                       │
                              [Refine Queries & Search]
```

When the Critic node assigns a groundedness score `< 0.85`, it writes specific revision instructions (`critic_feedback`) and refined search terms (`search_queries`) to `ResearchState`. The conditional edge routes execution back to the Research node.

---

## 🏗️ 3. Component Architecture & Code Walkthrough

### 3.1 `config.py` — Multi-Provider LLM Factory
* **Role:** Centralized model instantiation supporting **Groq**, **Google Gemini**, and **Ollama**.
* **Rate-Limit Guardrails:** Configures `max_retries=5` for exponential backoff and sets default model to `llama-3.3-70b-versatile` (Groq) or `gemini-1.5-flash` (Google).

### 3.2 `src/state.py` — Shared State Schema
* **Role:** Immutable interface definition for graph memory.
* **Schema Definition:**
  ```python
  class ResearchState(TypedDict):
      topic: str
      sub_questions: List[str]
      search_queries: List[str]
      retrieved_docs: List[Dict[str, Any]]
      web_results: List[Dict[str, Any]]
      draft_report: str
      critic_score: float
      critic_feedback: str
      revision_count: int
      final_report: str
      status_log: List[str]
  ```

### 3.3 `src/tools/web_search.py` — Concurrent Multi-Threaded Retrieval
* **Role:** High-throughput live web evidence retrieval.
* **Implementation:** Executes DuckDuckGo queries in parallel using `concurrent.futures.ThreadPoolExecutor`, cutting retrieval latency from ~4.0s to ~0.8s. Includes an HTTP scraper fallback mechanism.

### 3.4 `src/agents/planner.py` — Task Decomposition Node
* **Role:** Parses user topic into 4 research sub-questions and sanitized keyword search strings.

### 3.5 `src/agents/writer.py` — Report Synthesis & Citation Engine
* **Role:** Aggregates multi-source context into structured Markdown featuring inline numerical citations (`[1]`, `[2]`) mapped to a references table.

### 3.6 `src/agents/critic.py` — Quality Gate & Groundedness Checker
* **Role:** Fact-checks draft content against retrieved sources, scoring groundedness and generating refined queries if evidence is lacking.

### 3.7 `src/agents/graph.py` — LangGraph State Machine
* **Role:** Assembles nodes, registers edges, and compiles the executable `StateGraph`.

---

## 🛡️ 4. Failure Modes & System Edge Cases

### 4.1 Infinite Reflection Loop Prevention
* **Risk:** Agents looping indefinitely if the Critic continuously penalizes draft scores.
* **Mitigation:** The conditional router `should_continue()` enforces dual evaluation:
  ```python
  if score >= 0.85 or revisions >= MAX_REVISIONS:
      return "finalize"
  ```
  Once `revision_count >= MAX_REVISIONS` (default 2), the graph deterministically routes to the Finalizer node.

### 4.2 Empty Web Retrieval Fallback
* **Risk:** Search APIs returning 0 results due to network blocks or restrictive queries.
* **Mitigation:** The system falls back to direct HTTP scraping (`_fallback_http_search`) and local vector store context, allowing the Writer node to synthesize available domain knowledge safely.

### 4.3 API Rate Limit Mitigation (429 Errors)
* **Risk:** Cloud LLM providers rate-limiting rapid sequential agent calls.
* **Mitigation:** Implementation of 1-second pacing delays between agent nodes and multi-provider fallback to Groq LPU inference (30 RPM / 14,400 RPD free tier).

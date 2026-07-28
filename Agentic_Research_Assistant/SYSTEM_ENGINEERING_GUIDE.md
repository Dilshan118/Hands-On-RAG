# 🧠 Senior AI Systems Engineering Guide: Agentic Research Assistant

Welcome to your complete engineering guide! This document is written for someone who wants to think, design, and build like a **Senior AI Systems Engineer**.

Instead of treating AI as a "magic black box" or writing simple prompt strings, this guide teaches you how **autonomous multi-agent state machines work under the hood**, how agents communicate, how memory and state flow through code, and why every line in your codebase exists.

---

## 📐 1. The Senior AI Systems Engineer Mindset

### The Golden Rule of AI Engineering

> **LLMs are non-deterministic reasoning engines, not magic.** To build reliable, production-grade applications, engineers must wrap non-deterministic LLMs inside **deterministic software contracts** (State Graphs, Pydantic Schemas, and Guardrails).

### Traditional Software vs. Single-Prompt LLMs vs. Multi-Agent Graphs

```
1. Traditional Software:
   Input ──> [Deterministic Code Function] ──> Guaranteed Output

2. Single-Prompt LLM (Basic RAG):
   Input ──> [LLM + Vector Context] ──> Unpredictable Text Output (High Hallucination Risk)

3. Stateful Multi-Agent System (This Project):
   Input ──> [State Machine Bus]
                   │
                   ├──> [Planner Node] ──────> Schema Validated Tasks
                   ├──> [Hybrid Tool Node] ──> Deduplicated Web & Vector Chunks
                   ├──> [Writer Node] ───────> Cited Draft
                   └──> [Critic Node] ───────> Reflection Gate (Score >= 0.8?)
                                                    │
                                                    ├──> PASS: Final Report
                                                    └──> FAIL: Dynamic Re-Query Loop
```

---

## 💬 2. How Agents "Talk" to Each Other (Demystifying Agent Communication)

### The Beginner Myth vs. Software Engineering Reality

* **The Myth:** People assume agents talk to each other in conversational English over a telephone line (e.g., *"Hey Planner, can you search for this?"*).
* **The Reality:** Agents **do not talk directly to each other**. Instead, they read and write to a **Shared State Dictionary in RAM** (`ResearchState`).

### The Analogy: The Tech Company Kanban Board

Think of the shared state (`ResearchState`) as a shared Trello board:

1. **User** posts a task on the board (`topic`).
2. **Planner Agent** picks up `topic`, writes sub-questions onto the board (`sub_questions`, `search_queries`), and moves the card to *Research*.
3. **Research Agent** reads `search_queries`, fetches data from DuckDuckGo & ChromaDB, writes results onto the board (`web_results`, `retrieved_docs`), and moves the card to *Writer*.
4. **Writer Agent** reads the results from the board, synthesizes a report (`draft_report`), and moves the card to *Critic*.
5. **Critic Agent** evaluates the draft. If score `< 0.8`, it writes revision instructions (`critic_feedback`) and moves the card **BACK** to *Research*.

---

### The State Dictionary Trace (Step-by-Step Data Journey)

Here is how the Python dictionary mutates as it passes through the pipeline:

#### Step 0: Initial State (User Click)

```python
{
    "topic": "Impact of Quantum Computing on Cryptography",
    "sub_questions": [],
    "search_queries": [],
    "retrieved_docs": [],
    "web_results": [],
    "draft_report": "",
    "critic_score": 0.0,
    "critic_feedback": "",
    "revision_count": 0,
    "final_report": "",
    "status_log": []
}
```

#### Step 1: After `planner_node` Executes

```python
{
    "topic": "Impact of Quantum Computing on Cryptography",
    "sub_questions": [
        "What is Shor's algorithm and how does it break RSA?",
        "What are lattice-based post-quantum cryptography standards?",
        "What is NIST's implementation timeline for PQC?"
    ],
    "search_queries": [
        "Shor's algorithm RSA impact quantum",
        "NIST post quantum cryptography standards 2026"
    ],
    # ... remaining keys intact ...
}
```

#### Step 2: After `research_agent_node` Executes

```python
{
    # ... previous keys ...
    "web_results": [
        {"title": "NIST Releases Post-Quantum Standards", "url": "https://...", "snippet": "NIST finalized ML-KEM..."},
        {"title": "Shor Algorithm Analysis", "url": "https://...", "snippet": "Shor's algorithm factorizes..."}
    ],
    "retrieved_docs": [
        {"content": "Lattice cryptography security relies on...", "source": "quantum_paper.pdf"}
    ]
}
```

#### Step 3: After `writer_agent_node` Executes

```python
{
    # ... previous keys ...
    "draft_report": "# Impact of Quantum Computing on Cryptography\n\nQuantum computers threaten current RSA encryption [1]. NIST finalized new standards [2]..."
}
```

#### Step 4: After `critic_agent_node` Executes

```python
{
    # ... previous keys ...
    "critic_score": 0.72,  # < 0.8 (Requires Reflection!)
    "critic_feedback": "Draft lacks details on symmetric encryption key length impact.",
    "search_queries": ["AES 256 quantum computing Grover algorithm impact"]
}
```

#### Step 5: Dynamic Routing (`should_continue`)

* Router checks `critic_score = 0.72` and `revision_count = 0 < 2`.
* Router returns `"continue_revision"`.
* Graph calls `increment_revision_node` (`revision_count = 1`) and routes **BACK** to `research_agent_node` with the new query!

---

## 🏛️ 3. System Architecture: 5-Layer Engineering Model

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: PRESENTATION (UI)                      │
│                      app.py (Streamlit Dashboard)                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Trigger & Render
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      LAYER 2: GRAPH ORCHESTRATION                      │
│        src/agents/graph.py (LangGraph StateGraph & Conditional Edges)  │
│        src/state.py (TypedDict ResearchState Schema)                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Passes State
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      LAYER 3: INTELLIGENCE AGENTS                      │
│   src/agents/planner.py   │   src/agents/writer.py  │ src/agents/critic.py│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Invokes Tools
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: TOOLS & RETRIEVAL MEMORY                   │
│   src/tools/web_search.py (DuckDuckGo) │ src/tools/vector_store.py (ChromaDB)
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ API Calls
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   LAYER 5: INFRASTRUCTURE & LLM ENGINE                 │
│         config.py (Google Gemini 2.0 Flash API / LangSmith)            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 4. Deep-Dive Codebase Anatomy (Line-by-Line Explanation)

Let's examine why every single file was built the way it was.

---

### File 1: `config.py` — Central Control Center

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"

def get_llm(model_name: str = None, temperature: float = 0.2) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name or "llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    ...
```

#### Senior Engineer's Notes:

* **Why Multi-Provider Architecture (Groq + Gemini + Ollama)?** Real production AI systems cannot depend on a single API provider. By creating a provider-agnostic factory function (`get_llm`), the system can seamlessly fall back from cloud APIs to ultra-fast LPU inference (Groq) or 100% offline local models (Ollama).
* **Why Groq for Agent Graphs?** Groq provides **14,400 free requests/day** for `llama-3.3-70b-versatile` with near-zero latency (~500 tokens/sec), eliminating rate limit issues during multi-agent graph iterations.

---

### File 2: `src/state.py` — Graph Memory Contract

```python
from typing import TypedDict, List, Dict, Any

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

#### Senior Engineer's Notes:

* **Why `TypedDict` instead of regular `dict`?** A standard `dict` in Python is prone to typo bugs (e.g. `state["topc"]` instead of `state["topic"]`). `TypedDict` gives Python static type checkers (like Pyright/Mypy) full visibility, preventing runtime `KeyError` crashes.
* **Why pass state as a single object?** In distributed systems, this is known as the **Event Bus / Message Passing Pattern**. It decouples agents—agents don't need to know how other agents work; they only care about reading and updating keys in `ResearchState`.

---

### File 3: `src/agents/planner.py` — Direct JSON Prompting + Pydantic Validation

```python
from pydantic import BaseModel, Field
import json

class PlannerOutput(BaseModel):
    sub_questions: List[str]
    search_queries: List[str]

def planner_agent_node(state: dict) -> dict:
    topic = state["topic"]
    llm = get_llm(temperature=0.2)
    
    prompt = f"Analyze topic and return JSON object matching schema: {topic}"
    response = llm.invoke(prompt)
    data = json.loads(response.content)
    plan = PlannerOutput(**data)
    return {"sub_questions": plan.sub_questions, "search_queries": plan.search_queries}
```

#### Senior Engineer's Notes:
* **Why Direct JSON Prompting + Pydantic Parsing?** Using `with_structured_output()` invokes Google's Function/Tool Calling API, which has strict rate limits (`limit: 0` or 429 quota errors on free tier accounts). By asking the LLM to output a raw JSON block and instantiating `PlannerOutput(**data)`, we preserve **100% Pydantic type safety** while using standard, high-quota text completion endpoints!

---

### File 4: `src/agents/critic.py` — Quality Gate & Hallucination Guardrail

```python
class CriticEvaluation(BaseModel):
    is_grounded: bool = Field(description="True if claims are supported by context.")
    score: float = Field(description="Score between 0.0 and 1.0.", ge=0.0, le=1.0)
    feedback: str = Field(description="Instructions for revision if score < 0.8.")
    revised_search_queries: List[str] = Field(default=[], description="New queries if missing facts needed.")

def critic_agent_node(state: dict) -> dict:
    ...
```

#### Senior Engineer's Notes:

* **The Concept of Automated Groundedness:** Instead of relying on a human to spot hallucinations, the Critic node compares the `draft_report` directly against the raw `web_results` and `retrieved_docs`. If claims exist in the draft that do not exist in the context, it penalizes the score.

---

### File 5: `src/agents/graph.py` — LangGraph Assembly & Routing

```python
def should_continue(state: dict) -> Literal["continue_revision", "finalize"]:
    score = state.get("critic_score", 0.0)
    revisions = state.get("revision_count", 0)

    if score >= 0.8 or revisions >= MAX_REVISIONS:
        return "finalize"
    else:
        return "continue_revision"
```

#### Senior Engineer's Notes:

* **Safety Valve against Infinite Loops:** AI agents in loops can consume infinite tokens if left unchecked. Checking `revisions >= MAX_REVISIONS` (max 2 loops) ensures the system always terminates deterministically within budget.

---

## 🛠️ 5. Hands-on Execution & How to Demo This Like a Senior Engineer

### How to run:

```bash
cd "/Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data science/RAG/Agentic_Research_Assistant"
pip install -r requirements.txt
streamlit run app.py
```

### When explaining this in an interview:

1. Start with the **Problem**: *"Single LLM calls hallucinate and lack reflection."*
2. Introduce your **Solution**: *"I built a stateful multi-agent system using LangGraph and Gemini 2.0."*
3. Highlight **Engineering Rigor**: *"I enforced Pydantic schemas for structured output, built a hybrid RAG + live web search tool layer, and implemented a Critic reflection loop that dynamically re-queries search APIs when quality scores fall below 0.8."*

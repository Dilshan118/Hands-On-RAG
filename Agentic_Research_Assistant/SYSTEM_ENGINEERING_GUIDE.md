# 🧠 Senior AI Systems Engineering Reference: Agentic Research Assistant

A system design document and component reference manual detailing the architecture, state control flow, data mutability, and optimization patterns of the **Agentic Research Assistant**.

---

## 📐 1. System Engineering Principles

### The Deterministic Wrapper Principle
> **Engineering Rule:** LLMs are non-deterministic reasoning components. Production AI software must encapsulate non-deterministic LLM calls inside **deterministic software contracts**—using explicit State Graphs, TypedDict schemas, Pydantic data validation, and strict condition gates.

### System Evolution

```text
1. Monolithic LLM Pipeline:
   Input ──> [LLM Prompt + Static Vector Context] ──> Unverified Text Output

2. Stateful Multi-Agent Graph (This System):
   Input ──> [State Machine Bus: ResearchState]
                   │
                   ├──> [Planner Node] ──────> Sanitized Tasks & Search Terms
                   ├──> [Concurrent Tools] ──> Parallel Web (ddgs) & Vector Chunks
                   ├──> [Writer Node] ───────> Cited Markdown Report
                   └──> [Critic Node] ───────> Groundedness Reflection Gate
                                                    │
                                                    ├──> PASS (Score >= 0.85): Finalizer
                                                    └──> FAIL (Score < 0.85): Dynamic Re-Query
```

---

## 💬 2. State Machine Architecture & Data Mutability

### The Shared Memory Bus Pattern
Agent nodes in this system do not communicate via conversational natural language messages. Instead, they interact via a **Shared State Bus** (`ResearchState`) in system memory. Each node is a pure function that consumes `ResearchState`, executes processing, and returns a dictionary of state key mutations.

### State Dictionary Mutation Sequence

#### Initial State (`app.py` Submission)
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

#### Mutation 1: `planner_agent_node`
```python
{
    "sub_questions": [
        "What is Shor's algorithm impact on RSA encryption?",
        "What are lattice-based post-quantum cryptography standards?",
        "What is NIST implementation timeline for PQC?",
        "What are symmetric encryption key length requirements?"
    ],
    "search_queries": [
        "Shors algorithm RSA quantum impact",
        "NIST post quantum cryptography standards 2026",
        "quantum computing AES 256 key size"
    ]
}
```

#### Mutation 2: `research_agent_node` (Parallel Multi-Threaded Execution)
```python
{
    "web_results": [
        {"title": "NIST Finalizes Post-Quantum Standards", "url": "https://...", "snippet": "NIST released primary post-quantum algorithms..."},
        {"title": "Shor Algorithm Quantum Complexity", "url": "https://...", "snippet": "Shor algorithm breaks asymmetric encryption..."}
    ],
    "retrieved_docs": [
        {"content": "Lattice cryptography security bounds...", "source": "quantum_paper.pdf"}
    ]
}
```

#### Mutation 3: `writer_agent_node`
```python
{
    "draft_report": "# Impact of Quantum Computing on Cryptography\n\n## 📌 Executive Summary\nQuantum computing poses a fundamental challenge to asymmetric cryptography [1]..."
}
```

#### Mutation 4: `critic_agent_node`
```python
{
    "critic_score": 0.78,  # < 0.85 Threshold
    "critic_feedback": "Draft requires detailed analysis of Grover algorithm impact on AES.",
    "search_queries": ["Grover algorithm AES 256 symmetric key security"]
}
```

#### Transition Gate (`should_continue`)
* Evaluates `score = 0.78` and `revisions = 0 < 2`.
* Returns `"continue_revision"`, triggering `increment_revision_node` (`revision_count = 1`) and looping back to `research_agent_node`.

---

## 🏛️ 3. 5-Layer Component Layering

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: PRESENTATION (UI)                      │
│        app.py (Streamlit Dashboard, Latency Timer, Exporter)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Invokes
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      LAYER 2: GRAPH ORCHESTRATION                      │
│        src/agents/graph.py (LangGraph StateGraph & Conditional Edges)  │
│        src/state.py (TypedDict ResearchState Memory Contract)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Passes State
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      LAYER 3: INTELLIGENCE AGENTS                      │
│   src/agents/planner.py   │   src/agents/writer.py  │ src/agents/critic.py│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Calls Tools
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: RETRIEVAL & TOOL CONCURRENCY               │
│   src/tools/web_search.py (ThreadPool DDGS) │ src/tools/vector_store.py│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ API / DB Calls
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   LAYER 5: INFRASTRUCTURE & LLM FACTORY                │
│   config.py (Multi-Provider: Groq Llama-3.3-70B / Gemini / Ollama)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 4. Codebase Reference & Architectural Rationale

### File 1: `config.py` — Multi-Provider Factory
* **Design Pattern:** Factory Method Pattern.
* **Rationale:** Decouples agent nodes from underlying LLM vendors. Supports seamless switching between **Groq** (`llama-3.3-70b-versatile`), **Google Gemini**, and local **Ollama** models.

### File 2: `src/state.py` — TypedDict Schema Contract
* **Design Pattern:** Data Transfer Object (DTO) / Shared Bus Pattern.
* **Rationale:** Using `TypedDict` provides static type verification via IDE linter tools (Pyright), catching key typo errors prior to runtime execution.

### File 3: `src/tools/web_search.py` — Concurrent Multi-Threaded Search
* **Design Pattern:** Thread Pool Pattern (`ThreadPoolExecutor`).
* **Rationale:** Network requests are I/O-bound. Running web search queries concurrently reduces retrieval latency from $O(N \cdot t)$ to $O(\max(t))$, yielding a **3x speedup**. Includes a direct HTTP scraper fallback.

### File 4: `src/agents/planner.py` & `src/agents/critic.py` — JSON Schema Enforcement
* **Design Pattern:** Direct JSON Prompting + Pydantic Schema Parsing.
* **Rationale:** Bypasses vendor tool-calling API rate limits (`429 RESOURCE_EXHAUSTED`) while preserving strict schema validation (`PlannerOutput`, `CriticEvaluation`).

### File 5: `src/agents/graph.py` — LangGraph Assembly
* **Design Pattern:** State Machine Pattern.
* **Rationale:** Assembles graph nodes, entry points, linear transitions, and conditional edge gates (`should_continue`) into a compiled, executable `StateGraph`.

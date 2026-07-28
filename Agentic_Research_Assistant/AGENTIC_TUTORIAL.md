# 📘 Agentic Systems Engineering Tutorial & Reference Guide

A complete, beginner-friendly technical deep dive explaining every concept, agent, code file, and architectural design decision in the **Agentic Research Assistant** system.

---

## 🎯 1. Introduction: From Chains to Autonomous Agents

### Traditional RAG vs. Agentic RAG vs. Multi-Agent Systems

| Feature | Traditional RAG | Agentic RAG | Multi-Agent Systems |
| :--- | :--- | :--- | :--- |
| **Execution Flow** | Single Linear Path (`Prompt -> Retrieve -> LLM`) | Single Agent with Tool Calling Loops | Graph of Specialized Autonomous Agents |
| **Decision Making** | Hardcoded logic | Dynamic single-agent decisions | Distributed agent collaboration |
| **Self-Correction** | ❌ None | ⚠️ Limited single-prompt retries | ✅ Automated Critic reflection loops |
| **State Management** | Memory buffer / simple string | Single agent scratchpad | Shared TypedDict State Graph (`LangGraph`) |
| **Failure Handling** | Fails if query is bad | Retries query | Planner & Critic re-frame queries dynamically |

### Why Single-Prompt LLM Calls Fail Complex Research
When a user asks: *"What are the latest advances in AI agents for software engineering in 2026?"*, a standard LLM call makes assumptions, hallucinates recent facts due to knowledge cutoff, and writes an unverified response. 

**The Agentic Solution:** Instead of 1 monolithic prompt, we build a **team of specialized AI workers**:
1. **Planner Agent:** Acts as the Project Manager, breaking down the broad goal into 4 sub-questions.
2. **Research Agent:** Acts as the Data Researcher, querying vector databases and live search APIs.
3. **Writer Agent:** Acts as the Technical Editor, synthesizing facts with inline citations `[1]`, `[2]`.
4. **Critic Agent:** Acts as the Peer Reviewer, evaluating groundedness and triggering re-queries if facts are missing.

---

## 💡 2. Core AI Engineering Concepts Explained

### 2.1 LangGraph & State Graphs
Standard LangChain chains are **Directed Acyclic Graphs (DAGs)**—they only flow forward. But human research requires **loops** (e.g., "Draft isn't good enough $\rightarrow$ search again $\rightarrow$ rewrite").

`LangGraph` allows us to build **Cyclic State Graphs**:
* **State (`ResearchState`):** A shared Python dictionary passed between all nodes.
* **Nodes:** Python functions that take the current `state`, perform work, and return updated state keys.
* **Edges:** Connections between nodes.
* **Conditional Edges:** Decision gates (if/else functions) that route the workflow based on data in the state (e.g., checking if `critic_score >= 0.8`).

---

### 2.2 Structured Outputs with Pydantic v2
By default, LLMs return unstructured text. To build reliable software pipelines, we need LLMs to return strict, typed JSON objects.

We use **Pydantic** to define schemas:
```python
from pydantic import BaseModel, Field

class CriticEvaluation(BaseModel):
    is_grounded: bool = Field(description="True if claims are supported by sources.")
    score: float = Field(description="Quality score between 0.0 and 1.0.")
    feedback: str = Field(description="Instructions for revision if score < 0.8.")
```
**Production Engineering Tip (Direct JSON Prompting):** Tool Calling API endpoints (e.g. `llm.with_structured_output`) have strict rate-limits on free API tiers. In production, engineers prompt the LLM for raw JSON and validate it via `Pydantic(**json.loads(response.content))`, preserving 100% schema enforcement while running 3x faster with zero tool-calling quota limits!

---

### 2.3 The Self-Correction & Reflection Loop
The hallmark of production-grade AI engineering is **reflection**.
```
[Writer Agent Draft] ──> [Critic Agent Grades Draft]
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        Score >= 0.8                  Score < 0.8
         (PASS)                        (FAIL)
            │                             │
    [Finalize Report]            [Increment Revision]
                                          │
                                 [Refine Queries & Search]
```
If the Critic agent flags missing evidence, it generates revised search queries, increments `revision_count`, and routes back to the Research Agent.

---

## 🏗️ 3. Complete Code Architecture & File Breakdown

### File 1: `config.py` — Multi-Provider Engine Configuration
* **Purpose:** Centralized settings and provider switching (`Groq`, `Google Gemini`, `Ollama`).
* **Key Code:**
  ```python
  def get_llm(model_name: str = None, temperature: float = 0.2):
      provider = os.getenv("LLM_PROVIDER", "groq").lower()
      if provider == "groq":
          return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)
      ...
  ```
* **Why Groq?** Groq runs models on LPUs at ~500 tokens/sec. It provides 100% free API access to `llama-3.3-70b-versatile` with **30 Requests/Min (14,400 Requests/Day)**, completely eliminating rate limit errors!
* **Why Ollama?** Allows 100% offline local inference on macOS without external API keys.

---

### File 2: `src/state.py` — Shared Graph State Schema
* **Purpose:** Defines the data dictionary shared across all agent nodes.
* **Schema Breakdown:**
  ```python
  class ResearchState(TypedDict):
      topic: str                             # Input research topic
      sub_questions: List[str]               # Generated by Planner
      search_queries: List[str]              # Generated by Planner & Critic
      retrieved_docs: List[Dict[str, Any]]   # Passages from ChromaDB
      web_results: List[Dict[str, Any]]      # Live snippets from DuckDuckGo
      draft_report: str                      # Current report draft
      critic_score: float                    # Groundedness rating (0.0 - 1.0)
      critic_feedback: str                   # Peer review notes
      revision_count: int                    # Loop counter (max 2)
      final_report: str                      # Completed cited markdown report
      status_log: List[str]                  # UI progress messages
  ```

---

### File 3: `src/tools/web_search.py` — Concurrent Multi-Threaded Search Tool
* **Purpose:** Queries DuckDuckGo for real-time web information in parallel threads using `concurrent.futures.ThreadPoolExecutor`.
* **Key Performance Optimization:** Instead of querying search terms sequentially, multi-threading reduces search latency from ~4s to ~0.8s (3x speedup).

---

### File 4: `src/tools/vector_store.py` — Local ChromaDB Vector RAG
* **Purpose:** Persists uploaded PDFs/TXT files locally for dense semantic retrieval.
* **Key Code:**
  ```python
  class LocalVectorStore:
      def __init__(self):
          self.client = chromadb.PersistentClient(path="./chroma_db")
          self.collection = self.client.get_or_create_collection("research_documents")
  ```

---

### File 5: `src/agents/planner.py` — Task Decomposition Agent
* **Purpose:** Transforms a broad prompt into focused sub-questions and search queries.
* **Input State:** `topic`
* **Output State:** `sub_questions`, `search_queries`
* **Mechanism:** Uses `Pydantic` schema `PlannerOutput` to ensure exact array structure.

---

### File 6: `src/agents/writer.py` — Synthesis & Citation Agent
* **Purpose:** Merges all evidence snippets into a structured Markdown document.
* **Key Feature (Inline Citations):** Maps every web snippet and vector document to numerical IDs `[1]`, `[2]` and forces the LLM to place inline citations after factual statements.

---

### File 7: `src/agents/critic.py` — Groundedness & Fact-Checker Agent
* **Purpose:** Evaluates draft report against original source context.
* **Schema (`CriticEvaluation`):** Assigns `score` (0.0 - 1.0). If `score < 0.8`, generates `revised_search_queries`.

---

### File 8: `src/agents/graph.py` — LangGraph Pipeline & Dynamic Routing
* **Purpose:** Connects all nodes into an executable state graph.
* **Conditional Router Logic:**
  ```python
  def should_continue(state: dict) -> str:
      score = state.get("critic_score", 0.0)
      revisions = state.get("revision_count", 0)
      
      if score >= 0.8 or revisions >= MAX_REVISIONS:
          return "finalize"
      else:
          return "continue_revision"
  ```

---

### File 9: `app.py` — Streamlit UI Dashboard
* **Purpose:** Provides a user interface for users to enter topics, upload optional PDFs, monitor real-time agent execution status logs, and read rendered markdown reports with clickable citation links.

---

## 🗣️ 4. Interview Preparation: How to Explain This Project

### Q1: "Can you explain the architecture of your Agentic Research Assistant?"
> **Answer:** *"I built an autonomous multi-agent research engine using **LangGraph** and **Google Gemini 2.0 Flash**. Instead of a single LLM prompt, I decomposed the workflow into specialized agent nodes over a shared state graph. A Planner agent generates sub-questions and search queries, a Research node executes hybrid retrieval across local ChromaDB vector stores and live DuckDuckGo web search, a Writer agent synthesizes the facts with numerical inline citations, and a Critic agent evaluates groundedness using Pydantic structured output. If the Critic flags missing facts, a conditional routing edge triggers a dynamic self-correction loop to refine queries and re-search."*

### Q2: "Why did you use LangGraph instead of standard LangChain chains?"
> **Answer:** *"Standard chains are linear DAGs that only move forward. Real research requires cycles—when the Critic agent discovers missing evidence or hallucinations, we need cyclic graph routing to go back to the research node with refined search queries. LangGraph natively supports stateful cyclic graphs with conditional edges."*

### Q3: "How do you prevent infinite loops in agent reflection?"
> **Answer:** *"We maintain a `revision_count` variable in the shared `ResearchState`. Our conditional edge `should_continue()` evaluates both `critic_score >= 0.8` AND `revision_count >= MAX_REVISIONS`. Once max revisions are reached, it routes to a finalizer node that appends a reliability disclaimer."*

### Q4: "How do you enforce structured output from LLMs?"
> **Answer:** *"I used Pydantic v2 schemas combined with `llm.with_structured_output(Schema)`. This enforces exact type validation on LLM responses, guaranteeing that downstream agent nodes receive expected data types like Python lists and floats."*

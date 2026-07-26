# ⚡ RAG & Agentic Systems Engineering Portfolio

A high-performance repository demonstrating the evolution from **Fundamental RAG Ingestion & Vector Retrieval Pipelines** to **Autonomous Multi-Agent Collaboration Engines**.

---

## 📂 Sub-Projects Overview

| Project Name | Architecture | Key Technologies | Status |
| :--- | :--- | :--- | :--- |
| 🤖 **[Agentic Research Assistant](Agentic_Research_Assistant/)** | Autonomous Multi-Agent Graph + Dynamic Reflection Loop | LangGraph, Gemini 2.0, ChromaDB, DuckDuckGo Search, Pydantic, Streamlit | **Featured Project** |
| 📚 **[Modular RAG Pipeline](Modular_RAG_Pipeline/)** | Multi-Source Heterogeneous Ingestion & Hybrid Vector Search | Python, LangChain, SentenceTransformers, ChromaDB, Jupyter | **Study Project** |

---

## 🤖 1. Agentic Research Assistant (Featured Multi-Agent System)
📁 **[Explore Agentic Research Assistant Sub-Project](Agentic_Research_Assistant/)** \| 📘 **[Read Deep-Dive Technical Tutorial](Agentic_Research_Assistant/AGENTIC_TUTORIAL.md)**

An autonomous multi-agent research engine built with **LangGraph** that decomposes complex prompts into sub-questions, executes parallel retrieval across local vector databases and live web search, fact-checks draft reports for hallucinations using a Critic Agent, and automatically loops back to refine search queries if quality thresholds are not met.

### Key Capabilities:
* **Task Decomposition:** Planner Agent using Pydantic structured output.
* **Hybrid Retrieval:** Dense vector search (`ChromaDB`) + live web search (`DuckDuckGo`).
* **Self-Correction & Reflection Loop:** Automated groundedness scoring (0.0 to 1.0) with dynamic query re-submission.
* **Interactive Dashboard:** Full Streamlit UI with real-time agent execution logs and inline citations.

---

## 📚 2. Modular RAG Ingestion Pipeline (Study Project)
📁 **[Explore Modular RAG Pipeline Sub-Project](Modular_RAG_Pipeline/)**

A production-minded, modular RAG ingestion & retrieval pipeline built to handle heterogeneous data sources (PDFs, CSVs, plain text, SQL databases) with smart character chunking, embedding generation via `SentenceTransformers`, and persistent vector indexing.

### Key Capabilities:
* **Heterogeneous Data Loaders:** Custom loader for structured & unstructured data.
* **Smart Tabular Chunking:** Preserves CSV and SQL table structure during chunking.
* **Interactive Tutorial:** Accompanied by Jupyter Notebook ([`document.ipynb`](Modular_RAG_Pipeline/NoteBook/document.ipynb)) and deep-dive technical reference guides ([`RAG_TUTORIAL.md`](Modular_RAG_Pipeline/RAG_TUTORIAL.md)).

---

## 🛠️ Repository Architecture

```
RAG/
├── 🤖 Agentic_Research_Assistant/   # Autonomous Multi-Agent StateGraph System
│   ├── app.py                       # Streamlit UI Dashboard
│   ├── config.py                    # Gemini & LLM settings
│   ├── requirements.txt             # Agentic dependencies
│   ├── README.md                    # Detailed multi-agent system documentation
│   └── src/                         # Planner, Research, Writer, & Critic Agents
│
├── 📚 Modular_RAG_Pipeline/        # Fundamental RAG Data Loader & Ingestion Study Project
│   ├── README.md                    # Dedicated study project guide
│   ├── RAG_TUTORIAL.md              # Deep-dive RAG technical guide
│   ├── SETUP_GUIDE.md               # Environment setup instructions
│   ├── NoteBook/                    # Interactive Jupyter notebook
│   ├── Data/                        # Test datasets & vector storage
│   └── src/                         # Ingestion, Embedding, & Vector Store modules
│
└── README.md                        # Master Portfolio Hub (This file)
```

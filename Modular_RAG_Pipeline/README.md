# 📚 Modular RAG Ingestion & Vector Retrieval Pipeline

A production-minded, modular **Retrieval-Augmented Generation (RAG)** pipeline implementation in Python. This study project demonstrates how to ingest, validate, and chunk multi-source heterogenous data (PDFs, CSVs, plain text, and relational SQL databases), generate high-dimensional semantic embeddings, and index them into local vector stores.

---

## 🎯 Project Highlights & Learning Objectives

* **Multi-Source Data Ingestion**: Custom modular loader (`src/data_loader.py`) designed to handle PDFs, text files, CSV tables, and SQL database queries with built-in path validation and exception handling.
* **Smart Hybrid Chunking**: Implements `RecursiveCharacterTextSplitter` for unstructured text while dynamically preserving structured tabular rows (CSV/SQL) to prevent contextual fragmentation.
* **Vector Indexing & Local Storage**: Generates semantic embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`) and indexes vectors locally using `ChromaDB` persistent storage (`src/vector_store.py`).
* **Embedding Manager**: Centralized embedding model lifecycle manager (`src/embedding_manager.py`).

---

## 📂 Sub-Project Directory Structure

```
Modular_RAG_Pipeline/
├── Data/                  # Local storage for source documents & persistent vector DB
├── NoteBook/              
│   └── document.ipynb     # Interactive pipeline development & verification notebook
├── src/                   
│   ├── data_loader.py     # Modular ingestion module for PDFs, CSVs, TXT, & SQL DBs
│   ├── embedding_manager.py # SentenceTransformers embedding model manager
│   └── vector_store.py    # ChromaDB indexing & vector retrieval module
├── pyrightconfig.json     # Language Server environment sync config
├── requirements.txt       # Dependencies (LangChain, ChromaDB, SentenceTransformers)
├── SETUP_GUIDE.md         # Environment setup and troubleshooting documentation
└── RAG_TUTORIAL.md        # Technical reference guide & RAG pipeline breakdown
```

---

## 📖 Technical Documentation & Deep-Dives

* **⚙️ [Environment & Setup Guide](SETUP_GUIDE.md)**: Instructions for environment setup, kernel registration, and linter sync.
* **📘 [RAG System Deep-Dive & Reference](RAG_TUTORIAL.md)**: Comprehensive architectural notes covering ingestion strategies, chunking math, embedding models, and vector database comparisons.

---

## 🚀 Quickstart & Notebook Execution

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Interactive Notebook**:
   Open [`NoteBook/document.ipynb`](NoteBook/document.ipynb) in Jupyter / VS Code and execute the pipeline cells step-by-step.

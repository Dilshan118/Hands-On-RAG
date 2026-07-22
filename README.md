# ⚡ Modular RAG Ingestion & Vector Retrieval Engine

A production-minded, modular **Retrieval-Augmented Generation (RAG)** pipeline implementation in Python. This project demonstrates how to ingest, validate, and chunk multi-source heterogenous data (PDFs, CSVs, plain text, and relational SQL databases), generate high-dimensional semantic embeddings, and index them into local vector stores.

---

## 🎯 Architecture & Project Highlights

* **Multi-Source Data Ingestion**: Custom modular loader (`src/data_loader.py`) designed to handle PDFs, text files, CSV tables, and SQL database queries with built-in path validation and exception handling.
* **Smart Hybrid Chunking**: Implements `RecursiveCharacterTextSplitter` for unstructured text while dynamically preserving structured tabular rows (CSV/SQL) to prevent contextual fragmentation.
* **Vector Indexing & Local Storage**: Generates semantic embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`) and indexes vectors locally using `ChromaDB` persistent storage.
* **Robust Workspace & Environment Control**: Fully configured Conda environment (`rag_env`) and Language Server configuration (`pyrightconfig.json`) for seamless development.

---

## 📂 Project Architecture

```
RAG/
├── .vscode/               # Workspace interpreter settings
├── Data/                  # Local storage for source documents & persistent vector DB
├── NoteBook/              
│   └── document.ipynb     # Interactive pipeline development & verification notebook
├── src/                   
│   └── data_loader.py     # Modular ingestion module for PDFs, CSVs, TXT, & SQL DBs
├── .gitignore             # Version control exclusions (ignores DBs, vectors, API keys)
├── pyrightconfig.json     # Language Server environment sync config
├── requirements.txt       # Dependencies (LangChain, ChromaDB, SentenceTransformers)
├── SETUP_GUIDE.md         # Environment setup and troubleshooting documentation
└── RAG_TUTORIAL.md        # Technical reference guide & RAG pipeline breakdown
```

---

## 📖 Technical Documentation & Guides

* **⚙️ [Environment & Setup Guide](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/RAG/SETUP_GUIDE.md)**: Instructions for activating `rag_env`, resolving Linter errors, and Jupyter kernel registration.
* **📘 [RAG System Deep-Dive & Reference](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/RAG/RAG_TUTORIAL.md)**: Comprehensive architectural notes covering ingestion strategies, chunking math, embedding models, and vector database comparisons.

---

## 🚀 Quickstart

1. **Clone & Environment Setup**: Ensure Conda environment `rag_env` is active and dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Ingestion & Search Pipeline**: Open [document.ipynb](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/RAG/NoteBook/document.ipynb), set your kernel to `Python (rag_env)`, and execute the pipeline cells.

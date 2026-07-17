# 🤖 Hands-On RAG: A Beginner's Ingestion & Retrieval Playground

Welcome to the **RAG (Retrieval-Augmented Generation)** learning repository! This project serves as a step-by-step playground designed to document, learn, and implement RAG systems from scratch using **Python**, **Conda**, and **LangChain**.

If you are a beginner looking to understand how to connect Large Language Models to custom private data (like PDFs, CSVs, and databases), this repository is built specifically for you.

---

## 🗺️ Learning Roadmap & Guides

We have split our learning process and documentation into specific, easy-to-follow files:

* **📖 [Beginner RAG Tutorial & Theory](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/RAG/RAG_TUTORIAL.md)**  
  *Our core learning document.* It contains intuitive startup problem examples, fine-tuning vs. RAG comparisons, overall pipeline architecture flowcharts, and step-by-step code tutorials for mixed-source loading (PDF, CSV, SQL) and split strategies.
* **🛠️ [Python & Conda Setup Guide](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/RAG/SETUP_GUIDE.md)**  
  *How to set up your environment.* Covers creating your `rag_env` Conda environment, resolving VS Code interpreter mismatches, and clearing up language server "missing-import" red lines.

---

## 📂 Project Structure

```
RAG/
├── .vscode/               # VS Code workspace settings
├── Data/                  # Local directory for raw documents (PDFs, CSVs, etc.)
├── NoteBook/              
│   └── document.ipynb     # Jupyter Notebook where we run the RAG steps
├── src/                   
│   └── data_loader.py     # Unified Python module to load PDFs, CSVs, TXT, & DBs
├── .gitignore             # Git exclusion lists (ignores models, DBs, secret keys)
├── pyrightconfig.json     # Configuration to sync the linter to our Conda env
├── requirements.txt       # Necessary packages (LangChain, PyMuPDF, dotenv)
├── SETUP_GUIDE.md         # Step-by-step environment registration manual
└── RAG_TUTORIAL.md        # Deep dive into RAG theory, chunking, and code snippets
```

---

## 🚀 Getting Started

1. **Clone & Open:** Open this folder in VS Code or Antigravity IDE.
2. **Setup the Environment:** Follow the activation steps inside [SETUP_GUIDE.md](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/RAG/SETUP_GUIDE.md).
3. **Run the Code:** Open [document.ipynb](file:///Users/dilshanrajapakshe/Documents/SLIIT/GitHub/Data%20science/RAG/NoteBook/document.ipynb), select your registered Conda kernel (`Python (rag_env)`), and run the cells!

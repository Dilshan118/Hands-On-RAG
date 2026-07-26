import os
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import SQLDatabaseLoader

def load_multi_source_data(
    pdf_files: Optional[List[str]] = None,
    text_files: Optional[List[str]] = None,
    csv_files: Optional[List[str]] = None,
    db_configs: Optional[List[Dict[str, str]]] = None
) -> List[Document]:
    """
    Loads documents from multiple sources (PDF, TXT, CSV, and SQL Databases)
    and returns a single unified list of LangChain Document objects.
    
    Args:
        pdf_files: List of file paths to PDF documents.
        text_files: List of file paths to Text/Markdown documents.
        csv_files: List of file paths to CSV documents.
        db_configs: List of dictionaries containing database configuration details:
                    [{"uri": "sqlite:///db.sqlite", "query": "SELECT * FROM table"}]
                    
    Returns:
        List[Document]: Combined list of LangChain Document objects.
    """
    documents = []
    
    # 1. Load PDFs
    if pdf_files:
        for file_path in pdf_files:
            if os.path.exists(file_path):
                print(f"Loading PDF: {file_path}...")
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"Warning: PDF file not found at {file_path}")
                
    # 2. Load Text/MD Files
    if text_files:
        for file_path in text_files:
            if os.path.exists(file_path):
                print(f"Loading Text: {file_path}...")
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"Warning: Text file not found at {file_path}")
                
    # 3. Load CSVs
    if csv_files:
        for file_path in csv_files:
            if os.path.exists(file_path):
                print(f"Loading CSV: {file_path}...")
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"Warning: CSV file not found at {file_path}")
                
    # 4. Load Databases
    if db_configs:
        for config in db_configs:
            db_uri = config.get("uri")
            query = config.get("query")
            if db_uri and query:
                print(f"Loading from Database: {db_uri} with query...")
                try:
                    db = SQLDatabase.from_uri(db_uri)
                    loader = SQLDatabaseLoader.from_query(query, db)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading from Database ({db_uri}): {e}")
            else:
                print("Warning: Database configuration missing 'uri' or 'query'")
                
    print(f"--- Ingestion Complete: Unified {len(documents)} document pages/rows ---")
    return documents

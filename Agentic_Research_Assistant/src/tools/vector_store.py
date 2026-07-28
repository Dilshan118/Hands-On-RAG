"""
vector_store.py — Local ChromaDB Vector Database Manager
=========================================================

📚 BEGINNER EXPLANATION — What is a Vector Store?
    A vector store is a specialized database that stores text as numerical arrays
    (vectors) and can find "semantically similar" text using mathematical distance.

    NORMAL DATABASE:  "Find rows WHERE name = 'Alice'"  (exact keyword match)
    VECTOR DATABASE:  "Find chunks SIMILAR TO 'time off policy'"  (meaning match)

    When you upload a PDF about company leave policies, the text gets converted
    into vectors (lists of numbers). Later, when someone asks "how many vacation
    days do I get?", the query is ALSO converted to a vector, and the database
    finds which stored vectors are mathematically CLOSEST — even though the words
    are completely different!

🏗️ DESIGN PATTERN — Repository Pattern:
    The LocalVectorStore class encapsulates all database operations
    (add, search, collection management) behind a clean interface.
    The Research node in graph.py only calls:
        store = LocalVectorStore()
        results = store.search(queries)
    
    It doesn't need to know about ChromaDB internals, collection names,
    or query formatting. This is called ENCAPSULATION.

📚 WHY CHROMADB?
    - Zero setup: runs as a Python library, no separate server needed
    - Persistent: saves data to disk (survives app restarts)
    - Built-in embedding: can generate embeddings automatically for text
    - Free and open-source
    - Perfect for prototyping and learning
    
    For production at scale, you'd consider Pinecone, Weaviate, or pgvector.

📁 ARCHITECTURE ROLE:
    LAYER 4 (Retrieval & Tool Concurrency) — Called by the Research node.
    Also called by app.py for PDF ingestion.
"""

import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR

# Setup module-level logger
logger = logging.getLogger(__name__)


class LocalVectorStore:
    """
    Local ChromaDB Vector Database Manager for RAG document retrieval.

    📚 KEY CONCEPTS FOR BEGINNERS:

    COLLECTION:
        A "collection" in ChromaDB is like a TABLE in a regular database.
        It's a named container that holds documents and their vector embeddings.
        Our default collection is called "research_documents".

    PERSISTENT CLIENT:
        ChromaDB's PersistentClient saves data to disk at CHROMA_PERSIST_DIR.
        This means uploaded documents survive app restarts — you don't need
        to re-upload them every time you start the Streamlit app.

    AUTOMATIC EMBEDDING:
        ChromaDB can generate embeddings internally when you add documents
        as text (using its default embedding function). This means we don't
        need to manually compute embeddings — we just pass raw text strings.

    📚 LIFECYCLE:
        1. __init__() → Connect to ChromaDB, create/get collection
        2. add_documents() → Store text chunks (called during PDF upload)
        3. search() → Find relevant chunks for queries (called during research)
    """

    def __init__(self, collection_name: str = "research_documents"):
        """
        Initialize the ChromaDB persistent client and get/create the collection.

        📚 WHAT HAPPENS HERE:
            1. Ensure the storage directory exists (os.makedirs with exist_ok=True)
            2. Create a PersistentClient pointing to that directory
            3. Get or create the named collection
            
            get_or_create_collection() is IDEMPOTENT — calling it multiple times
            with the same name returns the SAME collection. It won't duplicate data.

        Args:
            collection_name: Name of the ChromaDB collection to use.
                            Default: "research_documents"
        """
        # Ensure the storage directory exists (creates parents if needed)
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # Connect to ChromaDB with persistent disk storage
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Get the collection if it exists, or create it if it doesn't
        self.collection = self.client.get_or_create_collection(name=collection_name)

        logger.info(
            f"ChromaDB initialized: collection='{collection_name}', "
            f"existing_docs={self.collection.count()}, "
            f"storage='{CHROMA_PERSIST_DIR}'"
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Adds text documents/chunks to the ChromaDB collection.

        📚 HOW CHROMADB STORES DOCUMENTS:
            Each document needs three things:
            1. id:       A unique string identifier (we generate "doc_0", "doc_1", etc.)
            2. document: The actual text content
            3. metadata: Optional key-value pairs (e.g., {"source": "paper.pdf"})
            
            ChromaDB automatically generates vector embeddings for the text
            using its built-in default embedding function (a small transformer model).

        📚 IMPORTANT LIMITATION:
            Our simple "doc_0", "doc_1" ID scheme means RE-UPLOADING documents
            will OVERWRITE previous entries. For production, you'd want
            deterministic content-based hashes (like in the Modular RAG Pipeline's
            VectorStoreManager which uses MD5 hashing).

        Args:
            documents: List of text strings to store in the vector database.
            metadatas: Optional list of metadata dicts (one per document).
                      If None, defaults to {"source": "uploaded_doc"} for each.
        """
        # Generate simple sequential IDs
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Provide default metadata if none supplied
        if not metadatas:
            metadatas = [{"source": "uploaded_doc"} for _ in range(len(documents))]

        # Add documents to ChromaDB (embeddings are generated automatically)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Added {len(documents)} documents to ChromaDB collection.")

    def search(self, queries: List[str], n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Queries the vector collection for semantically similar document chunks.

        📚 HOW SIMILARITY SEARCH WORKS:
            1. ChromaDB converts the query text into a vector (same embedding model)
            2. It compares the query vector against ALL stored document vectors
            3. It returns the top-N most similar documents (closest in vector space)
            
            "Similar" means the query and document share semantic meaning,
            even if they use completely different words.
            
            Example:
                Query: "vacation days"
                Match: "Employees receive 15 days of annual paid leave"
                Why:   Both are about time off from work (semantic similarity)

        📚 SAFETY CHECKS:
            - If the collection is empty (count == 0), we return an empty list
              immediately instead of making a pointless query
            - We cap n_results to the actual collection size to prevent
              ChromaDB from raising an error
            - Each individual query is wrapped in try/except to prevent
              one bad query from blocking all results

        Args:
            queries: List of search query strings.
            n_results: Maximum results per query (default: 3).

        Returns:
            List of result dicts with: query, content, source.
        """
        results_list = []

        # Early return if no documents have been indexed
        if self.collection.count() == 0:
            logger.info("Vector store is empty — no documents to search.")
            return results_list

        for query in queries:
            try:
                # Query ChromaDB — it handles embedding the query text internally
                res = self.collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self.collection.count())
                )

                # Extract documents and metadata from the nested response structure
                # ChromaDB returns: {"documents": [[doc1, doc2]], "metadatas": [[meta1, meta2]]}
                # The outer list is for multiple queries, inner list is for multiple results
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]

                for doc, meta in zip(docs, metas):
                    results_list.append({
                        "query": query,
                        "content": doc,
                        "source": meta.get("source", "local_doc")
                    })
            except Exception as e:
                logger.warning(f"Vector search failed for query '{query}': {e}")
                continue

        logger.info(f"Vector search returned {len(results_list)} results for {len(queries)} queries.")
        return results_list

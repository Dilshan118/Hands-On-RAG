import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR

class LocalVectorStore:
    """
    Local ChromaDB Vector Database Manager for RAG document retrieval.
    """
    def __init__(self, collection_name: str = "research_documents"):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Adds text documents/chunks to ChromaDB collection."""
        ids = [f"doc_{i}" for i in range(len(documents))]
        if not metadatas:
            metadatas = [{"source": "uploaded_doc"} for _ in range(len(documents))]
            
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, queries: List[str], n_results: int = 3) -> List[Dict[str, Any]]:
        """Queries the vector collection for a list of search query strings."""
        results_list = []
        if self.collection.count() == 0:
            return results_list
            
        for query in queries:
            try:
                res = self.collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self.collection.count())
                )
                
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                
                for doc, meta in zip(docs, metas):
                    results_list.append({
                        "query": query,
                        "content": doc,
                        "source": meta.get("source", "local_doc")
                    })
            except Exception as e:
                continue
                
        return results_list

import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

class VectorStoreManager:
    """Handles vector indexing and similarity search across local (ChromaDB) and cloud (Pinecone) databases."""

    def __init__(self, provider: str = "chroma", collection_name: str = "rag_docs"):
        self.provider = provider.lower()
        self.collection_name = collection_name
        
        if self.provider == "chroma":
            import chromadb
            # Local disk storage
            data_dir = os.path.join(os.path.dirname(__file__), "..", "Data", "chroma_db")
            self.client = chromadb.PersistentClient(path=data_dir)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            print(f"✅ ChromaDB initialized locally at '{data_dir}' (Collection: '{collection_name}')")
            
        elif self.provider == "pinecone":
            from pinecone import Pinecone
            api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME") or "elegant-walnut"
            
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")
                
            pc = Pinecone(api_key=api_key)
            self.index = pc.Index(index_name)
            print(f"✅ Connected to Cloud Pinecone Index: '{index_name}'")
        else:
            raise ValueError(f"Unsupported provider '{self.provider}'. Choose 'chroma' or 'pinecone'.")

    def add_documents(self, documents: List[Any], embeddings: List[List[float]]) -> None:
        """
        Stores document text chunks, their embeddings, and metadata into the vector database.
        """
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings.")

        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if self.provider == "chroma":
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"✅ Added {len(documents)} document chunks to ChromaDB.")

        elif self.provider == "pinecone":
            # Pinecone expects vectors formatted as: (id, values, metadata)
            vectors_to_upsert = []
            for doc_id, emb, text, meta in zip(ids, embeddings, texts, metadatas):
                clean_meta = dict(meta) if meta else {}
                clean_meta["text"] = text
                vectors_to_upsert.append((doc_id, emb, clean_meta))

            self.index.upsert(vectors=vectors_to_upsert)
            print(f"✅ Upserted {len(documents)} vectors to Cloud Pinecone index.")

    def search_similarity(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs similarity search against the vector database using a query embedding vector.
        """
        results = []
        
        if self.provider == "chroma":
            chroma_res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if chroma_res and chroma_res.get("documents"):
                for doc, meta in zip(chroma_res["documents"][0], chroma_res["metadatas"][0]):
                    results.append({"text": doc, "metadata": meta})
                    
        elif self.provider == "pinecone":
            pinecone_res = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            if pinecone_res and pinecone_res.get("matches"):
                for match in pinecone_res["matches"]:
                    meta = match.get("metadata", {})
                    text = meta.get("text", "")
                    results.append({"text": text, "metadata": meta, "score": match.get("score")})
                    
        return results


import os
from typing import List, Union, Optional
import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

class EmbeddingManager:
    """Handles document embedding generation using local SentenceTransformer or Google Gemini API."""

    def __init__(self, provider: str = "local", model_name: str = None, dimension: Optional[int] = None):
        """
        Args:
            provider: 'local' (HuggingFace) or 'google' (Google Gemini API)
            model_name: Model name. Defaults to 'all-MiniLM-L6-v2' for local,
                        or 'models/gemini-embedding-001' / 'text-embedding-004' for Google.
            dimension: Optional target vector dimension (e.g. 1024 for Pinecone).
        """
        self.provider = provider.lower()
        self.dimension = dimension
        
        if self.provider == "local":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self.model = None
            self._load_local_model()
        elif self.provider == "google":
            self.model_name = model_name or "models/gemini-embedding-001"
            self.model = None
            self._load_google_model()
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Choose 'local' or 'google'.")

    def _load_local_model(self):
        """Load local SentenceTransformer model"""
        from sentence_transformers import SentenceTransformer
        try:
            print(f"Loading local embedding model: '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Local Model loaded. Vector dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ Error loading local model {self.model_name}: {e}")
            raise

    def _load_google_model(self):
        """Load Google Gemini Embedding model using GOOGLE_API_KEY"""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        
        try:
            print(f"Loading Google Gemini embedding model: '{self.model_name}'...")
            kwargs = {
                "model": self.model_name,
                "google_api_key": api_key
            }
            if self.dimension:
                kwargs["output_dimensionality"] = self.dimension

            self.model = GoogleGenerativeAIEmbeddings(**kwargs)
            dim_msg = f" ({self.dimension} dimensions)" if self.dimension else ""
            print(f"✅ Google Gemini Embedding Model initialized successfully!{dim_msg}")
        except Exception as e:
            print(f"❌ Error initializing Google model {self.model_name}: {e}")
            raise

    def _adjust_dimension(self, vector: List[float]) -> List[float]:
        """Adjusts vector dimensionality to match self.dimension if specified."""
        if not self.dimension or len(vector) == self.dimension:
            return vector
        
        # Truncate if vector is longer than target dimension
        if len(vector) > self.dimension:
            return vector[:self.dimension]
        
        # Zero-pad if vector is shorter than target dimension
        padded = vector + [0.0] * (self.dimension - len(vector))
        return padded

    def generate_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for a single text query"""
        if not self.model:
            raise ValueError("Model is not initialized.")
            
        if self.provider == "local":
            raw_vec = self.model.encode(text).tolist()
        elif self.provider == "google":
            raw_vec = self.model.embed_query(text)
            
        return self._adjust_dimension(raw_vec)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text chunks"""
        if not self.model:
            raise ValueError("Model is not initialized.")
            
        if self.provider == "local":
            raw_vecs = self.model.encode(texts, show_progress_bar=True).tolist()
        elif self.provider == "google":
            raw_vecs = self.model.embed_documents(texts)
            
        return [self._adjust_dimension(v) for v in raw_vecs]

import os
from typing import List, Union
import numpy as np
from dotenv import load_dotenv, find_dotenv

# Search parent folders for .env if executing from subdirectories like NoteBook/
load_dotenv(find_dotenv())

class EmbeddingManager:
    """Handles document embedding generation using local SentenceTransformer or Google Gemini API."""

    def __init__(self, provider: str = "local", model_name: str = None):
        """
        Args:
            provider: 'local' (HuggingFace) or 'google' (Google Gemini API)
            model_name: Model name. Defaults to 'all-MiniLM-L6-v2' for local,
                        or 'models/text-embedding-004' for Google.
        """
        self.provider = provider.lower()
        
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
            self.model = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=api_key
            )
            print("✅ Google Gemini Embedding Model initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing Google model {self.model_name}: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for a single text query"""
        if not self.model:
            raise ValueError("Model is not initialized.")
            
        if self.provider == "local":
            return self.model.encode(text).tolist()
        elif self.provider == "google":
            return self.model.embed_query(text)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text chunks"""
        if not self.model:
            raise ValueError("Model is not initialized.")
            
        if self.provider == "local":
            return self.model.encode(texts, show_progress_bar=True).tolist()
        elif self.provider == "google":
            return self.model.embed_documents(texts)


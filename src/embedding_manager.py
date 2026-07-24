from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully. Vector dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ Error loading model {self.model_name}: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a vector embedding for a single text query"""
        if not self.model:
            raise ValueError("Model is not loaded.")
        return self.model.encode(text)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate vector embeddings for a list of text chunks"""
        if not self.model:
            raise ValueError("Model is not loaded.")
        return self.model.encode(texts, show_progress_bar=True)

"""
GPU Embedding manager using sentence-transformers.

Provides EmbeddingManager.generate_embeddings(texts) -> numpy array
"""

import numpy as np
from typing import List

# Attempt to import SentenceTransformer; if missing, raise helpful error
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence_transformers is required. Install with: pip install sentence-transformers"
    ) from e


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        """
        model_name: huggingface/sentence-transformers model name
        device: 'cuda' or 'cpu'. Use 'cuda' to leverage NVIDIA GPU.
        """
        print(f"[embeddings] Loading SentenceTransformer model '{model_name}' on device='{device}'")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[embeddings] Model loaded. Embedding dim = {self.dim}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of strings.

        Returns:
          numpy array shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.dim))
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print(f"[embeddings] Generated embeddings: {embeddings.shape}")
        return embeddings

"""
Simple RAG retriever module.

Exposes RAGRetriever that:
  - creates a query embedding via EmbeddingManager
  - queries Chroma collection
  - returns top-k docs with similarity scores
"""

from typing import List, Dict, Any
import numpy as np

# local imports - these will work if pipeline runs from package root or script context
try:
    from src.vectorstore import VectorStore
    from src.embeddings import EmbeddingManager
except Exception:
    from vectorstore import VectorStore
    from embeddings import EmbeddingManager


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents for the query.

        Returns a list of dicts:
          { 'content': str, 'metadata': dict, 'score': float, 'rank': int }
        """
        if not query:
            return []

        # Create query embedding
        q_emb = self.embedding_manager.generate_embeddings([query])[0]

        # Query chroma collection
        results = self.vector_store.collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)

        # results structure: {'ids': [...], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]  # chroma returns distance; convert to similarity

        retrieved = []
        for i, (doc_text, md, dist) in enumerate(zip(docs, metadatas, distances)):
            similarity = 1.0 - float(dist) if dist is not None else None
            retrieved.append({
                "content": doc_text,
                "metadata": md,
                "score": similarity,
                "rank": i + 1
            })

        return retrieved

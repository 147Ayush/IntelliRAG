"""
VectorStore wrapper for ChromaDB with safe batch insertion.
"""

import os
import uuid
import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "documents"):
        """
        Initialize a persistent ChromaDB collection.

        Args:
            persist_dir: Directory where the vector database will be stored.
            collection_name: Name of the Chroma collection.
        """

        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(allow_reset=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"[vectorstore] Initialized collection '{collection_name}' (persist_dir='{persist_dir}')")

    def file_exists(self, filename: str) -> bool:
        """
        Check if the file has already been indexed by searching metadata.
        """
        if not filename:
            return False

        results = self.collection.get(where={"source_file": filename})
        return len(results.get("ids", [])) > 0

    def add(self, chunks, vectors, batch_size: int = 5000):
        """
        Add documents + embeddings into ChromaDB in safe batches.

        Args:
            chunks: List of chunk objects containing page_content + metadata.
            vectors: List/ndarray of embeddings.
            batch_size: Maximum size per batch (default: 5000)
        """

        # Ensure each chunk has a unique ID
        ids = []
        for c in chunks:
            # If metadata lacks ID, create a new one
            if "id" not in c.metadata:
                c.metadata["id"] = str(uuid.uuid4())
            ids.append(c.metadata["id"])

        docs = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        total = len(chunks)
        print(f"[vectorstore] Adding {total} vectors in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            end = i + batch_size

            batch_ids = ids[i:end]
            batch_docs = docs[i:end]
            batch_meta = metadatas[i:end]
            batch_vecs = vectors[i:end]

            print(f"[vectorstore] Inserting batch {i // batch_size + 1} / {(total - 1) // batch_size + 1}")

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_vecs,
                metadatas=batch_meta,
            )

        print(f"[vectorstore] Successfully added all {total} vectors.")

    def query(self, query_embedding, top_k: int = 5):
        """
        Query the vector store.

        Args:
            query_embedding: Embedding vector.
            top_k: Number of results to fetch.

        Returns:
            List of dicts with content + metadata.
        """

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append({"content": doc, "metadata": meta})

        return output

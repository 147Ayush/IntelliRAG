"""
Main pipeline runner.

Usage:
  From project root (recommended):
    python -m src.pipeline

  Or (works too):
    python src/pipeline.py

The file adapts sys.path when run directly so imports work in both modes.
"""

import os
import sys

# When running this file directly (python src/pipeline.py), Python's import system
# may not treat 'src' as a package. To make imports robust, ensure project root is on sys.path.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Import modules (attempt package-style first, fallback to module-level)
try:
    from src.loaders import MultiFormatLoader
    from src.splitter import split_documents
    from src.embeddings import EmbeddingManager
    from src.vectorstore import VectorStore
    from src.retriever import RAGRetriever
    from src.llm import LocalLLM
except Exception:
    # fallback when running with sys.path patched (module names equal filenames)
    from loaders import MultiFormatLoader
    from splitter import split_documents
    from embeddings import EmbeddingManager
    from vectorstore import VectorStore
    from retriever import RAGRetriever
    from llm import LocalLLM


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if not os.path.isdir(data_dir):
        print(f"[pipeline] WARNING: data directory not found at {data_dir}. Create and add files (pdf/docx/txt/csv/xlsx).")
        os.makedirs(data_dir, exist_ok=True)

    # 1. Load documents
    loader = MultiFormatLoader(data_dir)
    documents = loader.load_all()

    if not documents:
        print("[pipeline] No documents found — exiting.")
        return

    # 2. Split into chunks
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)

    # 3. Initialize embeddings + vector store
    # Choose device "cuda" if you want GPU embeddings and have CUDA + torch installed
    try:
        embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2", device="cuda")
    except Exception:
        # fallback to CPU if GPU not available or model load fails
        embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2", device="cpu")

    vectorstore = VectorStore(persist_dir=os.path.join(PROJECT_ROOT, "data", "vector_store"), collection_name="documents")

    # 4. Add only chunks which come from files not yet processed
    new_chunks = [c for c in chunks if not vectorstore.file_exists(c.metadata.get("source_file", ""))]

    if new_chunks:
        texts = [c.page_content for c in new_chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vectorstore.add(new_chunks, embeddings)
    else:
        print("[pipeline] No new files to process.")

    # 5. Initialize retriever and LLM
    retriever = RAGRetriever(vectorstore, embedding_manager)

    try:
        llm = LocalLLM(model_name="microsoft/phi-3-mini-4k-instruct")
    except Exception as e:
        print(f"[pipeline] Warning: LLM load failed: {e}")
        print("[pipeline] Continuing without LLM. You can still run retrieval manually.")
        llm = None

    # 6. Example interactive loop (simple)
    query = "What is attention mechanism?"
    hits = retriever.retrieve(query, top_k=4)
    context = "\n\n".join([h["content"] for h in hits])
    print(f"\n[pipeline] Retrieved {len(hits)} hits for example query.\n")

    if llm:
        answer = llm.generate(query, context)
        print("\n--- Answer ---\n", answer)
    else:
        print("\n--- Context (no LLM) ---\n", context)


if __name__ == "__main__":
    main()

# ask_question.py
import os
import sys
from src.vectorstore import VectorStore
from src.embeddings import EmbeddingManager
from src.retriever import RAGRetriever    
from src.llm import LocalLLM

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR  # script is in project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # 1️⃣ Initialize Vector Store
    vectorstore_path = os.path.join(PROJECT_ROOT, "data", "vector_store")
    vectorstore = VectorStore(persist_dir=vectorstore_path)

    # 2️⃣ Initialize Embeddings
    try:
        embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2", device="cuda")
    except Exception:
        embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2", device="cpu")

    # 3️⃣ Initialize Retriever
    retriever = RAGRetriever(vectorstore, embedding_manager)

    # 4️⃣ Initialize LLM
    try:
        llm = LocalLLM(model_name="microsoft/phi-3-mini-4k-instruct")
    except Exception:
        print("LLM not available. Will return retrieved context only.")
        llm = None

    print("\n=== IntelliRAG Question-Answering ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # 5️⃣ Retrieve top-k relevant chunks
        hits = retriever.retrieve(query, top_k=5)
        if not hits:
            print("No relevant documents found.\n")
            continue

        # Combine retrieved chunks
        context = "\n\n".join([h["content"] for h in hits])

        # 6️⃣ Build improved prompt for LLM
        if llm:
            prompt = (
                f"You are a knowledgeable assistant. Based on the context below, "
                f"answer the question concisely in plain language. "
                f"If the context does not contain the answer, respond with 'Not found'.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\nAnswer:"
            )
            answer = llm.generate(query, prompt, max_length=512)
            print("\n--- Answer ---\n", answer)
        else:
            # If LLM unavailable, just show retrieved context
            print("\n--- Retrieved Context ---\n", context)

        print("\n---------------------------\n")

if __name__ == "__main__":
    main()

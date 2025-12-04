# check_dependencies.py
import importlib

def check_package(pkg_name, install_name=None):
    try:
        importlib.import_module(pkg_name)
        print(f"[OK] {pkg_name} is installed")
    except ImportError:
        install_name = install_name or pkg_name
        print(f"[MISSING] {pkg_name} is not installed. Install with: pip install {install_name}")

print("=== Checking Python dependencies for IntelliRAG ===\n")

# Core dependencies
packages = [
    ("torch", "torch"),
    ("sentence_transformers", "sentence-transformers"),
    ("chromadb", "chromadb"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("bitsandbytes", "bitsandbytes"),
    ("langchain_community", "langchain-community"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
]

for pkg, install_name in packages:
    check_package(pkg, install_name)

# Optional: test basic functionality
print("\n=== Testing basic functionality ===")

try:
    import torch
    print(f"PyTorch device available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"[ERROR] PyTorch test failed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"SentenceTransformer model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"[ERROR] SentenceTransformer test failed: {e}")

try:
    import chromadb
    client = chromadb.PersistentClient(path="./test_chroma")
    print("ChromaDB client initialized successfully")
except Exception as e:
    print(f"[ERROR] ChromaDB test failed: {e}")

print("\n✅ Dependency check complete.")

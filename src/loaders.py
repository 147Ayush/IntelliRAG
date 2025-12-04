"""
Multi-format document loader.

Supports: PDF, DOCX, TXT, CSV, XLSX (via langchain-community loaders).
Each loader returns a list of LangChain Document objects (one per page / chunk
as per that loader). We add standard metadata keys:
  - source_file: original filename
  - file_type: pdf/docx/txt/csv/xlsx
"""

from pathlib import Path
from typing import List, Any

# langchain-community loaders
# Make sure langchain-community and required loader packages are installed
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)


class MultiFormatLoader:
    def __init__(self, root_dir: str):
        """
        root_dir: directory to recursively scan for supported files
        """
        self.root_dir = Path(root_dir)
        self.supported_ext = [".pdf", ".docx", ".txt", ".csv", ".xlsx"]

    def load_all(self) -> List[Any]:
        """
        Load all supported documents from root_dir recursively.
        Returns a flat list of LangChain Document objects.
        """
        documents: List[Any] = []

        for ext in self.supported_ext:
            for file_path in self.root_dir.rglob(f"*{ext}"):
                try:
                    print(f"[loaders] Loading: {file_path.name} (ext={ext})")

                    if ext == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    elif ext == ".docx":
                        loader = Docx2txtLoader(str(file_path))
                    elif ext == ".txt":
                        loader = TextLoader(str(file_path))
                    elif ext == ".csv":
                        loader = CSVLoader(str(file_path))
                    elif ext == ".xlsx":
                        loader = UnstructuredExcelLoader(str(file_path))
                    else:
                        print(f"[loaders] Unsupported extension: {ext}")
                        continue

                    docs = loader.load()

                    # Attach metadata so downstream modules can check source_file
                    for d in docs:
                        # Some loaders might have existing metadata dict; ensure it exists
                        if not hasattr(d, "metadata") or d.metadata is None:
                            d.metadata = {}
                        d.metadata["source_file"] = file_path.name
                        d.metadata["file_type"] = ext.replace(".", "")

                    documents.extend(docs)

                except Exception as e:
                    print(f"[loaders] Error loading {file_path.name}: {e}")

        print(f"[loaders] Total loaded documents: {len(documents)}")
        return documents

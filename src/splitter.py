"""
Text splitting into chunks using LangChain text splitters.

Function:
  - split_documents(documents, chunk_size, chunk_overlap)
Returns:
  - list of LangChain Document objects (split chunks)
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Any


def split_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """
    Split a list of LangChain Document objects into chunks.

    Parameters:
      - documents: list of LangChain Document objects (each has .page_content and .metadata)
      - chunk_size: maximum characters per chunk
      - chunk_overlap: overlap in characters between adjacent chunks

    Returns:
      - list of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"[splitter] Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

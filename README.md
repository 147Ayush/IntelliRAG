# 📚 IntelliRAG – Local RAG Pipeline with LLM Support

---

## 🚀 Project Overview

IntelliRAG is a fully offline, modular, Retrieval-Augmented Generation (RAG) system built in Python. 

**empowers you to:**

- Load and process multi-format documents: PDF, DOCX, TXT, CSV, XLSX

- Split documents into chunks for embeddings

- Generate GPU-accelerated embeddings using SentenceTransformers

- Store and query vectors in a persistent ChromaDB

- Retrieve top-k relevant chunks for a query

- Generate context-aware answers using a local LLM (like Phi-3-mini)

It’s perfect for offline document QA, knowledge retrieval, internal company documentation search, or private RAG applications where data cannot leave your environment.

## 🗂 Project Structure
```bash
IntelliRAG/
├─ src/
│  ├─ loaders.py         # Multi-format document loader
│  ├─ splitter.py        # Split documents into chunks
│  ├─ embeddings.py      # GPU Embedding manager
│  ├─ vectorstore.py     # ChromaDB wrapper with batch insertion
│  ├─ retriever.py       # RAG retriever module
│  ├─ llm.py             # Local LLM wrapper
│  └─ pipeline.py        # Main pipeline runner
├─ ask_question.py       # Interactive Q&A script
├─ data/                 # Add PDF/DOCX/TXT/CSV/XLSX files here
│  └─ vector_store/      # Persistent vector database
├─ images/               # Screenshots/diagrams for documentation
├─ Dependency_check.py   # To check all requirements are install or not
├─ requirements.txt      # Python dependencies
└─ README.md             # Project documentation
```

---

## 🖼 Key Components
**1️⃣ Data Injection (Document Loading + Chunking)**

This stage loads all supported files, splits them into chunks, and stores them in the vector database.

**Features:**
- Recursive scanning of folder for supported files

- Automatic metadata generation: source_file, file_type, id, chunk_index

- Splits large documents into smaller chunks (default 1000 characters with 200 overlap)

- Supports thousands of files efficiently

**Metadata Example:**
```python
{
  "id": "uuid-1234",
  "source_file": "example.pdf",
  "file_type": "pdf",
  "chunk_index": 5
}
```
---
Image Placeholder: Data Injection Flow (add image in images/data_injection.png)

---

**2️⃣ Retriever Pipeline (Query + LLM)**

**After data injection:**

- Retriever fetches top-k chunks from ChromaDB using cosine similarity

- LLM generates concise, context-aware answers using the retrieved chunks

**Features:**

- GPU acceleration for query embedding

- Returns both content and metadata for traceability

- Configurable top_k results

- LLM generates context-aware, human-readable answers

**Example Query:**
```bash
"What is attention mechanism in NLP?"
```
**Example Retriever Output:**
```python
[
  {
    "content": "Attention mechanism allows models to focus on important parts of input...",
    "metadata": {"source_file": "ml_notes.pdf", "file_type": "pdf", "id": "uuid-5678"},
    "score": 0.92,
    "rank": 1
  }
]
```
---
Image Placeholder: Retriever + LLM Flow (add image in images/retriever_pipeline.png)

---

## ⚡ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd IntelliRAG
```
### 2. Create a Virtual Environment

**Linux/macOS**
```python
python -m venv rag_env && source rag_env/bin/activate
```
**Windows**
```python
python -m venv rag_env && rag_env\Scripts\activate
```
### 3. Install Dependencies
```python
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Verify GPU Availability
```python
python -c "import torch; print(torch.cuda.is_available())"
```
---
## 🏃 Running the Pipeline

### 1. Prepare Documents
Place all your documents inside the `data/` folder.

### 2. Run the Pipeline
```python
python -m src.pipeline
```
**Pipeline Workflow:**

- Loads and chunks documents

- Generates embeddings

- Stores chunks in ChromaDB

- Retrieves top-k relevant chunks

- Generates answers using LLM

---

## 🔹 Interactive Q&A

Use **ask_question.py** to interactively ask questions:

- Type your question to receive an answer from your document collection

- Type exit to quit the loop

**Example Interaction:**
```bash

Your question: What is shareholders equity?
Answer: Shareholders’ equity represents the owners’ claim on the company’s assets after liabilities are subtracted.

```
---

## ⚙ Configuration Options
```

| Parameter         | Description                        | Default                                |
|-------------------|------------------------------------|----------------------------------------|
| `chunk_size`      | Max characters per chunk           | `1000`                                 |
| `chunk_overlap`   | Overlap between chunks             | `200`                                  |
| `embedding_device`| `"cuda"` or `"cpu"`                | `"cuda"`                               |
| `LLM_model`       | HF model for generation            | `"microsoft/phi-3-mini-4k-instruct"`   |
| `vectorstore_path`| Persistent vector DB location      | `data/vector_store`                    |
| `top_k`           | Number of retrieved chunks/query   | `5`                                    |
```
---

# 🔍 How It Works – Workflow

- **Data Injection:** Load → split → generate embeddings → store in ChromaDB  
- **Retriever:** Convert query → embedding → search top-k relevant chunks  
- **LLM Generation:** Combine chunks → generate context-aware answer  
- **Benefits:** Accurate, scalable retrieval for thousands of documents without internet dependency  

---

## ⚡ Tips & Best Practices

- Use **GPU** for faster embedding and LLM inference  
- Avoid reprocessing files by checking `vectorstore.file_exists()`  
- Add `chunk_index` in metadata for traceable answers  
- Store all source files in `data/` for persistent knowledge

--- 

## 🛠 Troubleshooting
```

| Issue                                                    | Solution                                             |
|----------------------------------------------------------|------------------------------------------------------|
| `AttributeError: 'Document' object has no attribute 'id'`| Ensure `vectorstore.py` uses UUID for document ID    |
| Slow embeddings                                          | Switch to GPU (`cuda`) if available                  |
| LLM fails to load                                        | Install `transformers`, `accelerate`, `bitsandbytes` |
```

---

## 🌟 Advantages of IntelliRAG

- Fully offline & private  
- Modular and extensible architecture  
- Supports multi-format documents  
- Handles large datasets efficiently  
- Works with any HF-compatible LLM  

---

## 🔮 Future Improvements

- Deduplication of chunks  
- Multi-language support  
- Web or API interface for queries  
- Integration with larger or quantized LLMs  
- Real-time incremental updates  
- Automatic summarization for large chunks  
- Analytics dashboard for query and usage tracking  

---

## 🤝 Contributing

**1. Fork the repository**  
**2. Create a feature branch:** 
   ```bash
   git checkout -b feature/my-feature
   ```
**3. Commit your changes:**
   ```bash
   git commit -m "Add new feature"
   ```

**4. Push your branch:**
   ```bash
   git push origin feature/my-feature
   ```
**5. Open a pull request**

---
## 👤 Author

**Ayush Soni**  
- 💼 AI/ML Developer  
- 🧪 Building real-time AI-powered AR Virtual Try-On systems  
- 🎬 Creator of "What If YouTubers Had Alternate Lives?" AI series  
- 🧠 Passionate about RAG, LLMs, AR, and applied AI engineering   
- 📧 Email: ayushsaraf200@gmail.com

---

[MIT](https://choosealicense.com/licenses/mit/)

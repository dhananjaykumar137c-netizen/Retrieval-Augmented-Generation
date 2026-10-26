# 📄 Local PDF RAG Chatbot with Pinecone & Ollama

A Retrieval-Augmented Generation (RAG) system that enables interactive Q&A over PDF documents. This tool extracts text from a PDF, generates high-dimensional vector embeddings locally, stores them in a Pinecone serverless index, and uses a local **Llama 3.2** model via Ollama to provide accurate, context-aware answers.

---

## ✨ Features

* **Smart Deduplication:** Computes an MD5 hash of the source PDF and queries Pinecone metadata before processing, preventing redundant embedding generation and saving API quota.
* **Fast PDF Parsing:** Utilizes `PyMuPDFLoader` for rapid text extraction and `RecursiveCharacterTextSplitter` for context-preserving chunking (`chunk_size=500`, `chunk_overlap=100`).
* **Local Embeddings:** Generates 1024-dimensional dense vectors locally using Hugging Face's `thenlper/gte-large` model via `sentence-transformers`.
* **Batch Vector Upserts:** Uploads embeddings and document metadata to Pinecone in batches of 100 for reliable network performance.
* **Local LLM Inference:** Queries `llama3.2` locally through Ollama, keeping question-answering completely free of LLM API costs.

---

## 🛠️ Tech Stack

| Component | Technology | Details |
| :--- | :--- | :--- |
| **Document Loader & Splitter** | `langchain`, `langchain-community`, `PyMuPDF` | Extracts pages and splits text with overlap |
| **Embedding Model** | `sentence-transformers` (`thenlper/gte-large`) | **1024 dimensions** |
| **Vector Database** | `pinecone` | Stores vectors, text chunks, page numbers, and PDF hashes |
| **LLM Runtime** | `ollama` (`llama3.2`) | Local inference for context-grounded responses |

---

## 📋 Prerequisites

Before running the script, ensure you have the following set up:

1. **Python 3.9+** installed on your system.
2. **Pinecone Account & Index:**
   * Create a free account at [Pinecone](https://www.pinecone.io/).
   * Create an index (default name: `chatdatabase`) configured with **1024 dimensions** and **cosine** (or dotproduct/euclidean) metric to match the `thenlper/gte-large` embedding model.
3. **Ollama Installed & Running:**
   * Download and install [Ollama](https://ollama.com/).
   * Pull the `llama3.2` model:
     ```bash
     ollama pull llama3.2
     ```
   * Ensure the Ollama service is running in the background (`ollama serve`).

---

## 🚀 Installation & Setup

### 1. Clone the repository & navigate to the project folder
```bash
git clone <your-repo-url>
cd <your-repo-folder>

2. Create and activate a virtual environment

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

pip install langchain langchain-community pymupdf pinecone-client sentence-transformers ollama numpy tqdm python-dotenv

PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=chatdatabase
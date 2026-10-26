from langchain_community.document_loaders import PyMuPDFLoader
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np
from tqdm import tqdm
import hashlib
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
PDF_FILE = "os.pdf"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatdatabase")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Please check your .env file.")

# Initialize connections
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("thenlper/gte-large")

def get_pdf_hash(file_path):
    """Get MD5 hash of PDF file"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12]  # Use first 12 chars

def check_embeddings_exist(pdf_hash):
    """Check if embeddings for this PDF already exist"""
    try:
        # Try to find any vector with this pdf_hash
        dummy_vector = [0.0] * 1024  # gte-large dimension
        results = index.query(
            vector=dummy_vector,
            filter={"pdf_hash": pdf_hash},
            top_k=1
        )
        return len(results.matches) > 0
    except:
        return False

def create_chunks(file_path):
    """Load PDF and create chunks"""
    print(f"Loading PDF: {file_path}")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks

def create_and_store_embeddings(chunks, pdf_hash):
    """Create embeddings and store in Pinecone"""
    print("Creating embeddings...")
    
    vectors = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        # Create embedding
        embedding = embedder.encode(chunk.page_content)
        
        # Prepare vector for Pinecone
        vectors.append({
            "id": f"{pdf_hash}_{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk.page_content,
                "pdf_hash": pdf_hash,
                "chunk_id": i,
                "page": chunk.metadata.get('page', 0)
            }
        })
    
    # Upload to Pinecone in batches
    print("Uploading to Pinecone...")
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i+BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i//BATCH_SIZE + 1}/{(len(vectors)-1)//BATCH_SIZE + 1}")
    
    print(f"Stored {len(vectors)} embeddings")

def query_system(question, top_k=5):
    """Query the system and get answer"""
    print(f"\n Question: {question}")
    
    # Create embedding for question
    question_embedding = embedder.encode(question)
    
    # Search Pinecone
    results = index.query(
        vector=question_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    if not results.matches:
        return "No relevant information found."
    
    # Get context from top results
    context = "\n\n".join([match.metadata['text'] for match in results.matches])
    
    # Create prompt for Ollama
    prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate answer with Ollama
    print("Generating answer...")
    try:
        # Option 1: Ollama (local)
        response = ollama.generate(model='mistral', prompt=prompt)
        answer = response['response']
        
        print(f"Answer: {answer}")
        
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer. Check if Ollama is running."

# Main execution
def main():
    # Get PDF hash
    pdf_hash = get_pdf_hash(PDF_FILE)
    print(f"PDF Hash: {pdf_hash}")
    
    # Check if embeddings already exist
    if check_embeddings_exist(pdf_hash):
        print("Embeddings already exist, skipping creation")
    else:
        print("Creating new embeddings...")
        chunks = create_chunks(PDF_FILE)
        create_and_store_embeddings(chunks, pdf_hash)
    
    # Query loop
    print("\nSystem ready! Ask your questions (type 'quit' to exit)")
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        if question:
            # Use query_system(question, use_ai=False) to skip AI and just see relevant chunks
            query_system(question)

if __name__ == "__main__":
    main()
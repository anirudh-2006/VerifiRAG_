"""Phase 2: load the persisted Chroma store and expose a retriever."""

# src/retriever.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.load_env import load_project_dotenv

load_project_dotenv()

VECTORSTORE_DIR = "vectorstore"

def load_vectorstore():
    """Load the already-built ChromaDB from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def retrieve(query: str, k: int = 4):
    """
    Find the top-k most relevant chunks for a query.
    k=4 means we return the 4 best-matching chunks.
    """
    vectorstore = load_vectorstore()
    
    # similarity_search finds chunks whose vectors are closest to the query vector
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Chunk {i+1} (similarity score: {score:.3f}) ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}")
        print(doc.page_content[:300])  # preview first 300 chars
    
    return [doc for doc, score in results]

# Test it
if __name__ == "__main__":
    query = "What is the Fourier transform of a rectangular pulse?"
    docs = retrieve(query)
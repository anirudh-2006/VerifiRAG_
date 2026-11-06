"""Phase 1: load PDFs from data/pdfs and persist chunks to Chroma."""

# src/ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

PDF_DIR = "data/pdfs"
VECTORSTORE_DIR = "vectorstore"

def load_pdfs(pdf_dir):
    """Load all PDFs from a folder and return a list of pages."""
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            pages = loader.load()  # each page becomes one Document object
            all_docs.extend(pages)
    print(f"Loaded {len(all_docs)} pages total.")
    return all_docs

def split_documents(docs):
    """
    Split pages into smaller chunks.
    chunk_size=800: each chunk is ~800 characters
    chunk_overlap=100: chunks share 100 chars at edges so context isn't lost at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]  # tries to split at paragraph, then line, then sentence
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def embed_and_store(chunks):
    """Convert chunks to vectors and save to ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # cheap and good
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR  # saves to disk so you don't re-embed every time
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at '{VECTORSTORE_DIR}'")
    return vectorstore

if __name__ == "__main__":
    docs = load_pdfs(PDF_DIR)
    chunks = split_documents(docs)
    embed_and_store(chunks)
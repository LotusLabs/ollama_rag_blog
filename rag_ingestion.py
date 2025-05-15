import os
import sys
import shutil
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama as LlamaIndexOllama


# --- Configuration ---
PERSIST_DIR     = "./chroma_db"
COLLECTION_NAME = "survival_docs"
DATA_DIR        = "./survival_docs" 

LLM_MODEL_NAME       = 'qwen3:8b-q4_K_M'
EMBEDDING_MODEL_NAME = 'nomic-embed-text:137m-v1.5-fp16'

def setup_llm_and_embed_models():
    """Sets up the LLM and embedding models in LlamaIndex Settings."""
    print(f"Setting LLM model: {LLM_MODEL_NAME}")
    Settings.llm = LlamaIndexOllama(model=LLM_MODEL_NAME, request_timeout=120.0)
    print(f"Setting embedding model: {EMBEDDING_MODEL_NAME}")
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print("LLM and embedding models set.")

def initialize_vector_store():
    """Initializes or wipes and re-initializes the ChromaDB vector store."""
    if os.path.exists(PERSIST_DIR):
        print(f"Wiping previous database at {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
    
    os.makedirs(PERSIST_DIR, exist_ok=True) # Ensure dir exists after wipe

    print(f"Initializing ChromaDB client at {PERSIST_DIR}...")
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    print(f"Getting or creating Chroma collection: {COLLECTION_NAME}")
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print("ChromaDB vector store initialized.")
    return vector_store

def load_documents():
    """Loads documents from the DATA_DIR."""
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' is missing or empty.")
        print("Please create it and add your documents.")
        sys.exit(1)

    print(f"Loading documents from {DATA_DIR}...")
    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        if not documents:
            print(f"No documents found in {DATA_DIR}. Please add some text files.")
            sys.exit(1)
        print(f"Loaded {len(documents)} document(s).")
        return documents
    except ValueError as e:
        print(f"Error loading documents: {e}")
        print(f"Please ensure '{DATA_DIR}' exists and contains readable files.")
        sys.exit(1)

def ingest_documents(documents, vector_store):
    """Creates an index and ingests documents into the vector store."""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Creating index and ingesting documents into ChromaDB...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("Index created and documents ingested successfully.")
    return index

def main():
    """Main function to run the ingestion pipeline."""
    print("--- Starting RAG Ingestion Process ---")
    
    setup_llm_and_embed_models()
    vector_store = initialize_vector_store()
    documents = load_documents()
    ingest_documents(documents, vector_store)
    
    print("--- RAG Ingestion Process Complete ---")
    print(f"Data ingested into ChromaDB at {PERSIST_DIR} using collection '{COLLECTION_NAME}'.")
    print("You can now run query_rag.py to chat with your documents.")

if __name__ == "__main__":
    main()
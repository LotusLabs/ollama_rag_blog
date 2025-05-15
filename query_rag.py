import time
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
import os, sys

# --- Configuration ---
PERSIST_DIR = "./chroma_db"
LLM_MODEL_NAME = 'qwen3:8b-q4_K_M'
EMBEDDING_MODEL_NAME = 'nomic-embed-text:137m-v1.5-fp16'
COLLECTION_NAME = "survival_docs"



SYSTEM_PROMPT_RAG = """
You are an expert offline survival assistant. Use the provided context information
to answer the user's question accurately and concisely based on the context.
Prioritize the information provided in the context above all else.

# Context:
{context_str}

# User Question: {query_str}

# Answer:
"""


def setup_llm_and_embed_models():
    """Sets up the LLM and embedding models in LlamaIndex Settings."""
    print(f"Setting LLM model: {LLM_MODEL_NAME}")
    Settings.llm = LlamaIndexOllama(model=LLM_MODEL_NAME, request_timeout=120.0)
    print(f"Setting embedding model: {EMBEDDING_MODEL_NAME}")
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print("LLM and embedding models set.")

def load_vector_store_and_index():
    """Loads the existing ChromaDB vector store and creates an index from it."""
    if not os.path.exists(PERSIST_DIR):
        print(f"ERROR: ChromaDB persistence directory '{PERSIST_DIR}' not found.")
        print("Please run the rag_ingestion.py script first to create and populate the database.")
        sys.exit(1)

    print(f"Loading ChromaDB client from {PERSIST_DIR}...")
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    print(f"Getting Chroma collection: {COLLECTION_NAME}")
    try:
        chroma_collection = db.get_collection(COLLECTION_NAME)
    except Exception as e: 
        print(f"Error getting collection '{COLLECTION_NAME}': {e}")
        print(f"Ensure the collection was created by rag_ingestion.py.")
        sys.exit(1)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    print("Loading index from vector store...")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    print("Index loaded successfully from ChromaDB.")
    return index

def create_query_engine(index):
    """Creates a query engine from the loaded index."""
    qa_template = PromptTemplate(SYSTEM_PROMPT_RAG)

    print("Creating query engine (with streaming enabled)...")
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=qa_template,
        streaming=True  # Enable streaming
    )
    print("Query engine created.")
    return query_engine

def run_chat_loop(query_engine):
    """Runs the interactive chat loop."""
    print("\nStarting RAG chat. Type 'quit' or 'exit' to end.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if user_input.strip():
                print(f"\nQuerying with: '{user_input}'...")
                t0 = time.perf_counter()

                response_obj = query_engine.query(user_input)

                print("\n--- Assistant Response ---")
                if hasattr(response_obj, 'response_gen'):
                    for token in response_obj.response_gen:
                        print(token, end="", flush=True)
                    sys.stdout.flush() 
                    print() 
                else:
                    print("Streaming response generator not found. Displaying full response.")
                    if hasattr(response_obj, 'response') and response_obj.response:
                        print(response_obj.response.strip())
                    else:
                        print("No response text available.")
                
                t1 = time.perf_counter()

                print("-------------------------")
                print(f"Response time: {t1 - t0:.2f} seconds")

                print("\n--- Retrieved Sources ---")
                if response_obj.source_nodes:
                    for i, node in enumerate(response_obj.source_nodes):
                        print(f"  Source {i+1}:")
                        print(f"    ID: {node.node_id}")
                        print(f"    Score: {node.score:.4f}")
                        file_name = node.metadata.get('file_name', 'N/A') if node.metadata else 'N/A'
                        print(f"    File: {file_name}")
                else:
                    print("  - No sources retrieved.")
                print("-------------------------\n")
            else:
                print("Please enter a prompt.")
        except EOFError:
            print("\nExiting due to EOF.")
            break
        except KeyboardInterrupt:
            print("\nExiting due to KeyboardInterrupt.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
    print("\nExiting chat.")


def main():
    """Main function to run the RAG query interface."""
    print("--- Starting RAG Query Interface ---")
    
    setup_llm_and_embed_models()
    index = load_vector_store_and_index()
    query_engine = create_query_engine(index)
    run_chat_loop(query_engine)
    
    print("--- RAG Query Interface Closed ---")

if __name__ == "__main__":
    main()
import os
import chromadb
import ollama

# --- Configuration ---
CHROMA_PATH = "code_db_mistral"
COLLECTION_NAME = "mistral_codebase"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
CHUNK_SIZE = 1500  # Characters per chunk (adjust as needed)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def get_chunks(text, size):
    """Splits a string into smaller chunks."""
    return [text[i:i + size] for i in range(0, len(text), size)]

def index_code(directory):
    ignored_dirs = {'.git', 'venv', '__pycache__', 'code_db_mistral'}
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                print(f"Processing: {path}")
                
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Split large files into manageable chunks
                    chunks = get_chunks(content, CHUNK_SIZE)
                    
                    for i, chunk in enumerate(chunks):
                        # Generate unique ID for each chunk
                        chunk_id = f"{path}_chunk_{i}"
                        
                        # Get embedding from Ollama
                        response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                        
                        collection.add(
                            ids=[chunk_id],
                            embeddings=[response["embedding"]],
                            documents=[chunk],
                            metadatas=[{"source": path, "chunk_index": i}]
                        )
                except Exception as e:
                    print(f"Failed to index {path}: {e}")

def query_code(user_query):
    query_embed = ollama.embeddings(model=EMBED_MODEL, prompt=user_query)["embedding"]
    
    # Query for the top 3 most relevant chunks
    results = collection.query(query_embeddings=[query_embed], n_results=3)
    
    # Combine the found snippets into one context block
    context = "\n---\n".join(results["documents"][0])
    
    prompt = f"[INST] Use the code snippets below to answer: {user_query}\n\nContext:\n{context} [/INST]"
    
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"]

if __name__ == "__main__":
    # Index current directory
    index_code("./")
    
    # Test a query
    print("\nResult:", query_code("How is the main entry point structured?"))
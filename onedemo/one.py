import os
import sys
import chromadb
import ollama

# --- Configuration ---
# Uses environment variables from docker-compose or defaults to local 
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb") 
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "multi_lang_codebase"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
CHUNK_SIZE = 1500  

# Initialize the Remote Client 
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def get_chunks(text, size):
    """Splits a string into smaller chunks."""
    return [text[i:i + size] for i in range(0, len(text), size)]

def index_code(path_to_index):
    """
    Walks through the directory or processes a single file, 
    embedding and storing supported code types.
    """
    ignored_dirs = {'.git', 'venv', '__pycache__', 'node_modules', '.terraform'}
    supported_ext = {".py", ".java", ".js", ".tf", ".yml", ".yaml"}
    
    # Handle single file indexing if path_to_index is a file
    if os.path.isfile(path_to_index):
        files_to_process = [(os.path.dirname(path_to_index), [], [os.path.basename(path_to_index)])]
    else:
        files_to_process = os.walk(path_to_index)

    for root, dirs, files in files_to_process:
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            
            # Filter for supported types and specific GitLab CI files 
            is_gitlab = "gitlab-ci" in file.lower() and ext in {".yml", ".yaml"}
            if ext in supported_ext:
                if ext in {".yml", ".yaml"} and not is_gitlab:
                    continue
                
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    chunks = get_chunks(content, CHUNK_SIZE)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{file_path}_chunk_{i}"
                        
                        # Get embedding from Ollama 
                        response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                        
                        # Add to the remote ChromaDB collection 
                        collection.add(
                            ids=[chunk_id],
                            embeddings=[response["embedding"]],
                            documents=[chunk],
                            metadatas=[{"source": file_path, "type": ext}]
                        )
                except Exception as e:
                    print(f"Failed to index {file_path}: {e}")

def query_code(user_query):
    """Queries the vector database and generates a response via LLM."""
    query_embed = ollama.embeddings(model=EMBED_MODEL, prompt=user_query)["embedding"]
    
    # Query for the top 3 most relevant chunks 
    results = collection.query(query_embeddings=[query_embed], n_results=3)
    
    if not results["documents"][0]:
        return "No relevant context found."

    context = "\n---\n".join(results["documents"][0])
    prompt = f"[INST] Use the code snippets below to answer: {user_query}\n\nContext:\n{context} [/INST]"
    
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"]

if __name__ == "__main__":
    # Check if a specific path was passed as a CLI argument
    # Usage: python one.py ./path/to/folder_or_file
    target = sys.argv[1] if len(sys.argv) > 1 else "./"
    
    print(f"--- Starting indexing for: {target} ---")
    index_code(target)
    
    # Only run test query if no specific path was provided (standard run)
    if len(sys.argv) == 1:
        print("\nTest Query: How is the system structured?")
        print("Result:", query_code("How is the system structured?"))
    # Example queries to test the new files
        print("\nResult (general):", query_code("List all python files"))
        print("\nResult (Terraform):", query_code("What S3 bucket is defined?"))
        print("\nResult (GitLab):", query_code("What happens in the test stage?"))
        print("\nResult (Java):", query_code("show Java Calls"))
        print("\nResult (Javascript):", query_code("show Javascript Calls"))
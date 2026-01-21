import os
import chromadb
import ollama

# --- Configuration ---
CHROMA_PATH = "code_db"
COLLECTION_NAME = "python_codebase"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def index_code(directory):
    """Recursively reads python files and adds them to ChromaDB."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                    
                    # Create a unique ID for the file
                    file_id = path.replace("/", "_")
                    
                    # Generate embedding using Ollama
                    response = ollama.embeddings(model=EMBED_MODEL, prompt=code_content)
                    embedding = response["embedding"]
                    
                    # Store in ChromaDB
                    collection.add(
                        ids=[file_id],
                        embeddings=[embedding],
                        documents=[code_content],
                        metadatas=[{"source": path}]
                    )
    print(f"Successfully indexed files from {directory}")

def query_code(user_query):
    """Finds relevant code and generates a response via Ollama."""
    # 1. Embed the user query
    query_embed = ollama.embeddings(model=EMBED_MODEL, prompt=user_query)["embedding"]
    
    # 2. Retrieve the top 2 most relevant snippets from Chroma
    results = collection.query(query_embeddings=[query_embed], n_results=2)
    context = "\n\n".join(results["documents"][0])
    
    # 3. Use Ollama to generate an answer based on the code context
    prompt = f"Using the following code context, answer the question.\n\nContext:\n{context}\n\nQuestion: {user_query}"
    
    output = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return output["response"]

# --- Execution ---
if __name__ == "__main__":
    # 1. Index your local code (replace 'path/to/your/code' with your actual directory)
    # index_code("./my_project") 

    # 2. Query the system
    question = "How does the database connection work in this project?"
    answer = query_code(question)
    
    print(f"\n--- AI RESPONSE ---\n{answer}")
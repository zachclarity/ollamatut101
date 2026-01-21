This project demonstrates a local **Retrieval-Augmented Generation (RAG)** system that allows you to chat with your own Python codebase. It uses **Ollama** for language modeling and embeddings, **ChromaDB** as a vector database for storage, and **Docker Compose** to orchestrate these services.

---

## 🚀 Overview: How it Works

The application follows a standard RAG pipeline to provide context-aware answers about your code:

1. **Ingestion:** The system walks through your local directory, reads `.py` files, and splits them into smaller "chunks".
2. **Embedding:** Each chunk is sent to Ollama’s `nomic-embed-text` model to be converted into a numerical vector (embedding).
3. **Storage:** These vectors are stored in **ChromaDB** along with the original text and metadata (file path, chunk index).
4. **Retrieval:** When you ask a question, the system embeds your query and searches ChromaDB for the most relevant code snippets.
5. **Generation:** The relevant snippets are injected into a prompt as "context," and the **Mistral** model generates a response based on that specific code.

---

## 🛠️ Key Concepts & Tools

* **Ollama:** A local runtime for serving Large Language Models (LLMs). In this setup, it handles both the **Mistral** LLM for answering questions and the **nomic-embed-text** model for creating vector representations of your code.
* **ChromaDB:** A high-performance vector database. It is responsible for "remembering" your code by storing embeddings and allowing fast semantic searches to find relevant sections.
* **Docker Compose:** Orchestrates three distinct services (**ollama**, **chromadb**, and **backend**) so they can communicate seamlessly over a private network using service names as hostnames.
* **Vector Embeddings:** Numerical representations of text that capture semantic meaning. This allows the system to find code that is *logically* related to your query, even if the exact words don't match.

---

## 🗄️ How ChromaDB is Used

In this demo, ChromaDB operates in **Client/Server mode**:

* **Remote Server:** A standalone ChromaDB instance runs in a dedicated container. This ensures data persistence in the `chroma_data` volume even if the application container is restarted.
* **HttpClient:** The Python script uses `chromadb.HttpClient` to connect to the server via its network alias (`http://chromadb:8000`) rather than running an in-memory database.
* **Collections:** Data is organized into a collection named `mistral_codebase`. This acts like a "table" where your code chunks and their corresponding vectors are indexed for retrieval.

---

## 📁 File Structure & Responsibilities

| File | Purpose |
| --- | --- |
| **`docker-compose.yml`** | Defines the environment. It pulls the Ollama/Chroma images and automatically downloads the necessary models (`qwen2.5-coder`) on startup. |
| **`Dockerfile`** | Builds the Python environment for the `backend` app. It installs dependencies and prepares the script to run. |
| **`one.py`** | The "brain" of the app. It handles the logic for scanning files, embedding text via Ollama, and querying ChromaDB. |
| **`requirements.txt`** | Lists Python libraries like `chromadb` and `ollama` needed to communicate with the services. |

---

### Getting Started

To run this demo, simply navigate to the project root and run:

```bash
docker compose up --build

```

This will start the database, the AI engine, and the indexing script automatically.

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import BSHTMLLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Global variables to hold our chain
app_state = {}

from pathlib import Path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and Index inside the lifespan to avoid SpawnProcess errors
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embeddings = OllamaEmbeddings(model="qwen2.5-coder:0.5b", base_url=OLLAMA_URL)
    llm = ChatOllama(model="qwen2.5-coder:0.5b", base_url=OLLAMA_URL)

    loader = BSHTMLLoader("templates/reference.html")
    docs = loader.load()
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    prompt = ChatPromptTemplate.from_template("""
    Use the following context to generate Python code:
    {context}
    User Request: {input}
    Answer in valid Python code only.
    """)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # Store in app_state for access in routes
    app_state["chain"] = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
    
    yield
    # Clean up (if needed) on shutdown
    app_state.clear()

app = FastAPI(lifespan=lifespan)
# Get the absolute path to the current directory
BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/generate")
async def generate(user_input: str):
    chain = app_state.get("chain")
    if not chain:
        return {"error": "Chain not initialized"}
    
    response = chain.invoke({"input": user_input})
    return {"code": response["answer"]}
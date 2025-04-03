# rag_core.py
import os
import glob
import asyncio
from typing import List, Optional, Dict, Any

# Document Loading, Splitting, Embeddings, Vector Store (keep imports as before)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Replace LLM Placeholder with LlamaCpp ---
from langchain_community.llms import LlamaCpp # Import LlamaCpp

# Prompt Engineering & Chains (keep imports as before)
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Configuration ---
KNOWLEDGE_DIR = "knowledge_base"
VECTOR_STORE_PATH = "vector_store_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- NEW: Path to your downloaded GGUF model file ---
# Adjust this path to where you saved the downloaded model
MODEL_PATH = "./models/phi-2.Q4_K_M.gguf" # <--- CHANGE THIS AS NEEDED

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print("="*50)
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please download a GGUF model (e.g., from https://huggingface.co/TheBloke/phi-2-GGUF)")
    print("and update the MODEL_PATH variable in rag_core.py")
    print("="*50)
    # Optionally exit or raise an error here if the model is critical for startup
    # exit(1)

# --- Global State / Shared Resources ---
indexing_lock = asyncio.Lock()

# --- Core RAG Functions (load_documents_from_directory, create_vector_store, update_knowledge_base remain the same) ---

def load_documents_from_directory(dir_path: str = KNOWLEDGE_DIR) -> List[Document]:
    # ... (keep implementation as before) ...
    documents = []
    print(f"Loading documents from: {dir_path}")
    if not os.path.exists(dir_path):
        print(f"Warning: Knowledge directory '{dir_path}' not found.")
        return []

    for ext in ["*.pdf", "*.txt"]:
        files = glob.glob(os.path.join(dir_path, ext))
        for file_path in files:
            try:
                if ext == "*.pdf":
                    print(f"  Loading PDF: {os.path.basename(file_path)}")
                    loader = PyPDFLoader(file_path)
                elif ext == "*.txt":
                    print(f"  Loading TXT: {os.path.basename(file_path)}")
                    loader = TextLoader(file_path, encoding='utf-8')
                else: continue
                documents.extend(loader.load())
            except Exception as e:
                print(f"  Error loading {os.path.basename(file_path)}: {e}")
    print(f"Loaded {len(documents)} document sections initially.")
    return documents

def create_vector_store(docs: List[Document],
                        embedding_function: HuggingFaceEmbeddings,
                        store_path: str = VECTOR_STORE_PATH) -> Optional[FAISS]:
    # ... (keep implementation as before) ...
    if not docs:
        print("No documents provided to create vector store.")
        if os.path.exists(store_path): import shutil; print(f"Removing existing vector store at {store_path} due to empty document list."); shutil.rmtree(store_path)
        return None
    print("Creating new vector store...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
    split_docs = text_splitter.split_documents(docs)
    print(f"Split documents into {len(split_docs)} chunks.")
    if not split_docs:
        print("No text chunks generated after splitting. Cannot create vector store.")
        if os.path.exists(store_path): import shutil; print(f"Removing existing vector store at {store_path} due to lack of splittable content."); shutil.rmtree(store_path)
        return None
    try:
        vector_store = FAISS.from_documents(split_docs, embedding_function)
        vector_store.save_local(store_path)
        print(f"New vector store created and saved to: {store_path}")
        return vector_store
    except Exception as e:
        print(f"Error creating FAISS store: {e}")
        return None


async def update_knowledge_base():
    # ... (keep implementation as before, it calls create_vector_store) ...
    if indexing_lock.locked(): print("Indexing already in progress. Skipping update request."); return False
    async with indexing_lock:
        print("Starting knowledge base update...")
        try:
            docs = load_documents_from_directory(KNOWLEDGE_DIR)
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            create_vector_store(docs, embeddings, VECTOR_STORE_PATH)
            print("Knowledge base update completed.")
            return True
        except Exception as e:
            print(f"Error during knowledge base update: {e}")
            return False

# --- UPDATED: get_rag_chain uses the LLM passed to it ---
def get_rag_chain(embedding_function: HuggingFaceEmbeddings, llm: LlamaCpp) -> Optional[Any]: # Type hint updated
    """
    Loads the vector store and creates the RAG retrieval chain using the provided LLM.
    Returns None if the vector store doesn't exist.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found. Cannot create RAG chain.")
        return None
    if llm is None:
        print("LLM instance not provided. Cannot create RAG chain.")
        return None

    try:
        print("Loading vector store for RAG chain...")
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embedding_function,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        print("Vector store loaded, retriever created.")

        # Define the Prompt Template (same as before)
        prompt_template = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks based ONLY on the provided context.
            Use the following pieces of retrieved context to answer the question precisely and concisely.
            If the context does not contain the answer, state "Based on the provided documents, I cannot answer this question."
            Do not add any information not present in the context. Avoid introductory phrases like "Based on the context...".

            Context:
            {context}

            Question:
            {input}

            Answer:
            """
        )

        # Create the RAG Chain (same as before)
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        print("RAG chain created successfully.")
        return retrieval_chain

    except Exception as e:
        print(f"Error loading vector store or creating RAG chain: {e}")
        return None


# --- NEW: Function to initialize LlamaCpp ---
def initialize_llm(model_path: str = MODEL_PATH) -> Optional[LlamaCpp]:
    """Initializes the LlamaCpp model."""
    if not os.path.exists(model_path):
        print(f"LLM model file not found at {model_path}. Cannot initialize LLM.")
        return None

    try:
        print(f"Initializing LlamaCpp model from: {model_path}")
        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=2048,  # Context window size - adjust based on model capabilities & RAM
            n_gpu_layers=0, # Set to 0 for CPU only. Increase if using GPU offloading (e.g., 10, 20, etc.)
            n_batch=512,   # Batch size for prompt processing
            temperature=0.1, # Low temperature for more deterministic, factual answers
            max_tokens=512, # Maximum number of tokens to generate
            # top_p=0.95,
            verbose=False,  # Set to True for detailed LlamaCpp logging
            # f16_kv=True # Can improve speed on some systems if supported
        )
        print("LlamaCpp model initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing LlamaCpp model: {e}")
        return None
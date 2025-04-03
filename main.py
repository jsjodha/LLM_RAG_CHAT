# main.py
import glob
import os
import shutil
from typing import List, Dict, Optional, Any # Added Optional, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

# Import necessary components from rag_core
from rag_core import (
    KNOWLEDGE_DIR,
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL_NAME,
    # TeapotLLMPlaceholder, # REMOVED Placeholder
    indexing_lock,
    update_knowledge_base,
    get_rag_chain,
    initialize_llm # Added LLM initializer
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

import rag_core # Import LlamaCpp for type hint

# --- Pydantic Models (remain the same) ---
class AskRequest(BaseModel): question: str
class AskResponse(BaseModel): answer: str; status: str; message: Optional[str] = None
class UploadResponse(BaseModel): message: str; filenames: List[str]; background_task_status: str

# --- FastAPI App Initialization ---
app = FastAPI(title="RAG API with Local LLM (LlamaCpp)")

# --- Global Resources / Dependencies ---
# Use a simple global variable or cache for the LLM instance to avoid reloading it per request
llm_instance: Optional[LlamaCpp] = None

# Function to get or initialize the LLM instance
def get_llm_instance() -> Optional[LlamaCpp]:
    global llm_instance
    if llm_instance is None:
        print("LLM instance not found, attempting initialization...")
        llm_instance = initialize_llm() # Function from rag_core.py
    # else: print("Returning existing LLM instance.") # Optional logging
    return llm_instance

# Function to get embedding model (same as before)
def get_embedding_model():
    print("Loading embedding model for dependency injection...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Ensure knowledge directory exists on startup
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# --- API Endpoints ---

# /upload endpoint remains the same as before
@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    # ... (keep implementation as before) ...
    print(f"Received {len(files)} files for upload.")
    saved_filenames = []
    allowed_extensions = {".pdf", ".txt"}
    for file in files:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions: print(f"Skipping unsupported file type: {file.filename}"); continue
        file_path = os.path.join(KNOWLEDGE_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
            saved_filenames.append(file.filename)
            print(f"Successfully saved: {file.filename}")
        except Exception as e: print(f"Error saving file {file.filename}: {e}"); raise HTTPException(status_code=500, detail=f"Could not save file: {file.filename}. Error: {e}")
        finally: file.file.close()
    if not saved_filenames: raise HTTPException(status_code=400, detail="No valid files were uploaded or saved.")
    task_status = "triggered"
    if indexing_lock.locked(): task_status = "skipped (update already in progress)"; print("Indexing lock is held, skipping new background task trigger.")
    else: print("Adding knowledge base update to background tasks."); background_tasks.add_task(update_knowledge_base)
    return UploadResponse(message=f"Successfully saved {len(saved_filenames)} files.", filenames=saved_filenames, background_task_status=task_status)


# /ask endpoint updated to use the initialized LLM
@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    embeddings: HuggingFaceEmbeddings = Depends(get_embedding_model),
    # Use the dependency function to get the LLM instance
    llm: Optional[LlamaCpp] = Depends(get_llm_instance) # Inject LLM
):
    print(f"Received question: {request.question}")

    if llm is None:
         print("LLM is not available. Check model path and initialization logs.")
         raise HTTPException(status_code=503, detail="LLM is not initialized or failed to load. Cannot process request.")

    if indexing_lock.locked():
        print("Indexing is in progress. Please wait.")
        return AskResponse(answer="", status="indexing_in_progress", message="The knowledge base is currently being updated. Please try again shortly.")

    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found.")
        return AskResponse(answer="", status="error", message="Vector store not found. Please upload documents and wait for indexing.")

    # Get the RAG chain (loads vector store inside), passing the LLM instance
    retrieval_chain = get_rag_chain(embedding_function=embeddings, llm=llm)

    if retrieval_chain is None:
        return AskResponse(answer="", status="error", message="Failed to load vector store or create RAG chain. Check server logs.")

    try:
        print("Invoking RAG chain with local LLM...")
        # LlamaCpp usually runs synchronously in LangChain unless specifically configured otherwise
        # So we might not need await here, but check Langchain documentation if using async variants
        response = retrieval_chain.invoke({"input": request.question})

        final_answer = response.get("answer", "Error: Could not extract answer from LLM response.")
        print(f"Generated answer (local LLM): {final_answer}")

        return AskResponse(answer=final_answer, status="success")
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        # Add more specific error logging if possible
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the question with the local LLM: {e}")


# /status endpoint remains the same
@app.get("/status")
async def get_status() -> Dict[str, Any]:
    # ... (keep implementation as before) ...
    index_exists = os.path.exists(VECTOR_STORE_PATH)
    is_indexing = indexing_lock.locked()
    num_docs = len(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.*")))
    llm_status = "Initialized" if llm_instance is not None else "Not Initialized (check model path/logs)"
    return {
        "status": "online",
        "vector_store_exists": index_exists,
        "indexing_in_progress": is_indexing,
        "knowledge_base_docs": num_docs,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_status": llm_status,
        "llm_model_path": rag_core.MODEL_PATH # Show configured model path
    }

# --- Uvicorn Runner (commented out, use command line) ---
if __name__ == "__main__":
    import uvicorn
    # Initialize LLM on startup when running directly (optional)
    get_llm_instance()
    uvicorn.run(app, host="0.0.0.0", port=8000)
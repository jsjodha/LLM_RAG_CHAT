# RAG API User Guide

This guide explains how to interact with the RAG (Retrieval-Augmented Generation) API built with FastAPI. The API allows you to upload documents to build a knowledge base and then ask questions that are answered based on the content of those documents using a local Large Language Model (LLM).

## Prerequisites

*   The API server must be running (e.g., using `uvicorn main:app --host 0.0.0.0 --port 8000` or the provided `run.sh` script).
*   You need a tool like `curl` or a client application (like Postman or Insomnia) to send HTTP requests.

## API Endpoints

### 1. Upload Documents

*   **Method:** `POST`
*   **Path:** `/upload`
*   **Description:** Uploads one or more documents (`.pdf`, `.txt`) to the knowledge base directory. After successful upload, it triggers a background task to index the documents into a vector store. If an indexing task is already running, the new task is skipped.
*   **Request:**
    *   Content-Type: `multipart/form-data`
    *   Body: Contains one or more files attached with the key `files`.
*   **Response:**
    *   **Success (200 OK):**
        ```json
        {
          "message": "Successfully saved X files.",
          "filenames": ["doc1.pdf", "doc2.txt"],
          "background_task_status": "triggered" // or "skipped (update already in progress)"
        }
        ```
    *   **Error (400 Bad Request):** If no valid files are provided.
    *   **Error (500 Internal Server Error):** If there's an issue saving files.
*   **Example (`curl`):**
    ```bash
    curl -X POST "http://localhost:8000/upload" \
         -F "files=@/path/to/your/document1.pdf" \
         -F "files=@/path/to/your/document2.txt"
    ```

### 2. Ask a Question

*   **Method:** `POST`
*   **Path:** `/ask`
*   **Description:** Sends a question to the API. The system retrieves relevant information from the indexed knowledge base and uses the local LLM to generate an answer.
*   **Request:**
    *   Content-Type: `application/json`
    *   Body:
        ```json
        {
          "question": "Your question here?"
        }
        ```
*   **Response:**
    *   **Success (200 OK):**
        ```json
        {
          "answer": "The generated answer based on the documents.",
          "status": "success",
          "message": null
        }
        ```
    *   **Indexing In Progress (200 OK):** If the knowledge base is currently being updated.
        ```json
        {
          "answer": "",
          "status": "indexing_in_progress",
          "message": "The knowledge base is currently being updated. Please try again shortly."
        }
        ```
     *   **Vector Store Not Found (200 OK):** If no vector store exists (upload documents first).
        ```json
        {
          "answer": "",
          "status": "error",
          "message": "Vector store not found. Please upload documents and wait for indexing."
        }
        ```
    *   **Error (503 Service Unavailable):** If the LLM failed to initialize.
    *   **Error (500 Internal Server Error):** If an error occurs during the RAG process.
*   **Example (`curl`):**
    ```bash
    curl -X POST "http://localhost:8000/ask" \
         -H "Content-Type: application/json" \
         -d '{"question": "What is the main topic of the uploaded documents?"}'
    ```

### 3. Get API Status

*   **Method:** `GET`
*   **Path:** `/status`
*   **Description:** Retrieves the current status of the API, including whether the vector store exists, if indexing is in progress, the number of documents in the knowledge base, and the status of the LLM.
*   **Request:** None
*   **Response:**
    *   **Success (200 OK):**
        ```json
        {
          "status": "online",
          "vector_store_exists": true, // or false
          "indexing_in_progress": false, // or true
          "knowledge_base_docs": 5, // Number of files in knowledge_base/
          "embedding_model": "BAAI/bge-small-en-v1.5", // Example model name
          "llm_status": "Initialized", // or "Not Initialized (...)"
          "llm_model_path": "models/phi-2.Q4_K_M.gguf" // Example model path
        }
        ```
*   **Example (`curl`):**
    ```bash
    curl -X GET "http://localhost:8000/status"
    ```

## Workflow

1.  **Upload Documents:** Use the `/upload` endpoint to add your PDF and TXT files to the knowledge base. Wait for the `background_task_status` to indicate `triggered`.
2.  **Check Status (Optional):** Use the `/status` endpoint to monitor `indexing_in_progress`. Wait until it becomes `false`. You can also check `vector_store_exists`.
3.  **Ask Questions:** Once indexing is complete and the vector store exists, use the `/ask` endpoint to query your knowledge base.

# LLM RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system using Python. It leverages a language model combined with a knowledge base to generate responses.

## Directory Structure

```
.
├── main.py                   # Main application script (entry point)
├── rag_core.py               # Core RAG logic implementation
├── requirements.txt          # Python dependencies
├── run.sh                    # Script to run the application
├── knowledge_base/           # Directory for knowledge source documents
│   └── uploaded pdf files    # Example knowledge document
├── models/                   # Directory for language models
│   └── phi-2.Q4_K_M.gguf     # Language model file (Phi-2 GGUF)
└── README.md                 # This file
```

## Setup

1.  Ensure you have Python installed.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Usage

Execute the application using the provided run script:

```bash
./run.sh
```

## Core Components

*   **`main.py`**: Orchestrates the RAG process, handling input and output.
*   **`rag_core.py`**: Contains the functions for retrieving relevant information from the knowledge base and generating responses using the language model.
*   **`knowledge_base/`**: Stores the documents (like PDFs, text files, etc.) that the system searches through to find relevant context.
*   **`models/`**: Contains the pre-trained language model used for generation. This project uses the `phi-2.Q4_K_M.gguf` model.

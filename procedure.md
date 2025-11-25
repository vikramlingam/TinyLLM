# TinyLLM RAG and webui Setup and Usage Procedure

This guide explains how to set up and use the TinyLLM system on your local machine. Follow these steps in order to get everything running.

## 1. Install Dependencies

First, you need to install the required Python libraries. These libraries allow the code to run, manage the AI models, and create the web interface.

**Command:**
```bash
pip install -r requirements.txt
```

**What this does:**
It reads the `requirements.txt` file and installs packages like `onnxruntime-genai` (for the AI model), `nicegui` (for the interface), `sentence-transformers` (for document understanding), and `faiss-cpu` (for fast search).

## 2. Download the Chat Model

Next, you need to download the main AI model that you will chat with. We use a lightweight version of Microsoft's Phi-3 model optimized for CPUs.

**Command:**
```bash
python setup_model.py
```

**What this does:**
This script connects to Hugging Face and downloads the model files into a folder named `model` in your current directory. This is the "brain" of the chatbot.

## 3. Download the Embedding Model

If you want to chat with your own documents, you need a second, smaller model. This model converts text into numbers (embeddings) so the system can search through your documents.

**Command:**
```bash
python setup_embeddings.py
```

**What this does:**
This script downloads the `all-MiniLM-L6-v2` model. This model is specialized in understanding the meaning of sentences and is used to find relevant parts of your uploaded documents.

## 4. Run the Web Interface

Now that you have the models and dependencies, you can start the application.

**Command:**
```bash
python webui.py
```

**What this does:**
This starts the web server. It will open a window in your browser where you can chat with the AI.

### How to Use the Web Interface

1.  **Chatting:** You can type messages in the box at the bottom and press Enter to chat with the AI normally.
2.  **Uploading Documents:**
    *   Click the "Upload Text/PDF/DOCX" area in the sidebar.
    *   Select a `.txt`, `.pdf`, or `.docx` file from your computer.
    *   The system will read the file and prepare it for searching (this is called "indexing").
    *   You will see a "RAG: Active" status turn green.
3.  **Chatting with Documents:**
    *   Once a document is uploaded, ask questions about it.
    *   The system will find the relevant sections from your file and use them to answer your question.
    *   If you say simple things like "hello" or "thanks", the system will reply normally without searching the document.
4.  **Clearing Documents:**
    *   Click "Clear Document" to remove the current file and reset the search index.

## Code Overview

Here is a brief explanation of the files in the project:

*   **`webui.py`**: The main application file. It creates the user interface, handles user input, manages the chat history, and coordinates the AI models.
*   **`embedding_utils.py`**: Contains the logic for reading files (PDF, Word, Text), breaking them into small chunks, and creating the search index using FAISS.
*   **`setup_model.py`**: A one-time script to download the main chat model.
*   **`setup_embeddings.py`**: A one-time script to download the document embedding model.
*   **`requirements.txt`**: A list of all the software libraries needed to run this project.

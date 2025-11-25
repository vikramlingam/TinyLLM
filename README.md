# TinyLLM: Local CPU Inference System

TinyLLM project is a lightweight, privacy-focused AI inference system designed to run locally on standard laptops without a dedicated GPU. It utilizes Microsoft's ONNX Runtime (`onnxruntime-genai`) and the `Phi-3.5-mini-instruct` model optimized for CPU performance.

## 🚀 Features
- **Local Inference**: Runs entirely on your machine. No data leaves your device.
- **CPU Optimized**: Uses INT4 quantization for fast performance on standard hardware.
- **Dual Interface**: 
  - **CLI**: Interactive terminal chat.
  - **API**: FastAPI server for integration with other apps.

## 🛠️ Installation

1.  **Clone or Open the Project Directory**
    Ensure you are in the `TinyLLM` directory.

2.  **Install Dependencies**
    Create a virtual environment (optional but recommended) and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## 📥 Model Setup

Before running the chat or API, you must download the model weights.

Run the setup script:
```bash
python setup_model.py
```
This will download the `microsoft/Phi-3.5-mini-instruct-onnx` model (CPU INT4 version) into the `./model` directory.

## 🖥️ CLI Usage (Interactive Chat)

To chat with the model directly in your terminal:

1.  Run the CLI script:
    ```bash
    python chat_cli.py
    ```
2.  Wait for the "Model loaded" message.
3.  Type your message and press Enter. The assistant will stream the response back to you.
4.  Type `exit` or `quit` to close the application.

## 🌐 API Usage (Local Server)

To host the model as a local API service:

1.  Start the server:
    ```bash
    python serve_api.py
    ```
    The server will start at `http://0.0.0.0:8000`.

2.  **Endpoint**: `POST /chat`
    - **URL**: `http://localhost:8000/chat`
    - **Headers**: `Content-Type: application/json`
    - **Body**:
      ```json
      {
        "prompt": "Your prompt here",
        "max_tokens": 500
      }
      ```

## 🔌 Integration Examples

Here is how you can use the TinyLLM API in your own applications.

### 1. Python (using `requests`)

You can easily integrate this into any Python script or application.

```python
import requests
import json

def ask_tiny_llm(prompt):
    url = "http://localhost:8000/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise error for bad status codes
        
        result = response.json()
        return result["response"]
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Example Usage
user_input = "Explain quantum computing in one sentence."
answer = ask_tiny_llm(user_input)
print(f"TinyLLM: {answer}")
```

### 2. cURL (Command Line)

You can test the API directly from your terminal using `curl`:

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a haiku about coding.", "max_tokens": 100}'
```

### 3. JavaScript / Node.js (using `fetch`)

```javascript
async function askTinyLLM(prompt) {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: prompt,
      max_tokens: 500,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  return data.response;
}

// Example Usage
askTinyLLM("What is the capital of Japan?").then(answer => {
  console.log("TinyLLM:", answer);
});
```

## ❓ Troubleshooting

- **Model not found**: Ensure you ran `python setup_model.py` and that the `./model` directory contains `.onnx` files.
- **Out of Memory**: Close other memory-intensive applications. This model requires ~1-2GB of RAM.
- **Slow Generation**: Performance depends on your CPU. Ensure you are not running heavy background tasks.

Please refer to procedure.md for RAG and webui setup and usage instructions.
<img width="1724" height="1085" alt="image" src="https://github.com/user-attachments/assets/29dcb643-9f82-4741-a760-92399df598e7" />


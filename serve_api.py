from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime_genai as og
import os
import contextlib

# Global variables for model and tokenizer
model = None
tokenizer = None
tokenizer_stream = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, tokenizer_stream
    model_path = "./model"
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print("Error: Model not found in ./model. Please run setup_model.py first.")
        # In a real app we might raise an error, but here we just warn
    else:
        print("Loading model...")
        try:
            model = og.Model(model_path)
            tokenizer = og.Tokenizer(model)
            tokenizer_stream = tokenizer.create_stream()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
    yield
    # Cleanup if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 500

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Phi-3 prompt structure
        formatted_prompt = f"<|user|>\n{request.prompt}<|end|>\n<|assistant|>\n"
        
        input_tokens = tokenizer.encode(formatted_prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=request.max_tokens)
        
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)

        generated_text = ""
        
        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            decoded_token = tokenizer_stream.decode(new_token)
            generated_text += decoded_token
            
        return ChatResponse(response=generated_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

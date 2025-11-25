import os
from sentence_transformers import SentenceTransformer

def setup_embeddings():
    model_name = 'all-MiniLM-L6-v2'
    print(f"Downloading embedding model: {model_name}...")
    # This will download the model to the local cache (usually ~/.cache/huggingface/hub)
    # or we can specify a path if we want it local to the project, but cache is standard.
    model = SentenceTransformer(model_name)
    print("Embedding model downloaded successfully.")

if __name__ == "__main__":
    setup_embeddings()

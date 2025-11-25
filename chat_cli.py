import onnxruntime_genai as og
import os
import sys

def main():
    model_path = "./model"
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print("Error: Model not found in ./model. Please run setup_model.py first.")
        sys.exit(1)

    print("Loading model... (this may take a few seconds)")
    try:
        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("Model loaded. Type 'exit' to quit.")
    print("-" * 50)

    while True:
        try:
            text = input("User: ")
            if text.lower() in ["exit", "quit"]:
                break
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        # Phi-3 prompt structure
        prompt = f"<|user|>\n{text}<|end|>\n<|assistant|>\n"

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=2048)

        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)

        print("Assistant: ", end="", flush=True)

        try:
            while not generator.is_done():
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end="", flush=True)
        except Exception as e:
            print(f"\nError during generation: {e}")
        
        print("\n")

if __name__ == "__main__":
    main()

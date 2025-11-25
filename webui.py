import asyncio
import io
import os
import sys
import threading
import time
from typing import Generator

import gc
import onnxruntime_genai as og
import pypdf
from nicegui import app, ui, run
from embedding_utils import RAGHandler

# --- LLM Interface ---

class LLMInterface:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.tokenizer_stream = None
        self.is_loaded = False

    def load_model(self):
        """Loads the model. This should be run in a separate thread/executor to avoid blocking."""
        if self.is_loaded:
            return
        
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
            self.tokenizer_stream = self.tokenizer.create_stream()
            self.is_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    async def generate(self, prompt: str, max_tokens: int = 500) -> Generator[str, None, None]:
        """Generates text from the model, yielding tokens as they are generated."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Run the setup in an executor to avoid blocking
        def _prepare_generator():
            input_tokens = self.tokenizer.encode(prompt)
            input_len = len(input_tokens)
            
            # Calculate total max length (input + new tokens)
            # Ensure we don't exceed model context (approx 4096 for Phi-3)
            model_context_limit = 4096
            total_max_length = min(input_len + max_tokens, model_context_limit)
            
            if input_len >= model_context_limit:
                print(f"Warning: Input length ({input_len}) exceeds model limit ({model_context_limit}). Truncating...")
                # Simple truncation (not ideal but prevents crash)
                input_tokens = input_tokens[-model_context_limit+max_tokens:]
                total_max_length = model_context_limit

            params = og.GeneratorParams(self.model)
            params.set_search_options(max_length=total_max_length)
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            return generator

        generator = await run.io_bound(_prepare_generator)

        while not generator.is_done():
            # Run the next token generation in an executor
            await run.io_bound(generator.generate_next_token)
            
            new_token = generator.get_next_tokens()[0]
            decoded_token = self.tokenizer_stream.decode(new_token)
            yield decoded_token

# --- Global State ---

llm = LLMInterface("./model")
document_content = None
document_name = None
rag = RAGHandler()
rag_active = False

# --- UI ---

@ui.page('/')
async def main_page():
    # Theme settings
    ui.colors(primary='#3B82F6', secondary='#10B981', accent='#8B5CF6', dark='#1F2937')
    ui.query('body').classes('bg-gray-900 text-white')
    
    # State
    messages = []
    is_generating = False
    
    # Header
    with ui.header().classes('bg-gray-800 border-b border-gray-700 p-4 flex items-center justify-between'):
        ui.label('TinyLLM WebUI').classes('text-xl font-bold text-primary')
        with ui.row().classes('items-center gap-4'):
            rag_status = ui.label('RAG: Off').classes('text-sm text-gray-400')
            status_label = ui.label('Model: Not Loaded').classes('text-sm text-red-400')

    # Sidebar
    with ui.left_drawer(value=True).classes('bg-gray-800 border-r border-gray-700 p-4 space-y-4'):
        ui.label('Settings').classes('text-lg font-semibold text-gray-200')
        
        system_prompt = ui.textarea('System Prompt', value="You are a helpful AI assistant.").classes('w-full').props('rows=4')
        
        max_tokens = ui.slider(min=10, max=2048, value=500).props('label-always')
        ui.label('Max Tokens').classes('text-xs text-gray-400')
        
        ui.separator().classes('my-4')
        
        ui.label('Document Context').classes('text-lg font-semibold text-gray-200')
        
        doc_label = ui.label('No document loaded').classes('text-sm text-gray-400 italic break-all')
        
        async def handle_upload(e):
            nonlocal doc_label
            global document_content, document_name
            
            try:
                content = e.content.read()
                filename = e.name
                
                # Logic moved to RAGHandler
                document_content = content # Store raw bytes/content
                
                document_name = filename
                doc_label.text = f"Loaded: {filename}"
                doc_label.classes(remove='italic text-gray-400', add='text-green-400')
                
                # Create RAG index dynamically
                with ui.dialog() as loading_dialog, ui.card():
                    ui.label('Indexing document... Please wait.')
                    ui.spinner(size='lg')
                
                loading_dialog.open()
                try:
                    ui.notify(f"Indexing {filename}...", type='info')
                    # Use io_bound (threading) instead of cpu_bound (multiprocessing) to avoid pickling lag
                    await run.io_bound(rag.create_index, document_content, filename)
                    
                    global rag_active
                    rag_active = True
                    rag_status.text = 'RAG: Active'
                    rag_status.classes(remove='text-gray-400', add='text-green-400')
                    
                    ui.notify(f"Indexed {filename} successfully!", type='positive')
                finally:
                    loading_dialog.close()
                
            except Exception as err:
                ui.notify(f"Failed to upload: {err}", type='negative')

        ui.upload(on_upload=handle_upload, label="Upload Text/PDF/DOCX", auto_upload=True).props('accept=".txt,.pdf,.docx"').classes('w-full')
        
        def clear_doc():
            global document_content, document_name, rag_active
            document_content = None
            document_name = None
            rag.reset_index()
            rag_active = False
            rag_status.text = 'RAG: Off'
            rag_status.classes(remove='text-green-400', add='text-gray-400')
            
            doc_label.text = 'No document loaded'
            doc_label.classes(remove='text-green-400', add='italic text-gray-400')
            ui.notify("Document cleared and Index reset", type='info')

        ui.button('Clear Document', on_click=clear_doc).props('outline color=red size=sm').classes('w-full')

    # Chat Area
    with ui.column().classes('w-full max-w-4xl mx-auto p-4 flex-grow space-y-4 mb-20') as chat_container:
        pass # Messages will be added here

    # Input Area
    with ui.footer().classes('bg-gray-800 border-t border-gray-700 p-4'):
        with ui.row().classes('w-full max-w-4xl mx-auto items-center gap-2'):
            user_input = ui.input(placeholder='Type a message...').classes('flex-grow').props('rounded outlined input-class="text-white"')
            send_button = ui.button(icon='send').props('round flat color=primary')

    # Logic
    async def update_status():
        if llm.is_loaded:
            status_label.text = 'Model: Ready'
            status_label.classes(remove='text-red-400', add='text-green-400')
        else:
            status_label.text = 'Model: Loading...'
            status_label.classes(remove='text-green-400', add='text-yellow-400')

    async def load_model_task():
        if not os.path.exists(llm.model_path) or not os.listdir(llm.model_path):
            ui.notify("Model not found in ./model. Please run setup_model.py first.", type='negative', close_button=True, timeout=0)
            status_label.text = 'Model: Missing'
            return

        # Check for RAG index (not needed on startup anymore as it's dynamic)
        # global rag_active
        # if rag.load_index():
        #     rag_active = True
        #     rag_status.text = 'RAG: Active'
        #     rag_status.classes(remove='text-gray-400', add='text-green-400')
        #     ui.notify("RAG Index loaded.", type='positive')
        # else:
        #     rag_status.text = 'RAG: Off'
            
        await update_status()
        try:
            await run.io_bound(llm.load_model)
            await update_status()
            ui.notify("Model loaded successfully!", type='positive')
        except Exception as e:
            ui.notify(f"Failed to load model: {e}", type='negative')
            status_label.text = 'Model: Error'
            status_label.classes(remove='text-yellow-400', add='text-red-400')

    async def send_message():
        nonlocal is_generating
        text = user_input.value.strip()
        if not text or is_generating:
            return
        
        if not llm.is_loaded:
            ui.notify("Model is not loaded yet.", type='warning')
            return

        user_input.value = ''
        is_generating = True
        send_button.disable()
        
        # Add user message
        with chat_container:
            ui.chat_message(text, name='You', sent=True, avatar='https://robohash.org/you?set=set4')
        
        # Prepare prompt
        global document_content
        context_str = ""
        
        # Priority: Uploaded document > RAG Index
        # Note: document_content is now raw bytes/text, so we don't display it directly in prompt if it's binary
        # We rely on RAG for context.
        
        # Simple conversational filter
        conversational_triggers = {'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'goodbye', 'ok', 'okay'}
        is_conversational = text.lower().strip().strip('!.,?') in conversational_triggers
        
        if rag_active and not is_conversational:
            # Retrieve from RAG
            # Use io_bound for retrieval too
            retrieved_chunks = await run.io_bound(rag.query, text)
            if retrieved_chunks:
                context_str = "Context from knowledge base:\n" + "\n---\n".join(retrieved_chunks) + "\n\n"
        elif is_conversational:
             print(f"Skipping RAG for conversational input: {text}")
            
        full_prompt = f"<|user|>\n{system_prompt.value}\n{context_str}{text}<|end|>\n<|assistant|>\n"
        
        # Add assistant message placeholder
        with chat_container:
            message_row = ui.chat_message(name='Bot', sent=False, avatar='https://robohash.org/bot?set=set4')
            spinner = ui.spinner(size='sm')
            message_content = ui.html('')
            
            # Initially show spinner, remove it when first token arrives
            message_row.clear()
            with message_row:
                spinner
        
        response_text = ""
        
        try:
            async for token in llm.generate(full_prompt, max_tokens=int(max_tokens.value)):
                if spinner.visible:
                    spinner.visible = False
                    # Re-render message row to show text instead of spinner
                    message_row.clear()
                    with message_row:
                        message_content = ui.html(response_text)
                
                response_text += token
                message_content.content = response_text.replace('\n', '<br>')
                
        except Exception as e:
            ui.notify(f"Error generating response: {e}", type='negative')
        
        is_generating = False
        send_button.enable()

    user_input.on('keydown.enter', send_message)
    send_button.on('click', send_message)

    # Start loading model in background (using timer to keep context)
    ui.timer(0.1, load_model_task, once=True)

    # Shutdown handler
    def shutdown():
        print("Shutting down...")
        if llm.model:
            del llm.model
            llm.model = None
        if llm.tokenizer:
            del llm.tokenizer
            llm.tokenizer = None
        if llm.tokenizer_stream:
            del llm.tokenizer_stream
            llm.tokenizer_stream = None
        gc.collect()
        print("Cleanup complete.")

    app.on_shutdown(shutdown)

ui.run(title='TinyLLM WebUI', dark=True)

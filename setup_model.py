import os
from huggingface_hub import snapshot_download

def download_model():
    """
    Downloads the specific CPU-optimized version of Phi-3.5-mini-instruct-onnx.
    """
    model_id = "microsoft/Phi-3.5-mini-instruct-onnx"
    # Specific subfolder for CPU INT4 (AWQ version as RTN is missing)
    subfolder = "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
    local_dir = "./model"

    print(f"Downloading {model_id} (subfolder: {subfolder})...")
    
    try:
        snapshot_download(
            repo_id=model_id,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        # The snapshot_download with allow_patterns might create the directory structure
        # inside local_dir. We want the files directly in ./model if possible, 
        # or we just need to know where they are.
        # Let's check where it put them.
        # Usually it puts them in local_dir/subfolder if we use allow_patterns with full path?
        # Actually, let's just download to ./model and then if it's nested, the user code needs to handle it
        # or we move it.
        # But wait, snapshot_download with local_dir usually mirrors the repo structure if allow_patterns is used.
        # Let's see.
        
        print("Download complete.")
        
        # Check if the files are in ./model or ./model/cpu_and_mobile/...
        # If they are nested, we might want to move them up for simpler access, 
        # or just update our paths in the other scripts.
        # For simplicity in other scripts, let's move them to ./model root if they are nested.
        
        nested_path = os.path.join(local_dir, subfolder)
        if os.path.exists(nested_path):
            print(f"Moving files from {nested_path} to {local_dir}...")
            import shutil
            
            # Move files
            for filename in os.listdir(nested_path):
                src = os.path.join(nested_path, filename)
                dst = os.path.join(local_dir, filename)
                if os.path.isfile(src):
                    shutil.move(src, dst)
                elif os.path.isdir(src):
                    # If there are subdirectories, move them too (though unlikely for this model folder)
                    if os.path.exists(dst):
                         shutil.rmtree(dst)
                    shutil.move(src, dst)
            
            # Clean up empty directories
            # Remove cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
            # Remove cpu_and_mobile
            try:
                os.rmdir(nested_path)
                os.rmdir(os.path.join(local_dir, "cpu_and_mobile"))
            except OSError:
                pass # Directories might not be empty if something else is there
                
            print("Files moved to ./model root.")

    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model()


from huggingface_hub import hf_hub_download
import os

model_id = "earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
filename = "open_clip_model.safetensors"

print(f"Starting download for {model_id}/{filename}...")
print("This may take a while (approx 3.7 GB)...")

try:
    path = hf_hub_download(
        repo_id=model_id, 
        filename=filename,
        resume_download=True,
        local_files_only=False
    )
    print(f"\nSUCCESS: Weights downloaded to: {path}")
    print(f"File size: {os.path.getsize(path) / (1024**3):.2f} GB")
    
except Exception as e:
    print(f"\nFAILED: {e}")

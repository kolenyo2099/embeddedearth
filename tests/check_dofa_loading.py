
import torch
from transformers import AutoModel, AutoConfig

print("Attempting to load XShadow/DOFA-CLIP...")

try:
    model_id = "earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
    print(f"Checking {model_id} with explicit SiglipModel...")
    
    from transformers import SiglipConfig, SiglipModel
    
    # Try config
    config = SiglipConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"Config loaded: {config}")

    from huggingface_hub import list_repo_files
    files = list_repo_files(model_id)
    print(f"Files in repo: {files}")
    
    # Check for likely candidates
    weight_files = [f for f in files if f.endswith('.bin') or f.endswith('.safetensors') or f.endswith('.pt') or f.endswith('.pth')]
    print(f"Potential weight files: {weight_files}")
    
except Exception as e:
    print(f"Failed to load model: {e}")

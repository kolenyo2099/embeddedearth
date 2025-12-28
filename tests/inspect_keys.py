
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import sys

model_id = "earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
filename = "open_clip_model.safetensors"

print(f"Loading keys from {filename}...")
try:
    path = hf_hub_download(repo_id=model_id, filename=filename)
    state_dict = load_file(path)
    
    keys = sorted(list(state_dict.keys()))
    print(f"\nTotal keys in checkpoint: {len(keys)}")
    
    print("\n--- Checkpoint Keys (Visual Trunk start) ---")
    trunk_keys = [k for k in keys if "visual" in k and "trunk" in k]
    for k in trunk_keys[:20]:
        print(k)
        
    print("\n--- Checkpoint Keys & Shapes (Patch/Embed related) ---")
    embed_keys = [k for k in keys if "embed" in k or "patch" in k]
    for k in embed_keys:
        print(f"{k}: {state_dict[k].shape}")

except Exception as e:
    print(e)

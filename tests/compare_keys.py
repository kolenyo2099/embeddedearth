import torch
import sys
import os
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dofa_clip import get_model, DynamicPatchEmbed

def compare_keys():
    print("Downloading weights for inspection...")
    model_id = "earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
    checkpoint_path = hf_hub_download(repo_id=model_id, filename="open_clip_model.safetensors")
    
    print("Loading state dict keys...")
    state_dict = load_file(checkpoint_path)
    chk_keys = set(state_dict.keys())
    
    print("Initializing Model...")
    wrapper = get_model()
    model = wrapper.model
    model_keys = set(model.state_dict().keys())
    
    print("\n--- Key Analysis ---")
    
    # Check patch_embed keys specifically
    print("\nChecking 'patch_embed' keys in CHECKPOINT:")
    pe_chk = sorted([k for k in chk_keys if 'patch_embed' in k])
    for k in pe_chk: print(f"  {k}")
    
    print("\nChecking 'patch_embed' keys in MODEL:")
    pe_mod = sorted([k for k in model_keys if 'patch_embed' in k])
    for k in pe_mod: print(f"  {k}")
    
    # Intersection
    common = set(pe_chk) & set(pe_mod)
    missing_in_model = set(pe_chk) - set(pe_mod)
    missing_in_chk = set(pe_mod) - set(pe_chk)
    
    print(f"\nMatched Keys (Total): {len(chk_keys.intersection(model_keys))}")
    print(f"Total Checkpoint Keys: {len(chk_keys)}")
    print(f"Total Model Keys: {len(model_keys)}")
    
    print(f"\nMissing in Model (Checkpoint has, Model doesn't) - First 10:")
    for k in sorted(list(missing_in_model))[:10]: print(f"  {k}")
    
    print(f"\nMissing in Checkpoint (Model has, Checkpoint doesn't) - First 10:")
    for k in sorted(list(missing_in_chk))[:10]: print(f"  {k}")

if __name__ == "__main__":
    compare_keys()

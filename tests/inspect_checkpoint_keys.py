
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from models.dofa_clip import DynamicPatchEmbed

def inspect_mismatch():
    print("--- 1. Inspecting Checkpoint Keys ---")
    repo_id = "earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
    try:
        path = hf_hub_download(repo_id=repo_id, filename="open_clip_model.safetensors")
        state_dict = load_file(path)
        
        patch_keys = [k for k in state_dict.keys() if "patch_embed" in k]
        print(f"Found {len(patch_keys)} patch_embed keys in checkpoint.")
        for k in patch_keys[:10]:
            print(f"  CP Key: {k}")
            
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    print("\n--- 2. Inspecting Model Structure Keys ---")
    # Instantiate the layer as we do in the app
    layer = DynamicPatchEmbed(img_size=384, patch_size=14, embed_dim=1152, weight_dim=128)
    
    # We want to know what keys this layer expects if it were at 'visual.trunk.patch_embed'
    model_mock = nn.Sequential()
    visual = nn.Sequential()
    trunk = nn.Sequential()
    
    # Mimic the hierarchy: model.visual.trunk.patch_embed
    # BUT wait, state_dict keys are flattened.
    # If I assign it to a member, I can check expected keys.
    
    # Just print the local keys of the layer
    print("Layer Keys (local):")
    for name, param in layer.named_parameters():
        print(f"  Local Key: {name} | Shape: {param.shape}")
        
    print("\n--- 3. Hypothetical Full Keys ---")
    prefix = "visual.trunk.patch_embed."
    for name, _ in layer.named_parameters():
        print(f"  Expected: {prefix}{name}")

if __name__ == "__main__":
    inspect_mismatch()

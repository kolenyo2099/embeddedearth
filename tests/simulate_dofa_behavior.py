
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from models.dofa_clip import DynamicPatchEmbed
import math

def simulate():
    print("--- Simulating DOFA Behavior ---")
    
    # 1. Instantiate Layer
    layer = DynamicPatchEmbed(img_size=384, patch_size=14, embed_dim=1152, weight_dim=128)
    
    # 2. Load Weights
    repo_id = "earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
    path = hf_hub_download(repo_id=repo_id, filename="open_clip_model.safetensors")
    full_state_dict = load_file(path)
    
    # Extract keys for patch_embed
    layer_state_dict = {}
    prefix = "visual.trunk.patch_embed."
    
    print(f"Loading weights from {path}...")
    
    for k, v in full_state_dict.items():
        if k.startswith(prefix):
            local_key = k.replace(prefix, "")
            layer_state_dict[local_key] = v
            
    # Load
    missing, unexpected = layer.load_state_dict(layer_state_dict, strict=False)
    print(f"Loading Result: Missing={len(missing)}, Unexpected={len(unexpected)}")
    if missing:
        print(f"Missing keys: {missing}")
        
    # 3. Test Cases
    print("\n--- Testing Wavelengths ---")
    
    cases = {
        "Nanometers (490)": torch.tensor([490.0, 560.0, 665.0, 842.0]).float(),
        "Micrometers (0.49)": torch.tensor([0.490, 0.560, 0.665, 0.842]).float(),
        "Random (No Init)": torch.tensor([0.490]).float() # Control
    }
    
    inputs = torch.randn(1, 4, 384, 384) # 4 bands
    
    for name, waves in cases.items():
        if name == "Random (No Init)":
            # For this test we re-init random
            layer_rand = DynamicPatchEmbed(img_size=384, patch_size=14, embed_dim=1152, weight_dim=128)
            layer_rand.set_wavelengths(waves)
            # Use internal generator direct access to see produced weights
            wave_emb = layer_rand.fclayer(layer_rand.current_wavelengths).unsqueeze(0)
            w, b = layer_rand.weight_generator(wave_emb)
        else:
            layer.set_wavelengths(waves)
            print(f"  Fingerprint (Embed Mean): {layer.current_wavelengths.mean().item():.6f}")
            print(f"  Fingerprint (Embed Std):  {layer.current_wavelengths.std().item():.6f}")
            wave_emb = layer.fclayer(layer.current_wavelengths).unsqueeze(0)
            w, b = layer.weight_generator(wave_emb)
            
        print(f"\nCase: {name}")
        print(f"  Weights stats: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")
        print(f"  Bias stats:    min={b.min():.4f}, max={b.max():.4f}, mean={b.mean():.4f}")

if __name__ == "__main__":
    simulate()

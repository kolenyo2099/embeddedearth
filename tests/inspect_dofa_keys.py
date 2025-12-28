
import open_clip
import torch
import sys

model_name = "ViT-SO400M-14-SigLIP-384"
pretrained = "hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"

print(f"Loading {model_name} from {pretrained}...")

try:
    # Just load the state dict logic from open_clip if possible?
    # Or instantitate and see what loads.
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        force_custom_text=True,
    )
    
    print("\n[SUCCESS] Model loaded by open_clip.")
    print("Inspecting keys related to patch_embed...")
    
    keys = model.state_dict().keys()
    patch_keys = [k for k in keys if "patch_embed" in k]
    
    if not patch_keys:
        print("[WARN] No 'patch_embed' keys found!")
    else:
        for k in patch_keys:
            print(f"  {k} -> {model.state_dict()[k].shape}")
            
    # Check for weight generator
    gen_keys = [k for k in keys if "weight_generator" in k or "fclayer" in k]
    if gen_keys:
        print(f"\n[FOUND] Dynamic Weights keys found: {len(gen_keys)}")
        for k in gen_keys:
            print(f"  {k}")
    else:
        print("\n[FAIL] No 'weight_generator' or 'fclayer' keys found in the loaded model.")
        print("This confirms we are NOT loading the Dynamic Patch Embed weights.")
        
except Exception as e:
    print(f"[ERROR] {e}")


import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dofa_clip import get_model
from config import sentinel2_bands

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def test_forward():
    print("--- STARTING LOCAL MODEL VERIFICATION ---")
    
    # 1. Load Model
    print("1. Loading model...")
    wrapper = get_model()
    model = wrapper.model
    device = wrapper.device
    print(f"   Model loaded on {device}")
    
    # 2. Check Patch Embed
    print("\n2. Checking Patch Embed...")
    patch_embed = model.visual.trunk.patch_embed
    print(f"   Type: {type(patch_embed)}")
    
    # Check wavelengths
    waves = patch_embed.current_wavelengths
    print(f"   Current Wavelengths shape: {waves.shape}")
    print(f"   Current Wavelengths sample: {waves[0, :5]}...")
    if waves.shape == (1, 1):
        print("   CRITICAL: Wavelengths not set correctly (Still placeholder)!")
    
    # 3. Dummy Input
    print("\n3. Running Dummy Inference...")
    # Sentinel-2 has 13 bands in config? No, bands list has 12?
    # Let's check configured bands
    bands_list = sentinel2_bands.band_names
    print(f"   Configured bands: {len(bands_list)}")
    
    # Create input matching config bands
    dummy_input = torch.rand(1, len(bands_list), 384, 384).to(device)
    print(f"   Input shape: {dummy_input.shape}")
    
    # 4. Forward
    print("   Calling encode_image...")
    with torch.no_grad():
        emb = wrapper.encode_image(dummy_input)
        
    print(f"   Output embedding shape: {emb.shape}")
    print(f"   Output embedding norm: {emb.norm(dim=-1).item()}")
    
    # 4. Inspecting DynamicPatchEmbed Internals...
    
    # TEST: Try Micrometers!
    print("   \n[TEST] Switching to Micrometers (0.49) to check weight stability...")
    micrometers = torch.tensor(sentinel2_bands.get_wavelength_tensor()).float().to(device) / 1000.0
    print(f"   Micrometers: {micrometers[0]:.4f}...")
    
    wrapper.vision_model.trunk.patch_embed.set_wavelengths(micrometers)
    
    # Run just the patch embed
    print("   Running patch_embed.forward() manually with um...")
    try:
        x = patch_embed(dummy_input)
        print(f"   Patch Embed Result shape: {x.shape}")
        print(f"   Patch Embed Result stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
        
        if x.sum() == 0:
            print("   CRITICAL: Patch Embed output is ZERO!")
            
            # Debug why
            print("   Generic Weights debugging:")
            # Can we access weight generator?
            # Manually run generator
            wave_emb = patch_embed.fclayer(patch_embed.current_wavelengths).unsqueeze(0)
            w, b = patch_embed.weight_generator(wave_emb)
            print(f"   Generated w shape: {w.shape}")
            print(f"   Generated w stats: min={w.min():.4f}, max={w.max():.4f}")
            if w.sum() == 0:
                print("   CRITICAL: Generated weights are ZERO!")
            
    except Exception as e:
        print(f"   Failed to run patch_embed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forward()

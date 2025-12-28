
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dofa_clip import get_model

def inspect_structure():
    print("Initializing model...")
    wrapper = get_model()
    model = wrapper.model
    visual = model.visual
    
    print("\n[INSPECT] Visual Encoder Structure:")
    print(f"Type: {type(visual)}")
    
    has_patch_embed = hasattr(visual, 'patch_embed')
    print(f"Has 'patch_embed': {has_patch_embed}")
    
    if has_patch_embed:
        print(f"Type of visual.patch_embed: {type(visual.patch_embed)}")
        if hasattr(visual.patch_embed, 'weight_generator'):
            print("  -> CONFIRMED: visual.patch_embed IS DynamicPatchEmbed")
        else:
            print("  -> WARNING: visual.patch_embed is standard (Static)")
            
    has_trunk = hasattr(visual, 'trunk')
    print(f"Has 'trunk': {has_trunk}")
    
    if has_trunk:
        trunk = visual.trunk
        print(f"Type of visual.trunk: {type(trunk)}")
        
        has_trunk_pe = hasattr(trunk, 'patch_embed')
        print(f"Has 'trunk.patch_embed': {has_trunk_pe}")
        
        if has_trunk_pe:
            print(f"Type of visual.trunk.patch_embed: {type(trunk.patch_embed)}")
            if hasattr(trunk.patch_embed, 'weight_generator'):
                 print("  -> CONFIRMED: visual.trunk.patch_embed IS DynamicPatchEmbed")
            else:
                 print("  -> WARNING: visual.trunk.patch_embed is standard (Static)")

if __name__ == "__main__":
    inspect_structure()

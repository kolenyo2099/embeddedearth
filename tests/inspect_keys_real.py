
import torch
import sys
from pathlib import Path

path = "models/weights/RealESRGAN_x4plus.pth"
print(f"Loading {path}...")

try:
    loadnet = torch.load(path, map_location='cpu')
    if 'params_ema' in loadnet:
        state_dict = loadnet['params_ema']
    elif 'params' in loadnet:
        state_dict = loadnet['params']
    else:
        state_dict = loadnet
        
    print(f"Keys found: {len(state_dict)}")
    for k in list(state_dict.keys())[:20]:
        print(k)
        
    # Check for specific keys causing issues
    print("\nCheck specific patterns:")
    for k in state_dict.keys():
        if "conv" in k and "body" in k:
            print(f"Found body conv: {k}")
        if "trunk" in k:
            print(f"Found trunk: {k}")
            
except Exception as e:
    print(e)

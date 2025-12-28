
import sys
import os
sys.path.insert(0, os.getcwd())
import torch
from models.dofa_clip import get_model

print("Initializing DOFA-CLIP wrapper with OpenCLIP...")
wrapper = get_model()
wrapper._load_model()

print(f"Model loaded: {wrapper.model_name}")
print(f"Vision/Image Embed Dim: {wrapper.vision_model.output_dim if hasattr(wrapper.vision_model, 'output_dim') else 'Unknown'}")
if hasattr(wrapper.model, 'text_projection'):
    print(f"Text Projection: {wrapper.model.text_projection.shape if wrapper.model.text_projection is not None else 'None'}")
    
# Encode test
print("Running encoding test...")
txt_emb = wrapper.encode_text("test")
print(f"Text Emb Shape: {txt_emb.shape}")

img = torch.randn(1, 3, 224, 224)
img_emb = wrapper.encode_image(img)
print(f"Image Emb Shape: {img_emb.shape}")

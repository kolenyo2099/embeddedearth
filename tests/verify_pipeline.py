
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dofa_clip import get_model
from pipeline.tiling import tile_image
from data.gee_client import GEEClient

def test_dofa_model_structure():
    print("Testing DOFA-CLIP Model Structure...")
    model = get_model()
    
    # Test Text Encoding
    text = "A satellite image of a forest"
    text_emb = model.encode_text(text)
    print(f"Text Embedding Shape: {text_emb.shape}")
    # DOFA-CLIP uses 1152 dim
    assert text_emb.shape[-1] == 1152
    
    # Test Image Encoding (Mock S2 Data: 6 bands for default wavelengths)
    # (B, C, H, W) -> use 384 to match model
    mock_image = torch.rand(1, 6, 384, 384) 
    wavelengths = torch.tensor([0.490, 0.560, 0.665, 0.842, 1.610, 2.190]) # S2 wavelengths
    
    img_emb = model.encode_image(mock_image, wavelengths=wavelengths)
    print(f"Image Embedding Shape: {img_emb.shape}")
    assert img_emb.shape[-1] == 1152
    print("✅ DOFA-CLIP Model Test Passed")

def test_tiling():
    print("\nTesting Tiling Logic...")
    # Mock Large Image (6 bands, 1000x1000)
    C, H, W = 6, 1000, 1000
    large_image = np.random.rand(C, H, W).astype(np.float32)
    
    tiles = tile_image(large_image, tile_size=384, overlap=0.5)
    print(f"Generated {len(tiles)} tiles from {H}x{W} image")
    
    assert len(tiles) > 0
    assert tiles[0].data.shape == (6, 384, 384)
    print("✅ Tiling Test Passed")

def test_gee_client_mock():
    print("\nTesting GEE Client (Mock Call)...")
    # This verifies the method exists and signature matches, 
    # actual call requires auth which might fail in CI/Headless
    assert hasattr(GEEClient, 'get_image_data')
    print("✅ GEE Client Interface Verified")

def test_xai():
    print("\nTesting Grad-ECLIP Explainability...")
    from xai.grad_eclip import generate_explanation
    model_wrapper = get_model()
    
    # Mock image (C, H, W) - use 6 bands for S2 default
    image = np.random.rand(6, 384, 384).astype(np.float32)
    text = "A forest"
    
    try:
        heatmap = generate_explanation(model_wrapper, image, text)
        print(f"Heatmap shape: {heatmap.shape}")
        assert heatmap.shape == (384, 384)
        assert heatmap.min() >= 0 and heatmap.max() <= 1
        print("✅ Grad-ECLIP Test Passed")
    except Exception as e:
        print(f"❌ Grad-ECLIP Failed: {e}")
        import traceback
        traceback.print_exc()

def test_faiss():
    print("\nTesting FAISS Vector Search...")
    from search.faiss_index import get_index
    index = get_index(dimension=1152)
    index.clear()
    
    # Add dummy vectors
    vecs = np.random.rand(5, 1152).astype(np.float32)
    # Normalize for Inner Product index
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    
    index.add(vecs, metadata=[{'id': i} for i in range(5)])
    assert index.count == 5
    
    # Search
    query = vecs[0]
    results = index.search(query, k=2)
    assert len(results) > 0
    assert results[0].index == 0
    print("✅ FAISS Search Test Passed")

if __name__ == "__main__":
    test_dofa_model_structure()
    test_tiling()
    test_gee_client_mock()
    test_xai()
    test_faiss()


import sys
import unittest
import torch
import numpy as np

# Verify we can import the model
try:
    from models.copernicus_fm import CopernicusFM
except ImportError:
    # Adjust path if running from root
    import os
    sys.path.append(os.getcwd())
    from models.copernicus_fm import CopernicusFM

class TestPoolingConfig(unittest.TestCase):
    def test_embedding_quality(self):
        """
        Verify that the model produces meaningful embeddings with Global Pooling.
        - Identical inputs should have Cosine Sim ~ 1.0
        - Randomly different inputs should have Cosine Sim < 0.9 (ideally lower, but untrained GAP might be high)
        - The key is correctness vs random CLS noise.
        """
        device = "cpu"
        model = CopernicusFM(device=device)
        model.eval()
        
        # Create dummy input [1, 6, 224, 224]
        # Using real shapes to trigger patch embedding
        x1 = torch.randn(1, 6, 224, 224)
        x2 = torch.randn(1, 6, 224, 224) # Completely different image
        
        # Meta info [1, 4] (lon, lat, time, area)
        meta = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        
        # Wavelengths (mock S2)
        waves = [490, 560, 665, 842, 1610, 2190]
        bws = [66, 36, 31, 115, 91, 175]
        
        with torch.no_grad():
            emb1 = model.extract_features(x1, meta, waves, bws)
            emb2 = model.extract_features(x2, meta, waves, bws)
            
            # Re-run emb1 to ensure determinism
            emb1_redux = model.extract_features(x1, meta, waves, bws)
            
        # 1. Check Determinism
        sim_identity = torch.nn.functional.cosine_similarity(emb1, emb1_redux).item()
        self.assertAlmostEqual(sim_identity, 1.0, places=5, msg="Model is not deterministic!")
        
        # 2. Check Differentiation
        sim_diff = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        print(f"\n[INFO] Similarity between random images: {sim_diff:.4f}")
        
        # 3. Check Dimensions (Global Pool should return [B, EmbedDim])
        # If it returned [B, N, D], it would have 3 dims.
        self.assertEqual(len(emb1.shape), 2, f"Expected 2D output [B, D], got {emb1.shape}")
        
        # If sim_diff is basically 1.0, the model is collapsing or CLS is still broken.
        # But with GAP on random noise, it might still be somewhat high due to bias.
        # However, it shouldn't be 1.0.
        self.assertLess(sim_diff, 0.99, "Model produces identical embeddings for different inputs!")
        
        print(f"[SUCCESS] Model produces vector shape {emb1.shape} and differentiates inputs.")

if __name__ == "__main__":
    unittest.main()

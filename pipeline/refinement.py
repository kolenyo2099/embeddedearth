
import torch
import numpy as np
import open_clip
from typing import List, Tuple
from PIL import Image
from dataclasses import dataclass

from pipeline.super_resolution import get_sr_model
from pipeline.tiling import tile_image, Tile
from config import model_config

@dataclass
class RefinedResult:
    """A result from the semantic refinement pipeline."""
    original_tile_idx: int
    sub_tile: Tile
    score: float
    # We might add heatmap later

class SemanticRefiner:
    """
    Refines coarse search results using Super-Resolution and Fine-Grained RGB Search.
    
    Pipeline:
    1. Input: Low-Res (10m) Candidate Tiles.
    2. Upscale: 10m -> 2.5m (4x) using ESRGAN-WorldStrat.
    3. Re-Tile: Slide a window over the High-Res image.
    4. Rank: Score sub-tiles with a standard RGB CLIP model (SigLIP).
    """
    
    def __init__(self, device: str = None):
        self.device = device or model_config.device
        self.sr_model = get_sr_model() # Shared Singleton
        
        # Load RGB-Optimized Model (SigLIP)
        # Using a solid standard model: ViT-B-16-SigLIP (WebLI)
        # It's smaller than SO400M but much better than standard CLIP for web images.
        print("Loading Refinement Model (SigLIP)...")
        model_name = 'ViT-B-16-SigLIP'
        pretrained = 'webli'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        print("Refinement Model Loaded.")

    def refine_candidates(
        self, 
        candidates: List[Tuple[int, np.ndarray, object]], # (idx, image_data_HWC_uint8, original_tile_obj)
        query_text: str,
        top_k: int = 5
    ) -> List[RefinedResult]:
        """
        Run simple refinement on a batch of candidates.
        
        Args:
            candidates: List of (index, image_array, metadata). 
                       image_array should be RGB, HWC, 0-255 uint8 (or float 0-1).
            query_text: The text to search for (e.g. "small white plane").
            
        Returns:
            List of RefinedResult objects.
        """
        # Encode Text
        text = self.tokenizer([query_text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        results = []
        
        for idx, img_data, meta in candidates:
            # 1. Upscale
            # SR Model expects RGB.
            # Convert to float 0-1 if unit8
            if img_data.dtype == np.uint8:
                img_float = img_data.astype(np.float32) / 255.0
            else:
                img_float = img_data
            
            # Upscale (C, H, W) or (H, W, C)?
            # SR wrapper handles (H,W,C) input logic but expects standard format.
            # Assuming img_data is HWC from the way we visualize it (PIL-like).
            
            # We need to make sure we are passing consistent format.
            # Let's assume HWC for input to this function.
            
            sr_image = self.sr_model.upscale(img_data) # Returns (C, H*4, W*4) float 0-1
            
            # Convert back to HWC for tiling or viewing?
            # Tiling module expects (C, H, W) usually.
            # Let's check tiling.py... it takes (C, H, W).
            
            # 2. Re-Tile (Sliding Window on the 4x image)
            # Original: 384px -> Upscaled: 1536px.
            # If we tile at 384px again with overlap, we get ~16+ sub-tiles.
            # This allows "zooming in" effectively.
            
            sub_tiles = tile_image(
                sr_image, 
                tile_size=224, # Smaller clip size for local features? Or 384? SigLIP standard is 224/Varies.
                overlap=0.2 # Small overlap for speed
            )
            
            # 3. Batch Inference on Sub-Tiles
            batch_tensors = []
            valid_sub_tiles = []
            
            for st in sub_tiles:
                # Convert to PIL for Preprocess
                # (C, H, W) -> (H, W, C)
                tile_hwc = st.data.transpose(1, 2, 0)
                tile_hwc = (np.clip(tile_hwc, 0, 1) * 255).astype(np.uint8)
                pil_img = Image.fromarray(tile_hwc)
                
                # Preprocess (Resize, Normalize)
                tensor = self.preprocess(pil_img)
                batch_tensors.append(tensor)
                valid_sub_tiles.append(st)
                
            if not batch_tensors:
                continue
                
            input_batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(input_batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Similarity
                probs = (image_features @ text_features.T).flatten()
                
            # Store results
            for i, score in enumerate(probs.cpu().numpy()):
                results.append(RefinedResult(
                    original_tile_idx=idx,
                    sub_tile=valid_sub_tiles[i],
                    score=float(score)
                ))
                
        # Sort and Top-K
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

# Singleton (Lazy Load)
_refiner = None

def get_refiner():
    global _refiner
    if _refiner is None:
        _refiner = SemanticRefiner()
    return _refiner

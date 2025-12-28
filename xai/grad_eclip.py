"""
Grad-ECLIP Implementation

Provides gradient-based explainability for DOFA-CLIP models,
specifically adapted for Vision Transformers as specified in research.md.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import model_config

class GradECLIP:
    """
    Grad-ECLIP: Gradient-based Explanation for CLIP.
    
    Adapted for the custom DOFACLIPWrapper architecture.
    """
    
    def __init__(
        self,
        model_wrapper,
        target_layer_name: str = "trunk.norm"
    ):
        """
        Initialize Grad-ECLIP.
        
        Args:
            model_wrapper: Instance of DOFACLIPWrapper.
            target_layer_name: Dot-path to the target attention layer.
            Default 'trunk.norm' targets the final norm before pooling in OpenCLIP/Timm ViT.
        """
        self.wrapper = model_wrapper
        self.model = model_wrapper  # formatting alias
        self.target_layer_name = target_layer_name
        
        # Storage
        self._activations = None
        self._gradients = None
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        layer = self._get_layer(self.target_layer_name)
        if layer is None:
            # Fallback for old/custom models or different structures
            # Try 'visual.trunk.norm' if 'trunk.norm' failed on wrapper.vision_model
            # Note: _get_layer starts at self.wrapper usually?
            # self.wrapper.vision_model IS the starting point as per my previous code??
            # NO, _get_layer implementation uses `self.wrapper`.
            # self.wrapper.vision_model is `_model.visual`.
            # So `trunk.norm` should be `vision_model.trunk.norm`.
            # Let's adjust _get_layer usage or the path.
            # If I pass "vision_model.trunk.norm", that's safer.
            print(f"Warning: GradECLIP could not find layer {self.target_layer_name}. Trying explicit path...")
            layer = self._get_layer("vision_model.trunk.norm")
        
        if layer is None:
             print(f"Error: Could not find target layer for hooks.")
             return
             
        print(f"GradECLIP: Hooking into {layer}")
        
        # Hook for capturing activations (forward)
        h1 = layer.register_forward_hook(self._forward_hook)
        
        # Hook for capturing gradients (backward)
        h2 = layer.register_full_backward_hook(self._backward_hook)
        
        self.hooks.append(h1)
        self.hooks.append(h2)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _get_layer(self, layer_path: str):
        """Traverse the model to find the layer."""
        parts = layer_path.split('.')
        obj = self.wrapper
        
        for part in parts:
            if part.startswith('-'):
                try:
                    idx = int(part)
                    if hasattr(obj, '__getitem__'):
                        obj = obj[idx]
                    elif hasattr(obj, 'children'):
                        obj = list(obj.children())[idx]
                    else:
                        return None
                except:
                    return None
            else:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return None
        return obj

    def _forward_hook(self, module, input, output):
        """Capture activations (output of the layer)."""
        self._activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients w.r.t. output."""
        self._gradients = grad_output[0]

    def generate_similarity_map(
        self,
        image: torch.Tensor,
        text: str,
        image_size: int = 384
    ) -> np.ndarray:
        """
        Generate Token-Text Similarity Map (Non-Gradient).
        
        Computes the cosine similarity between the text embedding
        and each visual patch token.
        """
        # 1. Encode Text
        text_emb = self.wrapper.encode_text(text, normalize=True) # (1, D)
        
        # 2. Get Visual Tokens (Pre-Pooled)
        # Output: (1, N_tokens, D)
        # N_tokens is usually 14x14 = 196 (for SigLIP 384/14) or similar.
        visual_tokens = self.wrapper.get_visual_tokens(image, normalize=True)
        print(f"[DEBUG XAI] visual_tokens shape: {visual_tokens.shape}")
        
        # 3. Compute Similarity Map
        # (1, N, D) * (1, D) -> (1, N, D) -> sum(dim=-1) -> (1, N)
        # Efficient: Matmul (1, N, D) @ (1, D, 1) -> (1, N, 1)
        sim_map = torch.matmul(visual_tokens, text_emb.T).squeeze(-1) # (1, N)
        print(f"[DEBUG XAI] sim_map raw shape: {sim_map.shape}")
        
        # 4. Reshape to Grid
        num_tokens = sim_map.shape[1]
        
        # Check for CLS token? SigLIP usually doesn't have one (MAP/GAP).
        # OpenCLIP SigLIP uses Global Average Pooling usually, so tokens are pure spatial.
        # But if there IS a CLS token or Register, we need to handle it.
        # Simple check: is sqrt integer?
        
        grid_h = int(np.sqrt(num_tokens))
        if grid_h * grid_h != num_tokens:
             # Try removing one token (CLS)
             if int(np.sqrt(num_tokens - 1))**2 == num_tokens - 1:
                 # Has CLS, remove it (usually index 0? or last?)
                 # Standard ViT has CLS at 0
                 sim_map = sim_map[:, 1:]
                 grid_h = int(np.sqrt(num_tokens - 1))
             else:
                 print(f"[WARN] Unknown token count {num_tokens} for square grid.")
                 return np.zeros((image_size, image_size))
        
        # Reshape (1, H, W)
        sim_map = sim_map.reshape(1, 1, grid_h, grid_h)
        
        # 5. Normalize Map for Visualization
        # Clip negative correlations? Or just Min-Max?
        # User wants "contribution to match".
        # Typically similarity is [-1, 1].
        # We can map [-1, 1] to [0, 1] strictly? 
        # Or just min-max of the specific map to highlight RELATIVE importance?
        # Standard approach: ReLU (keep only positive) -> MinMax.
        # Or just MinMax.
        
        # Let's keep negatives as zero (irrelevant/anti-correlated)
        sim_map = F.relu(sim_map)
        
        # Min-Max
        map_min, map_max = sim_map.min(), sim_map.max()
        range_val = map_max - map_min
        
        # Ghost Pattern Fix:
        # If delta is tiny, we are just amplifying noise/bias.
        # Threshold: if range is < 0.01 (1% similarity difference), it's noise.
        
        if range_val > 0.01:
             sim_map = (sim_map - map_min) / range_val
        else:
             # Return empty/flat heatmap if no meaningful contrast
             return np.zeros((image_size, image_size))
        
        # 6. Upsample
        sim_map = F.interpolate(sim_map, size=(image_size, image_size), mode='bilinear', align_corners=False)
        
        return sim_map.squeeze().detach().cpu().numpy()

def generate_explanation(
    model_wrapper,
    image: np.ndarray,
    text: str,
    device: str = 'cpu',
    image_size: int = 384
) -> np.ndarray:
    """
    High-level explanation function.
    """
    # Prepare image tensor
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] <= 13: # (H, W, C)
            image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
        
    image = image.to(device or model_config.device)
    
    # Use the new Similarity Map method (Grad-free)
    # Re-using the GradECLIP class structure for convenience but could be standalone.
    explainer = GradECLIP(model_wrapper)
    
    # Note: No hooks needed anymore!
    heatmap = explainer.generate_similarity_map(image, text, image_size=image_size)
    
    return heatmap

def verify_explanation_perturbation(
    model_wrapper,
    image: np.ndarray,
    text: str,
    heatmap: np.ndarray,
    top_percentile: float = 20.0,
    device: str = 'cpu'
) -> Tuple[float, np.ndarray]:
    """
    Perform a perturbation test to verify explanation faithfulness.
    
    1. Identifying top X% active regions in the heatmap.
    2. Masking them in the original image (Occlusion).
    3. Measuring the drop in similarity score.
    
    Args:
        model_wrapper: Model instance.
        image: Original image array (H, W, 3) or (C, H, W).
        text: Query text.
        heatmap: Explanation heatmap (H, W).
        top_percentile: Percent of pixels to mask.
        
    Returns:
        (score_drop, masked_image_preview)
    """
    # Prepare image
    if isinstance(image, np.ndarray):
        # Ensure (C, H, W) for torch
        if image.ndim == 3 and image.shape[2] <= 13: # (H, W, C)
            proc_image = image.transpose(2, 0, 1)
        else:
            proc_image = image
        tensor_img = torch.from_numpy(proc_image).float()
    
    if tensor_img.dim() == 3:
        tensor_img = tensor_img.unsqueeze(0)
    
    tensor_img = tensor_img.to(device or model_config.device)
    
    # Check for RGB input (3 dims) vs S2 input (6 dims)
    # If 3 dims, we must provide RGB wavelengths to the model
    rgb_waves = None
    orig_waves = None
    
    if tensor_img.shape[1] == 3:
        # B4, B3, B2 for S2. (Red=665, Green=560, Blue=490)
        # Note: image from result_grid is likely RGB.
        rgb_waves = torch.tensor([665.0, 560.0, 490.0]).float().to(tensor_img.device)
        
        # We need to restore the original 6-band config later to not break search
        # Ideally we can read it, but set_wavelengths doesn't return it.
        # We know it's Sentinel-2 default.
        import config
        # CRITICAL FIX: Ensure we restore MICROMETERS, not Nanometers
        orig_waves = torch.tensor(config.sentinel2_bands.get_wavelength_tensor()).float().to(tensor_img.device) / 1000.0

    try:
        # Get original score
        # Pass wavelengths if RGB
        # DEBUG: Check input stats
        print(f"[DEBUG VERIFY] Input stats: Max={tensor_img.max():.6f}, Min={tensor_img.min():.6f}, Mean={tensor_img.mean():.6f}")
        
        orig_emb = model_wrapper.encode_image(tensor_img, wavelengths=rgb_waves, normalize=True)
        text_emb = model_wrapper.encode_text(text, normalize=True)
        orig_score = float((orig_emb @ text_emb.T).item())
        print(f"[DEBUG VERIFY] Orig Score: {orig_score:.6f}")
        
        # Create Mask from Heatmap
        # Threshold for top X percentile
        threshold = np.percentile(heatmap, 100 - top_percentile)
        mask = (heatmap >= threshold)
        print(f"[DEBUG VERIFY] Mask active pixels: {mask.sum()}/{mask.size} ({mask.mean()*100:.1f}%)")
        
        # Apply mask to image (set to Noise instead of Black)
        # Black (0.0) can be interpreted as shadows/water in remote sensing.
        # Gaussian noise is scientifically more robust for perturbation.
        torch_mask = torch.from_numpy(mask).to(tensor_img.device)
        masked_img = tensor_img.clone()
        
        # Generate noise matching the image statistics
        mean = tensor_img.mean()
        std = tensor_img.std()
        noise = torch.randn_like(tensor_img) * std + mean
        
        # Apply noise only to masked regions
        # We need to expand mask to (C, H, W)
        # torch_mask is (H, W) -> (1, 1, H, W) broadcast?
        # masked_img is (1, C, H, W)
        mask_expanded = torch_mask.unsqueeze(0).unsqueeze(0).expand_as(masked_img)
        
        masked_img[mask_expanded] = noise[mask_expanded]
        
        # Get perturbed score
        pert_emb = model_wrapper.encode_image(masked_img, wavelengths=rgb_waves, normalize=True)
        pert_score = float((pert_emb @ text_emb.T).item())
        print(f"[DEBUG VERIFY] Pert Score: {pert_score:.6f}")
        
        score_drop = orig_score - pert_score
        print(f"[DEBUG VERIFY] Drop: {score_drop:.6f}")
        
    finally:
        # Restore wavelengths if we changed them
        if orig_waves is not None:
             model_wrapper.model.visual.trunk.patch_embed.set_wavelengths(orig_waves)
    
    # Create preview of masked image for UI
    # Take first 3 channels for RGB preview
    if proc_image.shape[0] >= 3:
        rgb_preview = proc_image[:3].transpose(1, 2, 0).copy()
    else:
        rgb_preview = proc_image[0]
    
    # Apply mask visually (set to black/zero)
    rgb_preview[mask] = 0
    
    # Normalize for display if needed
    if rgb_preview.max() > 1.0:
        rgb_preview = rgb_preview / rgb_preview.max()
        
    return score_drop, (rgb_preview * 255).astype(np.uint8)

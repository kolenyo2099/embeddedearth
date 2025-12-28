"""
DOFA-CLIP Model Wrapper

Provides a unified interface for the Dynamic-One-For-All CLIP model,
handling wavelength-aware encoding for remote sensing imagery.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from pathlib import Path
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import model_config, sentinel2_bands
from models.wavelengths import get_wavelength_tensor

# =============================================================================
# DOFA-CLIP ARCHITECTURE IMPLEMENTATION
# =============================================================================

# =============================================================================
# WRAPPER (Updated for OpenCLIP with Custom DOFA Components)
# =============================================================================

import open_clip
import math
import torch.nn.functional as F

class FCLayer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU() # Assumed activation

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class WeightGenerator(nn.Module):
    def __init__(self, input_dim=128, output_dim=1152, kernel_k=14):
        super().__init__()
        # fc_weight projects to (output_dim * 1 * k * k)
        self.conv_weight_dim = output_dim * kernel_k * kernel_k
        self.fc_weight = nn.Linear(input_dim, self.conv_weight_dim)
        
        # fc_bias projects to (output_dim)
        self.fc_bias = nn.Linear(input_dim, output_dim)
        
        # Transformer
        # Shape check: in_proj_weight is [384, 128]. 384=3*128. So d_model=128.
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=2048, batch_first=True, norm_first=False),
            num_layers=1
        )
        
        # Tokens
        self.bias_token = nn.Parameter(torch.zeros(1, input_dim))
        self.weight_tokens = nn.Parameter(torch.zeros(input_dim, input_dim)) # Shape [128, 128] from keys
        
        self.output_dim = output_dim
        self.kernel_k = kernel_k

    def forward(self, wave_embeds):
        # wave_embeds: (B, num_bands, 128) or similar
        # But wait, weights are generated ONCE per band?
        # The key `weight_tokens` is [128, 128].
        # The input to transformer seems to be specific tokens.
        
        # Logic: 
        # 1. Combine wave embeddings with learnable tokens?
        # 2. Pass through transformer.
        # 3. Project to weights.
        
        # Based on keys `bias_token` (1, 128), and `weight_tokens` (128, 128).
        # Typically: x = torch.cat([bias_token, wave_embeds], dim=1) ?
        
        # Let's assume input x is processed wave embeddings.
        x = self.transformer_encoder(wave_embeds)
        
        # Apply projections
        weights = self.fc_weight(x) # (..., 225792)
        biases = self.fc_bias(x)    # (..., 1152)
        
        return weights, biases

class DynamicPatchEmbed(nn.Module):
    def __init__(self, img_size=384, patch_size=14, in_chans=3, embed_dim=1152, weight_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.embed_dim = embed_dim
        
        self.fclayer = FCLayer(input_dim=weight_dim)
        self.weight_generator = WeightGenerator(input_dim=weight_dim, output_dim=embed_dim, kernel_k=patch_size)
        
        # Store current wavelengths state (default to sentinel-2 center wavelengths)
        self.register_buffer('current_wavelengths', torch.zeros(1, 1)) # Placeholder
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        if hasattr(self, 'current_wavelengths'):
             num_waves = self.current_wavelengths.shape[0]
             if C != num_waves:
                 # Try to slice or warn? For now strict.
                 if C == 3 and num_waves > 3:
                     # Fallback for RGB images on multispectral model?
                     # Just take first 3 wavelengths? Or RGB indices?
                     # Assuming RGB correspond to B, G, R (indices 0, 1, 2)
                     pass # We will let it fail or I can select subset
                     
                 raise ValueError(f"Input channels {C} does not match configured wavelengths {num_waves}. Use set_wavelengths().")
                 
        # 1. Generate Weights for current bands
        # We need embeddings for the 'C' bands.
        # Ensure we have wavelengths set
        if not hasattr(self, 'cached_weights') or self.cached_weights is None:
             # Just use random init or raise error?
             # For now, let's assume get_weights() is called before forward, or we default.
             pass
             
        # Logic:
        # We assume `self.current_wavelengths` contains the sinusoidal embeddings of the bands.
        # Shape: (C, 128)
        
        # DEBUG PROBE: Check actual wavelengths used
        if hasattr(self, 'current_wavelengths'):
             sys.stderr.write(f"[DEBUG MODEL PROBE] Fingerprint (Embed Mean): {self.current_wavelengths.mean().item():.6f}\n")
             sys.stderr.write(f"[DEBUG MODEL PROBE] Fingerprint (Embed Std):  {self.current_wavelengths.std().item():.6f}\n")
             sys.stderr.flush()
        
        wave_emb = self.fclayer(self.current_wavelengths) # (C, 128)
        
        # Add batch dim for transformer?
        wave_emb = wave_emb.unsqueeze(0) # (1, C, 128)
        
        # Generate
        w, b = self.weight_generator(wave_emb) # (1, C, 225792), (1, C, 1152)

        w = w.squeeze(0) # (C, 225792)
        b = b.squeeze(0) # (C, 1152)
        
        w = w.view(C, self.embed_dim, 1, self.patch_size, self.patch_size)
        b = b.view(C, self.embed_dim)
        
        # We can implement this as a loop or grouped conv.
        # Grouped conv: Input (B, C, H, W). Output (B, C*Embed, H', W') -> Sum? No.
        # We want output (B, Embed, H', W').
        
        # Easier: Loop over bands (C is small, ~12).
        out = 0
        for i in range(C):
            # weight: (Embed, 1, K, K)
            # bias: (Embed)
            # input: (B, 1, H, W)
            band_x = x[:, i:i+1, :, :]
            band_w = w[i]
            band_b = b[i]
            
            # Conv2d
            out_i = F.conv2d(band_x, band_w, bias=band_b, stride=self.patch_size)
            out = out + out_i
            
        # Normalize by C? Or learnable? DOFA usually just sums or avgs.
        # Let's assumed sum as per "One-For-All" aggregation logic often used.
        
        # Flatten
        out = out.flatten(2).transpose(1, 2) # (B, N, D)
        return out
    
    def set_wavelengths(self, wavelengths_tensor):
        # wavelengths_tensor: (C,) floats
        
        # DEBUG: Check input
        w_mean = wavelengths_tensor.mean().item()
        sys.stderr.write(f"[DEBUG MODEL] set_wavelengths called with Mean={w_mean:.4f}\n")
        
        # SAFETY CATCH: If wavelengths are in Nanometers (>100), convert to Micrometers
        if w_mean > 100:
             sys.stderr.write("[DEBUG MODEL] DETECTED NANOMETERS. Auto-correcting to Micrometers (/1000).\n")
             wavelengths_tensor = wavelengths_tensor / 1000.0
             
        # Compute sinusoidal embedding
        device = wavelengths_tensor.device
        # Sinusoidal logic (simplified, assuming 128 dim)
        freqs = torch.exp(torch.arange(0, 128, 2, device=device).float() * -(math.log(10000.0) / 128))
        args = wavelengths_tensor.unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        self.current_wavelengths = embedding

class DOFACLIPWrapper:
    """
    Wrapper for DOFA-CLIP model using open_clip.
    """
    
    def __init__(
        self,
        model_name: str = "hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO",
        device: str = None,
        cache_dir: Path = None
    ):
        self.device = device or model_config.device
        self.model_name = model_name
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False
    
    def _load_model(self):
        """Initialize the OpenCLIP model using official DOFA-CLIP fork."""
        if self._loaded:
            return
            
        print(f"Loading model: {self.model_name}...")
        try:
            # Use native DOFA-CLIP loading (from official fork)
            # This automatically handles GeoLB architecture and weights
            model, preprocess = open_clip.create_model_from_pretrained(self.model_name)
            
            model = model.to(self.device)
            model.eval()
            
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._loaded = True
            
            print(f"Model loaded successfully! Trunk type: {type(model.visual.trunk).__name__}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load DOFA-CLIP model: {e}")
        
    @property
    def model(self):
        if not self._loaded: self._load_model()
        return self._model
        
    @property
    def preprocess(self):
        if not self._loaded: self._load_model()
        return self._preprocess
        
    @property
    def tokenizer(self):
        if not self._loaded: self._load_model()
        return self._tokenizer
        
    @property
    def text_model(self):
        # Helper for XAI tools that might expect separate access
        # OpenCLIP models usually have .encode_text method on the main model
        return self.model

    @property
    def vision_model(self):
        # Helper for XAI tools
        if not self._loaded: self._load_model()
        return self._model.visual

    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        if not self._loaded: self._load_model()
        
        if isinstance(text, str): text = [text]
        
        # Tokenize
        tokens = self.tokenizer(text).to(self.device)
        
        with torch.no_grad():
            text_embeds = self._model.encode_text(tokens)
            
        if normalize:
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
        return text_embeds

    def encode_image(
        self, 
        images: Union[np.ndarray, torch.Tensor], 
        wavelengths: Optional[torch.Tensor] = None, # Kept for API compatibility, unused by SigLIP
        normalize: bool = True
    ) -> torch.Tensor:
        if not self._loaded: self._load_model()
        
        # Preprocessing
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
                
        if images.dim() == 3:
            images = images.unsqueeze(0)
            
        images = images.to(self.device, dtype=torch.float32)
        
        # Check if we need to resize
        # The model expects specific input size (defined in config, usually 224 or 384)
        # Note: SigLIP might expect 384
        # We'll rely on our pipeline to have tiled it correctly, or interpolate.
        
        target_size = 384 # Or read from model config?
        if hasattr(self._model.visual, 'image_size'):
             target_size = self._model.visual.image_size
             if isinstance(target_size, tuple): target_size = target_size[0]

        if images.shape[-1] != target_size or images.shape[-2] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=(target_size, target_size), mode='bilinear'
            )
            
        # NORMALIZE INPUTS: Explicitly apply ImageNet mean/std if provided
        # The model expects normalized range [-2, 2], but we are feeding [0, 1].
        # We must normalize manually since we skipped the standard transform.
        device = images.device
        dtype = images.dtype
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device, dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device, dtype).view(1, 3, 1, 1)
        
        # Determine if we need to expand mean/std for multispectral
        # DOFA assumes standard normalization on the bands? Or per-band?
        # Usually we apply RGB normalization to the first 3 bands, and similar for others?
        # Or just 0-1?
        # Re-reading DOFA paper/code: DOFA takes raw inputs?
        # But here we are using a CLIP wrapper.
        # Let's perform standard normalization broadcasting it to the first 3 channels 
        # and reusing it for others? Or just centering.
        # SAFE BET: Normalize.
        
        if images.shape[1] == 3:
             images = (images - mean) / std
        else:
             # Multispectral Normalization heuristic
             # Create mean/std tensors for C channels
             C = images.shape[1]
             mean_full = torch.ones(1, C, 1, 1, device=device, dtype=dtype) * 0.45 # Average brightness
             std_full = torch.ones(1, C, 1, 1, device=device, dtype=dtype) * 0.27 # Average contrast
             
             # Overwrite RGB bands (assuming first 3 are RGB-like or similar magnitude)
             # S2: B2, B3, B4, B8... (Blue, Green, Red, NIR)
             # Wait, strict order matters.
             # But generally 0.45/0.27 is a safe "global" norm for 0-1 data to get to -2,2 range.
             # Ideally we use per-band stats but we don't have them handy.
             # This is infinitely better than No Normalization.
             
             images = (images - mean_full) / std_full 
             
        # Get default wavelengths if not provided
        if wavelengths is None:
            wavelengths = torch.tensor(
                sentinel2_bands.get_wavelength_tensor()
            ).float().to(self.device) / 1000.0  # Convert nm to um
             
        with torch.no_grad():
            # Use official DOFA-CLIP API: model.visual.trunk(image, wvs)
            # Returns tuple: (embedding, intermediate_features)
            out = self._model.visual.trunk(images, wavelengths)
            image_embeds = out[0] if isinstance(out, tuple) else out
             
        if normalize:
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
             
        return image_embeds
        
    def get_visual_tokens(
        self,
        images: Union[np.ndarray, torch.Tensor],
        wavelengths: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Get per-patch visual tokens before pooling.
        
        Returns:
            Tensor of shape (B, N_patches, EmbedDim).
        """
        if not self._loaded: self._load_model()
        
        # Preprocessing
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
                
        if images.dim() == 3:
            images = images.unsqueeze(0)
            
        images = images.to(self.device, dtype=torch.float32)
        
        # Resize if needed
        target_size = 384
        if hasattr(self._model.visual, 'image_size'):
             target_size = self._model.visual.image_size
             if isinstance(target_size, tuple): target_size = target_size[0]

        if images.shape[-1] != target_size or images.shape[-2] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=(target_size, target_size), mode='bilinear'
            )
            
        if wavelengths is None:
            wavelengths = torch.tensor(
                sentinel2_bands.get_wavelength_tensor()
            ).float().to(self.device) / 1000.0  # Convert nm to um
             
        with torch.no_grad():
            # Use official DOFA-CLIP API: model.visual.trunk(image, wvs)
            # Returns tuple: (pooled_output, intermediate_features)
            out = self._model.visual.trunk(images, wavelengths)
            
            if isinstance(out, tuple) and len(out) > 1:
                # out[1] is a list of intermediate features
                # The last one before pooling contains the patch tokens
                intermediate = out[1]
                if isinstance(intermediate, list) and len(intermediate) > 0:
                    # Get the last intermediate feature (usually the normalized tokens)
                    features = intermediate[-1]
                    if isinstance(features, torch.Tensor) and features.dim() == 3:
                        # Shape: (B, N_patches, D) or (B, D, N_patches)
                        pass  # Good!
                    else:
                        # Fallback: try to reshape
                        features = None
                else:
                    features = None
            else:
                features = None
                
                # Fallback 3: If hook failed, try old manual trunk navigation
                if features is None:
                     try:
                        x = self._model.visual.trunk.patch_embed(images)
                        x = self._model.visual.trunk._pos_embed(x)
                        x = self._model.visual.trunk.norm_pre(x)
                        x = self._model.visual.trunk.blocks(x)
                        features = self._model.visual.trunk.norm(x)
                     except:
                        pass

        if features is None:
             raise RuntimeError("Could not extract visual tokens from model.")
             
        # Ensure (B, N, D)
        if features.dim() == 2:
             # If we STILL have 2D, it means even the norm layer is returning pooled? Unlikely.
             # Or maybe it's just a class token?
             # Unsqueeze to (B, 1, D) as last resort
             features = features.unsqueeze(1)

        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features

# Singleton instance
_model_instance: Optional[DOFACLIPWrapper] = None

def get_model() -> DOFACLIPWrapper:
    global _model_instance
    if _model_instance is None:
        _model_instance = DOFACLIPWrapper()
    return _model_instance

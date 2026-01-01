"""
DINOv3 Model Wrapper

This module provides an interface to the DINOv3 model (specifically the satellite variant)
using the Hugging Face Transformers library.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import numpy as np
from typing import Union, Tuple, Optional
import os

class DINOv3Wrapper:
    """
    Wrapper for DINOv3 model to facilitate Zero-Shot Detection.
    """
    
    def __init__(
        self, 
        model_id: str = "facebook/dinov3-vitl16-pretrain-sat493m", 
        device: str = None, 
        token: str = None
    ):
        """
        Initialize DINOv3 Wrapper.
        
        Args:
            model_id: Hugging Face model ID.
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
            token: Hugging Face API Token.
        """
        self.model_id = model_id
        if device is None:
             if torch.cuda.is_available():
                 self.device = "cuda"
             elif torch.backends.mps.is_available():
                 self.device = "mps"
             else:
                 self.device = "cpu"
        else:
            self.device = device
            
        self.token = token
        self._model = None
        self._processor = None
        
    def _load_model(self):
        """Load model and processor if not already loaded."""
        if self._model is not None:
            return
            
        print(f"Loading DINOv3 model: {self.model_id}...")
        try:
            # 1. Try loading without token first (cache check or public model)
            # If it fails, retry with token.
            token_arg = self.token if self.token else None # Start with None
            
            try:
               # Attempt without token first if user didn't provide one
               # or checking cache
               self._processor = AutoImageProcessor.from_pretrained(
                   self.model_id, 
                   token=token_arg,
                   trust_remote_code=True,
                   local_files_only=False
               )
               self._model = AutoModel.from_pretrained(
                   self.model_id, 
                   token=token_arg,
                   trust_remote_code=True,
                   local_files_only=False,
                   attn_implementation="eager" # Required for output_attentions=True
               )
            except Exception as e:
                # If error is auth related and we have a token (or not), try specific logic
                if "401" in str(e) or "token" in str(e).lower() or "gated" in str(e).lower():
                    if not self.token:
                        # try environment variable HF_TOKEN
                        env_token = os.environ.get("HF_TOKEN")
                        if env_token:
                            self._processor = AutoImageProcessor.from_pretrained(self.model_id, token=env_token, trust_remote_code=True)
                            self._model = AutoModel.from_pretrained(
                                self.model_id, 
                                token=env_token, 
                                trust_remote_code=True,
                                attn_implementation="eager" # Required for output_attentions=True
                            )
                        else:
                             raise e
                    else:
                        raise e
                else:
                    raise e
            
            self._model.to(self.device)
            self._model.eval()
            print("✅ DINOv3 loaded successfully.")
            
        except Exception as e:
            if "401" in str(e) or "token" in str(e).lower():
                raise ValueError(
                    "Authentication failed. Please provide a valid Hugging Face Token "
                    "with access to the DINOv3 model."
                )
            raise e

    def _filter_features(
        self,
        features: torch.Tensor,
        h_model: int,
        w_model: int,
        patch_size: int
    ) -> torch.Tensor:
        """
        Helper to remove CLS/Register tokens and reshape to 2D grid.
        Returns: (1, N_patches, D) flat spatial tokens.
        """
        grid_h = h_model // patch_size
        grid_w = w_model // patch_size
        n_patches = grid_h * grid_w
        n_tokens = features.shape[1]
        
        if n_tokens > n_patches:
            # Assume [CLS, Reg1...RegK, Patch1...PatchN]
            # Debugging showed registers are at the start.
            n_extra = n_tokens - n_patches
            patch_tokens = features[:, n_extra:, :] 
        else:
            # Fallback or exact match (no CLS? unlikely for ViT-L)
            patch_tokens = features
            
        return patch_tokens

    def extract_features(
        self, 
        image: Union[np.ndarray, torch.Tensor], 
        resize: bool = True,
        center_features: bool = False
    ) -> torch.Tensor:
        """
        Extract dense features from an image.
        
        Args:
            image: Input image (H, W, C) numpy array or Tensor.
            resize: Whether to resize via processor.
            center_features: Whether to subtract mean feature (Robust Matching).
            
        Returns:
            Tensor of shape (1, N_patches, Embed_Dim)
        """
        self._load_model()
        
        # Preprocess
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get dimensions for reshaping logic
        pixel_values = inputs["pixel_values"]
        _, _, h_model, w_model = pixel_values.shape
        patch_size = 14
        if hasattr(self._model.config, 'patch_size'):
             patch_size = self._model.config.patch_size
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            
            # Filter and return only spatial tokens
            patch_tokens = self._filter_features(last_hidden_state, h_model, w_model, patch_size)
            
            if center_features:
                 # patch_tokens shape (1, N, D)
                 mean = patch_tokens.mean(dim=1, keepdim=True)
                 patch_tokens = patch_tokens - mean
                 
            return patch_tokens

    def get_patch_embedding(
        self, 
        image: np.ndarray, 
        bbox: list
    ) -> torch.Tensor:
        """
        Extract the average feature vector for a specific bounding box.
        
        Args:
            image: Original image extracted from (H, W, C).
            bbox: [x_min, y_min, width, height] in pixels relative to image.
            
        Returns:
            Query vector of shape (Embed_Dim,).
        """
        self._load_model()
        
        # 1. Process whole image to get feature map
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device) # (1, 3, H_in, W_in)
        
        # Get actual input dimensions after resizing
        # The processor usually resizes to a multiple of patch_size (e.g. 14 or 16)
        _, _, h_model, w_model = pixel_values.shape
        patch_size = 14 # Default for ViT-L/14, DINOv3 is typically 14
        if hasattr(self._model.config, 'patch_size'):
             patch_size = self._model.config.patch_size
        
        # Calculate scale factor between original image and model input
        h_orig, w_orig = image.shape[:2]
        scale_y = h_model / h_orig
        scale_x = w_model / w_orig
        
        # 2. Extract features
        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)
            # (1, N_tokens, D)
            features = outputs.last_hidden_state
        
        # Remove tokens that are not spatial patches using helper
        patch_tokens = self._filter_features(features, h_model, w_model, patch_size)
        
        # Centering (Robust Matching)
        # Subtract mean of this image
        mean = patch_tokens.mean(dim=1, keepdim=True)
        patch_tokens = patch_tokens - mean
            
        # Reshape
        try:
             # Calculate grid dimensions first
             grid_h = h_model // patch_size
             grid_w = w_model // patch_size
             patch_features = patch_tokens.reshape(1, grid_h, grid_w, -1)
        except Exception as e:
             # Fallback debug
             print(f"Shape error: Tokens {patch_tokens.shape}")
             raise e
        
        # 3. Map bbox to feature grid coords
        x, y, w, h = bbox
        
        # Scale to feature grid dimensions
        grid_x1 = int((x * scale_x) / patch_size)
        grid_y1 = int((y * scale_y) / patch_size)
        grid_x2 = int(((x + w) * scale_x) / patch_size)
        grid_y2 = int(((y + h) * scale_y) / patch_size)
        
        # Clamp
        grid_x1 = max(0, grid_x1)
        grid_y1 = max(0, grid_y1)
        grid_x2 = min(grid_w, max(grid_x1 + 1, grid_x2))
        grid_y2 = min(grid_h, max(grid_y1 + 1, grid_y2))
        
        # 4. ROI Pooling (Average)
        roi_features = patch_features[0, grid_y1:grid_y2, grid_x1:grid_x2, :]
        query_vector = roi_features.mean(dim=(0, 1)) # (D,)
        
        # Normalize
        query_vector = query_vector / query_vector.norm(p=2)
        
        return query_vector.cpu()

    def get_attention_map(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Get the self-attention map of the [CLS] token from the last layer.
        This often segments the foreground object natively.
        
        Args:
           image: (H, W, 3) numpy array (uint8).
           
        Returns:
           Attention map (H_img, W_img) float 0-1.
        """
        self._load_model()
        import torch.nn.functional as F
        
        # Preprocess
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get dimensions
        pixel_values = inputs["pixel_values"]
        _, _, h_model, w_model = pixel_values.shape
        patch_size = 14
        if hasattr(self._model.config, 'patch_size'):
             patch_size = self._model.config.patch_size
             
        grid_h = h_model // patch_size
        grid_w = w_model // patch_size
        
        with torch.no_grad():
            # Force config flag (sometimes kwarg is ignored by custom models)
            if hasattr(self._model, 'config'):
                self._model.config.output_attentions = True
                
            # Run with output_attentions=True (kwarg)
            outputs = self._model(**inputs, output_attentions=True)
            
            # Check if attentions are returned
            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                print("[WARN] DINOv3 model did not return attentions. Explainability unavailable.")
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
            # Get last layer attentions
            # Shape: (Batch, Heads, Tokens, Tokens)
            attentions = outputs.attentions[-1]
            
            # We want [CLS] token attending to [Patches]
            # Average over heads
            # Shape: (Batch, Tokens, Tokens)
            attn_avg = attentions.mean(dim=1)
            
            # Select Batch 0, CLS token (index 0), and all Patch tokens
            # Note: We need to handle registers/distillation tokens if present.
            # Using our helper logic concept, but reversed.
            # CLS is usually index 0.
            # But which indices are the patches?
            # They are likely 1 : 1+N_patches if standard/reg
            
            n_patches = grid_h * grid_w
            
            # Get the row corresponding to CLS
            # attn_cls shape: (Tokens,)
            attn_cls = attn_avg[0, 0, :]
            
            # Define count
            n_tokens = attn_cls.shape[0]
            
            # Extract Patch attention
            # Assume [CLS, Reg1...RegK, Patch1...PatchN]
            if n_tokens > n_patches:
                 n_extra = n_tokens - n_patches
                 patch_attn = attn_cls[n_extra:]
            else:
                 patch_attn = attn_cls
                 
            # Reshape
            patch_attn = patch_attn.reshape(grid_h, grid_w)
            
            # Interpolate to original image size
            # Needs 4D input (N, C, H, W)
            patch_attn = patch_attn.unsqueeze(0).unsqueeze(0)
            
            patch_attn = F.interpolate(
                patch_attn, 
                size=(image.shape[0], image.shape[1]), 
                mode='bicubic', 
                align_corners=False
            )
            
            # Normalize min-max for visualization
            patch_attn = patch_attn.squeeze()
            patch_attn = patch_attn - patch_attn.min()
            patch_attn = patch_attn / (patch_attn.max() + 1e-6)
            
            return patch_attn.cpu().numpy()

    def get_pca_map(
        self,
        image: np.ndarray,
        center_features: bool = True
    ) -> np.ndarray:
        """
        Compute PCA of local features to visualize semantic clusters.
        Returns: (H, W, 3) image in [0, 255].
        """
        import torch.nn.functional as F
        
        # 1. Extract raw features (N_patches, D)
        # We use our own extract_features logic but need keeping it simpler here 
        # to avoid double loading or circular deps, but calling existing method is best.
        features = self.extract_features(image, resize=True) # (1, N, D)
        features = features.squeeze(0) # (N, D)
        
        # 2. Centering (Crucial for DINO)
        # Subtract mean of this image's features
        if center_features:
            mean = features.mean(dim=0, keepdim=True)
            features = features - mean
            
        # 3. PCA via SVD
        # U, S, V = torch.pca_lowrank(features, q=3) # shape (N, 3) 
        # lowrank is stochastic, SVD is deterministic.
        # For N=196, full SVD is fast.
        try:
             # Full SVD on (N, D)
             U, S, V = torch.linalg.svd(features, full_matrices=False)
             # Project to top 3 components: X * V[:3].T
             # Or just use U * S for the projection in row space?
             # PCA scores = U * S
             coords = U[:, :3] @ torch.diag(S[:3])
        except Exception as e:
            print(f"PCA failed: {e}")
            return np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            
        # 4. Normalize min-max per channel to 0-1
        for i in range(3):
            c = coords[:, i]
            c_min = c.min()
            c_max = c.max()
            coords[:, i] = (c - c_min) / (c_max - c_min + 1e-6)
            
        # 5. Reshape to grid
        n_patches = features.shape[0]
        side = int(np.sqrt(n_patches))
        pca_map = coords.reshape(side, side, 3)
        
        # 6. Resize to original image
        # (N, C, H, W) for interpolate
        pca_map = pca_map.permute(2, 0, 1).unsqueeze(0) # (1, 3, S, S)
        
        pca_map = F.interpolate(
            pca_map, 
            size=(image.shape[0], image.shape[1]), 
            mode='bicubic', 
            align_corners=False
        )
        
        pca_map = pca_map.squeeze(0).permute(1, 2, 0) # (H, W, 3)
        
        return (pca_map.cpu().numpy() * 255).astype(np.uint8)

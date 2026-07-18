"""
DOFA-CLIP Model Wrapper

Provides a unified interface for the Dynamic-One-For-All CLIP model,
handling wavelength-aware encoding for remote sensing imagery.
"""

import torch
from typing import Optional, Union, List
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import model_config, sentinel2_bands

import open_clip


class DOFACLIPWrapper:
    """
    Wrapper for DOFA-CLIP model using open_clip.

    Loaded model: hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO
    Architecture: SigLIP-objective ViT-14 fine-tuned on GeoLangBind-2M with
    a wavelength-aware (DOFA) trunk. Accepts wavelength tensors in micrometers (μm).

    Input preprocessing:
    - Images must be in [0, 1] range before calling encode_image().
    - preprocess_tensor() applies SigLIP normalization (mean=0.5, std=0.5)
      which maps [0, 1] → [-1, 1].
    - Wavelengths must be in micrometers (μm). Pass nm values divided by 1000.
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        cache_dir: Path = None
    ):
        self.device = device or model_config.device
        # Use config as single source of truth for the model identifier
        self.model_name = model_name or model_config.model_name
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False

    def _load_model(self):
        """Initialize the OpenCLIP model."""
        if self._loaded:
            return

        print(f"Loading model: {self.model_name}...")
        try:
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
        return self.model

    @property
    def vision_model(self):
        if not self._loaded: self._load_model()
        return self._model.visual

    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        if not self._loaded: self._load_model()

        if isinstance(text, str): text = [text]

        tokens = self.tokenizer(text).to(self.device)

        with torch.no_grad():
            text_embeds = self._model.encode_text(tokens)

        if normalize:
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds

    def preprocess_tensor(
        self,
        images: Union[np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Centralized tensor preprocessing (Resize + Normalize).

        Args:
            images: Input images (numpy or tensor) in [0, 1] range.
                    Shape: (C, H, W) or (B, C, H, W) or (H, W, C).
            normalize: Whether to apply SigLIP channel normalization.

        Returns:
            Preprocessed tensor of shape (B, C, H, W) ready for model input.
        """
        if isinstance(images, np.ndarray):
            # Detect (H, W, C) layout when C is a plausible channel count (≤13)
            if images.ndim == 3 and images.shape[2] <= 13:
                images = images.transpose(2, 0, 1)
            images = torch.from_numpy(images)

        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device, dtype=torch.float32)

        # Resize to model's expected input size
        target_size = 384
        if hasattr(self._model.visual, 'image_size'):
            target_size = self._model.visual.image_size
            if isinstance(target_size, tuple): target_size = target_size[0]

        if images.shape[-1] != target_size or images.shape[-2] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=(target_size, target_size), mode='bilinear'
            )

        if normalize:
            device = images.device
            dtype = images.dtype
            C = images.shape[1]

            # SigLIP normalization: mean=0.5, std=0.5 maps [0,1] → [-1,1].
            # Applied uniformly across all channel counts (RGB or multispectral).
            mean = torch.ones(1, C, 1, 1, device=device, dtype=dtype) * 0.5
            std  = torch.ones(1, C, 1, 1, device=device, dtype=dtype) * 0.5
            images = (images - mean) / std

        return images

    def encode_image(
        self,
        images: Union[np.ndarray, torch.Tensor],
        wavelengths: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode image(s) to embedding vectors.

        Args:
            images: Array/tensor in [0, 1] range, shape (C, H, W) or (B, C, H, W).
            wavelengths: Band wavelengths in micrometers (μm). If None, defaults to
                         Sentinel-2 bands. To convert from nm: divide by 1000.
            normalize: L2-normalize output embeddings. Input (SigLIP) normalization
                       is always applied regardless of this flag.

        Returns:
            Tensor of shape (B, embedding_dim).
        """
        if not self._loaded: self._load_model()

        images = self.preprocess_tensor(images, normalize=True)

        # Default: Sentinel-2 wavelengths (converted to μm via helper).
        if wavelengths is None:
            from models.wavelengths import to_micrometers
            wavelengths = to_micrometers(
                torch.tensor(sentinel2_bands.get_wavelength_tensor())
                .float()
                .to(self.device)
            )

        with torch.no_grad():
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

        Args:
            images: Array/tensor in [0, 1] range.
            wavelengths: Band wavelengths in μm. Defaults to Sentinel-2.
            normalize: L2-normalize output tokens. Input (SigLIP) normalization
                       is always applied regardless of this flag.

        Returns:
            Tensor of shape (B, N_patches, EmbedDim).
        """
        if not self._loaded: self._load_model()

        images = self.preprocess_tensor(images, normalize=True)

        if wavelengths is None:
            from models.wavelengths import to_micrometers
            wavelengths = to_micrometers(
                torch.tensor(sentinel2_bands.get_wavelength_tensor())
                .float()
                .to(self.device)
            )

        with torch.no_grad():
            out = self._model.visual.trunk(images, wavelengths)

            if isinstance(out, tuple) and len(out) > 1:
                intermediate = out[1]
                if isinstance(intermediate, list) and len(intermediate) > 0:
                    features = intermediate[-1]
                else:
                    features = None
            else:
                features = None

        if features is None:
            raise RuntimeError("Could not extract visual tokens from model.")

        if features.dim() == 2:
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

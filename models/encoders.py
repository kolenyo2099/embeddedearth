"""
Text and Image Encoders Module

Provides standalone encoder classes for embedding generation,
using the DOFA-CLIP backbone.
"""

import torch
import numpy as np
from typing import List, Optional, Union

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import model_config
from models.dofa_clip import get_model, DOFACLIPWrapper
from models.wavelengths import get_wavelength_tensor


class TextEncoder:
    """
    Encodes text queries into embedding vectors.
    
    Wraps the DOFA-CLIP text encoder with a simple interface.
    """
    
    def __init__(self, model: DOFACLIPWrapper = None):
        """
        Initialize the text encoder.
        
        Args:
            model: Optional pre-loaded model instance.
        """
        self._model = model
    
    @property
    def model(self) -> DOFACLIPWrapper:
        """Get the underlying model."""
        if self._model is None:
            self._model = get_model()
        return self._model
    
    def encode(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Query string or list of strings.
            normalize: L2-normalize the output.
            
        Returns:
            Numpy array of shape (batch_size, 768).
        """
        embeddings = self.model.encode_text(text, normalize=normalize)
        return embeddings.cpu().numpy()
    
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """Shorthand for encode()."""
        return self.encode(text)


class ImageEncoder:
    """
    Encodes satellite imagery into embedding vectors.
    
    Handles wavelength-aware encoding for multispectral data.
    """
    
    def __init__(
        self,
        model: DOFACLIPWrapper = None,
        bands: List[str] = None
    ):
        """
        Initialize the image encoder.
        
        Args:
            model: Optional pre-loaded model instance.
            bands: Band names for wavelength lookup.
        """
        self._model = model
        self._bands = bands
        self._wavelengths = None
    
    @property
    def model(self) -> DOFACLIPWrapper:
        """Get the underlying model."""
        if self._model is None:
            self._model = get_model()
        return self._model
    
    @property
    def wavelengths(self) -> torch.Tensor:
        """Get wavelength tensor for current bands."""
        if self._wavelengths is None:
            self._wavelengths = get_wavelength_tensor(
                self._bands,
                device=self.model.device
            )
        return self._wavelengths
    
    def encode(
        self,
        images: Union[np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images to embedding vectors.
        
        Args:
            images: Array of shape (B, C, H, W) or (C, H, W).
            normalize: L2-normalize the output.
            
        Returns:
            Numpy array of shape (batch_size, 768).
        """
        embeddings = self.model.encode_image(
            images,
            wavelengths=self.wavelengths,
            normalize=normalize
        )
        return embeddings.cpu().numpy()
    
    def encode_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = None
    ) -> np.ndarray:
        """
        Encode a list of images in batches.
        
        Args:
            images: List of arrays, each (C, H, W).
            batch_size: Batch size for processing.
            
        Returns:
            Stacked embeddings of shape (N, 768).
        """
        batch_size = batch_size or model_config.batch_size
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_array = np.stack(batch, axis=0)
            embeddings = self.encode(batch_array)
            all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def __call__(
        self,
        images: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Shorthand for encode()."""
        return self.encode(images)


def create_encoders(
    model: DOFACLIPWrapper = None,
    bands: List[str] = None
) -> tuple:
    """
    Create text and image encoder pair.
    
    Args:
        model: Shared model instance.
        bands: Band configuration for image encoder.
        
    Returns:
        Tuple of (TextEncoder, ImageEncoder).
    """
    if model is None:
        model = get_model()
    
    text_encoder = TextEncoder(model)
    image_encoder = ImageEncoder(model, bands)
    
    return text_encoder, image_encoder

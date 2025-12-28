"""
Wavelength Configuration Module

Defines Sentinel-2 band wavelengths for DOFA-CLIP's 
wavelength-aware spectral encoding.
"""

from typing import Dict, List
import torch

# Sentinel-2 MSI Band Wavelengths (nanometers)
# Reference: ESA Sentinel-2 User Handbook

SENTINEL2_WAVELENGTHS: Dict[str, int] = {
    # Visible
    'B01': 443,   # Coastal aerosol
    'B02': 490,   # Blue
    'B03': 560,   # Green
    'B04': 665,   # Red
    
    # Red Edge
    'B05': 705,   # Vegetation Red Edge 1
    'B06': 740,   # Vegetation Red Edge 2
    'B07': 783,   # Vegetation Red Edge 3
    
    # Near-Infrared
    'B08': 842,   # NIR
    'B8A': 865,   # Vegetation Red Edge 4
    
    # Short-Wave Infrared
    'B09': 945,   # Water Vapour
    'B10': 1375,  # SWIR - Cirrus
    'B11': 1610,  # SWIR 1
    'B12': 2190,  # SWIR 2
}

# Default bands for DOFA-CLIP (6-band configuration)
DEFAULT_BANDS: List[str] = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']


def get_wavelengths_for_bands(bands: List[str] = None) -> List[int]:
    """
    Get wavelength values for specified bands.
    
    Args:
        bands: List of band names. Default: DEFAULT_BANDS.
        
    Returns:
        List of wavelengths in nanometers.
    """
    bands = bands or DEFAULT_BANDS
    return [SENTINEL2_WAVELENGTHS[b] for b in bands]


def get_wavelength_tensor(
    bands: List[str] = None,
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Get wavelengths as a PyTorch tensor.
    
    Args:
        bands: List of band names.
        dtype: Tensor data type.
        device: Target device.
        
    Returns:
        Tensor of shape (num_bands,).
    """
    wavelengths = get_wavelengths_for_bands(bands)
    return torch.tensor(wavelengths, dtype=dtype, device=device)


def normalize_wavelengths(
    wavelengths: torch.Tensor,
    min_wl: float = 400.0,
    max_wl: float = 2500.0
) -> torch.Tensor:
    """
    Normalize wavelengths to [0, 1] range for model input.
    
    Args:
        wavelengths: Tensor of wavelengths in nm.
        min_wl: Minimum wavelength for normalization.
        max_wl: Maximum wavelength for normalization.
        
    Returns:
        Normalized wavelength tensor.
    """
    return (wavelengths - min_wl) / (max_wl - min_wl)

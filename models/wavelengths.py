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

    # Non-zero-padded aliases used by GEE band names (B2, B3, ...) and config.py
    'B1':  443,
    'B2':  490,
    'B3':  560,
    'B4':  665,
    'B5':  705,
    'B6':  740,
    'B7':  783,
    'B8':  842,
    'B9':  945,
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
    Get wavelengths as a PyTorch tensor, in nanometers (nm).

    Call-site conversion to micrometers happens via `to_micrometers` before
    the tensor is passed into the DOFA trunk.

    Args:
        bands: List of band names.
        dtype: Tensor data type.
        device: Target device.

    Returns:
        Tensor of shape (num_bands,) in nm.
    """
    wavelengths = get_wavelengths_for_bands(bands)
    return torch.tensor(wavelengths, dtype=dtype, device=device)


def to_micrometers(wavelengths: torch.Tensor) -> torch.Tensor:
    """
    Convert a wavelength tensor to micrometers (μm).

    Optical Sentinel-2 values are stored as nm (~400–2500). SAR C-band is
    stored directly as μm (~55500). The heuristic below keeps both working
    as the single source of truth for unit conversion used across the app.

    - Any value already > 10_000 is assumed μm (SAR) and left untouched.
    - Any value between 100 and 10_000 is assumed nm and divided by 1000.
    - Any value < 100 is assumed μm and left untouched.

    WARNING: the heuristic cannot represent wavelengths above 10 μm expressed
    in nm (e.g. thermal IR ~10_900 nm would be misread as μm SAR). If thermal
    bands are ever added, store their config values in μm directly.
    """
    wl_max = float(wavelengths.max().item())
    if wl_max > 10_000.0:
        return wavelengths  # SAR μm already
    if wl_max >= 100.0:
        return wavelengths / 1000.0  # nm → μm
    return wavelengths  # already μm

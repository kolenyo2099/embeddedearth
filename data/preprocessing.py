"""
Data Preprocessing Module

Handles conversion of GEE imagery to numpy arrays and normalization
for DOFA-CLIP model input.
"""

import ee
import numpy as np
import requests
from PIL import Image
from rasterio.io import MemoryFile
import logging

# Suppress annoying rasterio/GDAL warnings about photometric interpretation
logging.getLogger('rasterio').setLevel(logging.ERROR)

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import sentinel2_bands, model_config
from data.gee_client import GEEClient


def download_image_as_array(
    image: ee.Image,
    aoi: ee.Geometry,
    bands: list = None,
    scale: int = 10
) -> np.ndarray:
    """
    Download an Earth Engine image as a numpy array.
    
    Args:
        image: ee.Image to download.
        aoi: Geometry defining the region.
        bands: List of band names. Default: Sentinel-2 bands for DOFA.
        scale: Resolution in meters.
        
    Returns:
        numpy array of shape (C, H, W).
    """
    GEEClient.initialize()
    
    bands = bands or sentinel2_bands.band_names
    
    # Get download URL with fallback for size limits
    try:
        url = image.getDownloadUrl({
            'region': aoi,
            'scale': scale,
            'format': 'GEO_TIFF',
            'bands': bands,
        })
    except Exception as e:
        if "Total request size" in str(e) or "User memory limit exceeded" in str(e):
            new_scale = scale * 2
            if new_scale > 60: # Limit recursion
                raise e
            print(f"[WARN] GEE download limit reached. Retrying with scale={new_scale}m...")
            return download_image_as_array(image, aoi, bands, new_scale)
        raise e
    
    # Download the image
    response = requests.get(url)
    response.raise_for_status()
    
    # Read with rasterio
    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            # Read all bands (shape: bands, height, width)
            data = dataset.read()
            
    return data.astype(np.float32)


def normalize_reflectance(
    data: np.ndarray,
    scale_factor: float = None
) -> np.ndarray:
    """
    Normalize Sentinel-2 reflectance values to 0-1 range.
    
    Args:
        data: Array of shape (C, H, W) with raw reflectance.
        scale_factor: Division factor. Default from config.
        
    Returns:
        Normalized array in range [0, 1].
    """
    scale = scale_factor or sentinel2_bands.scale_factor
    normalized = data / scale
    
    # Clip to valid range
    normalized = np.clip(normalized, 0, 1)
    
    return normalized


def prepare_for_model(
    data: np.ndarray,
    target_size: int = None
) -> np.ndarray:
    """
    Prepare image data for DOFA-CLIP model input.
    
    Resizes to model input size and ensures correct format.
    
    Args:
        data: Normalized array of shape (C, H, W).
        target_size: Target size for height and width.
        
    Returns:
        Array of shape (C, target_size, target_size).
    """
    target = target_size or model_config.image_size
    
    C, H, W = data.shape
    
    if H == target and W == target:
        return data
    
    # Resize using PIL for each channel
    resized_channels = []
    for c in range(C):
        img = Image.fromarray(data[c])
        img_resized = img.resize((target, target), Image.BILINEAR)
        resized_channels.append(np.array(img_resized))
    
    return np.stack(resized_channels, axis=0)


def get_rgb_visualization(
    data: np.ndarray,
    bands: list = None,
    brightness_factor: float = 2.5
) -> np.ndarray:
    """
    Extract RGB bands for visualization.
    
    Args:
        data: Array of shape (C, H, W).
        bands: Band names in order. Default from config.
        brightness_factor: Multiplier for visibility.
        
    Returns:
        RGB array of shape (H, W, 3) in uint8 format.
    """
    bands = bands or sentinel2_bands.band_names
    
    # If there are fewer than 3 channels (e.g., Sentinel-1 VV/VH), create a safe pseudo-RGB.
    c = data.shape[0]
    if c == 1:
        gray = np.clip(data[0] * brightness_factor, 0, 1)
        rgb = np.stack([gray, gray, gray], axis=-1)
        return (rgb * 255).astype(np.uint8)
    if c == 2:
        ch0 = np.clip(data[0] * brightness_factor, 0, 1)
        ch1 = np.clip(data[1] * brightness_factor, 0, 1)
        rgb = np.stack([ch0, ch1, ch0], axis=-1)
        return (rgb * 255).astype(np.uint8)

    # Find RGB band indices (B4=Red, B3=Green, B2=Blue), accepting both
    # GEE-style (B4) and zero-padded (B04) band names
    def _find_band(candidates):
        for name in candidates:
            if name in bands:
                return bands.index(name)
        return None

    r_idx = _find_band(['B4', 'B04'])
    g_idx = _find_band(['B3', 'B03'])
    b_idx = _find_band(['B2', 'B02'])

    if r_idx is None or g_idx is None or b_idx is None:
        # Fallback to first three bands
        r_idx, g_idx, b_idx = 0, 1, 2
    
    # Extract and stack RGB
    rgb = np.stack([
        data[r_idx],
        data[g_idx],
        data[b_idx]
    ], axis=-1)
    
    # Apply brightness and clip
    rgb = rgb * brightness_factor
    rgb = np.clip(rgb, 0, 1)

    # Convert to uint8
    return (rgb * 255).astype(np.uint8)

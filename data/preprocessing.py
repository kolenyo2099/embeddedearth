"""
Data Preprocessing Module

Handles conversion of GEE imagery to numpy arrays and normalization
for DOFA-CLIP model input.
"""

import ee
import numpy as np
import requests
import io
from typing import Tuple, Optional, Union
from PIL import Image
import rasterio
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
    
    # Find RGB band indices (B4=Red, B3=Green, B2=Blue)
    try:
        r_idx = bands.index('B4')
        g_idx = bands.index('B3')
        b_idx = bands.index('B2')
    except ValueError:
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


def get_wavelength_tensor() -> np.ndarray:
    """
    Get wavelength values as numpy array for DOFA-CLIP.
    
    Returns:
        Array of wavelengths in nanometers.
    """
    return np.array(sentinel2_bands.get_wavelength_tensor(), dtype=np.float32)


class ImageProcessor:
    """
    Complete image processing pipeline for converting
    GEE imagery to model-ready tensors.
    """
    
    def __init__(self):
        """Initialize processor with configuration."""
        self._bands = sentinel2_bands.band_names
        self._scale_factor = sentinel2_bands.scale_factor
        self._target_size = model_config.image_size
    
    def process(
        self,
        image: ee.Image,
        aoi: ee.Geometry,
        scale: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full processing pipeline.
        
        Args:
            image: GEE image to process.
            aoi: Area of interest geometry.
            scale: Resolution in meters.
            
        Returns:
            Tuple of (processed_array, wavelengths).
        """
        # Download
        raw = download_image_as_array(image, aoi, self._bands, scale)
        
        # Normalize
        normalized = normalize_reflectance(raw, self._scale_factor)
        
        # Resize
        resized = prepare_for_model(normalized, self._target_size)
        
        # Get wavelengths
        wavelengths = get_wavelength_tensor()
        
        return resized, wavelengths
    
    def get_visualization(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Get RGB visualization of processed data.
        
        Args:
            data: Processed array of shape (C, H, W).
            
        Returns:
            RGB array for display.
        """
        return get_rgb_visualization(data, self._bands)

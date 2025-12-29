"""EmbeddedEarth Configuration Module

Centralized configuration for the entire application.
All settings, paths, and constants are defined here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models_cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# =============================================================================
# GOOGLE EARTH ENGINE CONFIGURATION
# =============================================================================

@dataclass
class GEEConfig:
    """Configuration for Google Earth Engine."""
    
    # GEE Project ID (set via environment variable or directly)
    # If not set, GEE might try to use default credentials which is fine for local dev
    # GEE Project ID (set via environment variable or directly)
    # If not set, GEE might try to use default credentials which is fine for local dev
    project_id: str = os.getenv("GEE_PROJECT_ID")
    
    # Sentinel-2 Collection
    s2_collection: str = "COPERNICUS/S2_SR_HARMONIZED"
    
    # Cloud Score+ Collection for masking
    cloud_score_collection: str = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    
    # Cloud masking threshold (0-1, lower = stricter)
    cloud_threshold: float = 0.65
    
    # Maximum cloud cover percentage for initial filtering
    max_cloud_cover: int = 30
    
    # Default date range (days before today)
    default_days_back: int = 90


# =============================================================================
# SENTINEL-2 BAND CONFIGURATION
# =============================================================================

@dataclass
class Sentinel2Bands:
    """Sentinel-2 band configuration for DOFA-CLIP."""
    
    # Bands to use (in order)
    band_names: List[str] = field(default_factory=lambda: [
        'B2', 'B3', 'B4', 'B8', 'B11', 'B12'
    ])
    
    # Corresponding wavelengths in nanometers (for DOFA-CLIP)
    wavelengths: Dict[str, int] = field(default_factory=lambda: {
        'B2': 490,   # Blue
        'B3': 560,   # Green
        'B4': 665,   # Red
        'B8': 842,   # NIR
        'B11': 1610, # SWIR1
        'B12': 2190, # SWIR2
    })
    
    # Scale factor for reflectance normalization
    scale_factor: float = 10000.0
    
    def get_wavelength_tensor(self) -> List[int]:
        """Get wavelengths as ordered list for model input."""
        return [self.wavelengths[b] for b in self.band_names]


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for DOFA-CLIP model."""
    
    # Model identifier (for Hugging Face or local path)
    model_name: str = "XShadow/DOFA-CLIP"
    
    # Vision Transformer settings
    image_size: int = 384
    patch_size: int = 14
    
    # Embedding dimension
    embedding_dim: int = 1152
    
    # Device configuration
    device: str = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
    
    # Batch size for inference
    batch_size: int = 32
    
    # Cache directory for model weights
    cache_dir: Path = MODELS_DIR


# =============================================================================
# TILING CONFIGURATION
# =============================================================================

@dataclass
class TilingConfig:
    """Configuration for image tiling."""
    
    # Tile size in pixels (must match model input)
    tile_size: int = 384
    
    # Overlap ratio (0.5 = 50% overlap as per research.md)
    overlap_ratio: float = 0.5
    
    # Stride = tile_size * (1 - overlap_ratio)
    @property
    def stride(self) -> int:
        return int(self.tile_size * (1 - self.overlap_ratio))
    
    # Maximum AOI size in degrees (to prevent excessive processing)
    max_aoi_degrees: float = 0.5  # ~50km at equator


# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for vector similarity search."""
    
    # Number of top results to return
    top_k: int = 10
    
    # Similarity threshold (0-1)
    # Default lowered to 0.1 to account for potentially low cosine scores in raw models
    similarity_threshold: float = 0.1
    
    # Use approximate search for large indexes
    use_approximate: bool = False
    
    # FAISS index type
    index_type: str = "IndexFlatIP"  # Inner product for normalized vectors


# =============================================================================
# UI CONFIGURATION
# =============================================================================

@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""
    
    # Page settings
    page_title: str = "EmbeddedEarth"
    page_icon: str = "🌍"
    layout: str = "wide"
    
    # Map settings
    default_center: tuple = (40.0, -3.7)  # Spain
    default_zoom: int = 6
    
    # Result display
    results_per_row: int = 3
    show_heatmaps: bool = True


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

# Create default config instances
gee_config = GEEConfig()
sentinel2_bands = Sentinel2Bands()
model_config = ModelConfig()
tiling_config = TilingConfig()
search_config = SearchConfig()
ui_config = UIConfig()

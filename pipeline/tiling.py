"""
Image Tiling Module

Implements the sliding window approach with 50% overlap
for processing large satellite imagery.
"""

import numpy as np
from typing import List, Tuple, Iterator, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import tiling_config


@dataclass
class Tile:
    """Represents a single tile extracted from an image."""
    
    # Pixel coordinates in source image
    x: int
    y: int
    
    # Tile dimensions
    width: int
    height: int
    
    # Image data (C, H, W)
    data: np.ndarray
    
    # Geospatial bounds (minx, miny, maxx, maxy)
    bounds: Optional[Tuple[float, float, float, float]] = None
    
    # Affine transform for this specific tile
    transform: Optional[object] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get tile center in pixel coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class TileGenerator:
    """
    Generates overlapping tiles from large images.
    
    Uses a sliding window approach with configurable overlap.
    """
    
    def __init__(
        self,
        tile_size: int = None,
        overlap_ratio: float = None
    ):
        """
        Initialize the tile generator.
        
        Args:
            tile_size: Size of each tile in pixels.
            overlap_ratio: Overlap between tiles (0.5 = 50%).
        """
        self.tile_size = tile_size or tiling_config.tile_size
        self.overlap_ratio = overlap_ratio or tiling_config.overlap_ratio
        self.stride = int(self.tile_size * (1 - self.overlap_ratio))
    
    def generate(
        self,
        image: np.ndarray,
        transform: object = None,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Iterator[Tile]:
        """
        Generate tiles from an image.
        
        Args:
            image: Array of shape (C, H, W).
            transform: Rasterio-like affine transform (optional but recommended for georeferencing).
            bounds: Optional total bounds (minx, miny, maxx, maxy).
            
        Yields:
            Tile objects.
        """
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        
        C, H, W = image.shape
        
        # Calculate number of tiles in each dimension
        n_tiles_y = max(1, (H - self.tile_size) // self.stride + 1)
        n_tiles_x = max(1, (W - self.tile_size) // self.stride + 1)
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Calculate pixel coordinates
                y = ty * self.stride
                x = tx * self.stride
                
                # Handle edge cases (ensure we don't go out of bounds)
                if y + self.tile_size > H:
                    y = H - self.tile_size
                if x + self.tile_size > W:
                    x = W - self.tile_size
                
                # Skip if we'd get negative coordinates (small image)
                y = max(0, int(y))
                x = max(0, int(x))
                
                # Slice logic
                h_slice = min(self.tile_size, H - y)
                w_slice = min(self.tile_size, W - x)
                
                # Extract tile data
                tile_data = image[:, y:y+h_slice, x:x+w_slice]
                
                # Pad if strictly required to be square (usually handled by model resizing, but good safety)
                if tile_data.shape[1] != self.tile_size or tile_data.shape[2] != self.tile_size:
                    # For now, we rely on the model wrapper's interpolate/resizing
                    pass
                
                # Calculate geospatial bounds for tile
                tile_bounds = None
                tile_transform = None
                
                if transform:
                    # Calculate new transform for this tile -> translate origin
                    # Affine(a, b, c, d, e, f)
                    # c' = c + a*x + b*y
                    # f' = f + d*x + e*y
                    # Assuming standard north-up raster: a>0, e<0, b=d=0 usually
                    
                    # Manual calculation if transform allows it (rasterio Affine object)
                    try:
                        tile_transform = transform * transform.translation(x, y)
                        
                        # Calculate bounds from transform
                        # (0,0) -> (w, h) in tile specific coords
                        minx, maxy = tile_transform * (0, 0)
                        maxx, miny = tile_transform * (w_slice, h_slice)
                        tile_bounds = (minx, miny, maxx, maxy)
                    except:
                        pass
                
                elif bounds:
                    # Fallback to linear interpolation of bounds if no transform
                    b_minx, b_miny, b_maxx, b_maxy = bounds
                    px_to_geo_x = (b_maxx - b_minx) / W
                    px_to_geo_y = (b_maxy - b_miny) / H
                    
                    tile_bounds = (
                        b_minx + x * px_to_geo_x,
                        b_miny + y * px_to_geo_y,
                        b_minx + (x + w_slice) * px_to_geo_x,
                        b_miny + (y + h_slice) * px_to_geo_y
                    )
                
                yield Tile(
                    x=x,
                    y=y,
                    width=w_slice,
                    height=h_slice,
                    data=tile_data,
                    bounds=tile_bounds,
                    transform=tile_transform
                )

def tile_image(
    image: np.ndarray,
    tile_size: int = None,
    overlap: float = None,
    transform: object = None,
    bounds: tuple = None
) -> List[Tile]:
    """
    Convenience function to tile an image.
    
    Args:
        image: Array of shape (C, H, W).
        tile_size: Size of each tile.
        overlap: Overlap ratio.
        transform: Affine transform.
        bounds: Geospatial bounds.
        
    Returns:
        List of Tile objects.
    """
    generator = TileGenerator(tile_size, overlap)
    return list(generator.generate(image, transform=transform, bounds=bounds))


def reconstruct_from_tiles(
    tiles: List[Tile],
    output_shape: Tuple[int, int, int],
    reduce: str = 'mean'
) -> np.ndarray:
    """
    Reconstruct an image from overlapping tiles.
    
    Useful for creating heatmaps from per-tile scores.
    
    Args:
        tiles: List of Tile objects.
        output_shape: Target shape (C, H, W).
        reduce: How to handle overlaps ('mean', 'max', 'first').
        
    Returns:
        Reconstructed array.
    """
    C, H, W = output_shape
    
    if reduce == 'mean':
        output = np.zeros((C, H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)
        
        for tile in tiles:
            y, x = tile.y, tile.x
            h, w = tile.height, tile.width
            
            output[:, y:y+h, x:x+w] += tile.data
            counts[y:y+h, x:x+w] += 1
        
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        output = output / counts
        
    elif reduce == 'max':
        output = np.full((C, H, W), -np.inf, dtype=np.float32)
        
        for tile in tiles:
            y, x = tile.y, tile.x
            h, w = tile.height, tile.width
            output[:, y:y+h, x:x+w] = np.maximum(
                output[:, y:y+h, x:x+w],
                tile.data
            )
    
    else:  # 'first'
        output = np.zeros((C, H, W), dtype=np.float32)
        filled = np.zeros((H, W), dtype=bool)
        
        for tile in tiles:
            y, x = tile.y, tile.x
            h, w = tile.height, tile.width
            
            mask = ~filled[y:y+h, x:x+w]
            for c in range(C):
                output[c, y:y+h, x:x+w] = np.where(
                    mask,
                    tile.data[c],
                    output[c, y:y+h, x:x+w]
                )
            filled[y:y+h, x:x+w] = True
    
    return output

def generate_geo_grid(
    bounds: Tuple[float, float, float, float],
    resolution: float = 10.0,
    tile_size: int = 384,
    overlap_ratio: float = 0.5
) -> Iterator[Tuple[Tuple[float, float, float, float], int, int]]:
    """
    Generate geospatial bounds for overlapping tiles.
    
    Args:
        bounds: Total AOI bounds (minx, miny, maxx, maxy) in degrees.
        resolution: Resolution in meters per pixel (default 10m for Sentinel-2).
        tile_size: Tile size in pixels (default 384).
        overlap_ratio: Overlap (default 0.5).
        
    Yields:
        Tuple of (tile_bounds, col_idx, row_idx).
        tile_bounds is (minx, miny, maxx, maxy).
    """
    minx, miny, maxx, maxy = bounds
    
    # Calculate tile size in meters
    tile_size_m = tile_size * resolution
    stride_m = tile_size_m * (1 - overlap_ratio)
    
    # Degrees per meter (approximate)
    # Latitude: 1 deg ~= 111,320 meters
    deg_per_m_lat = 1 / 111320.0
    
    current_y = maxy
    row = 0
    
    while current_y > miny:
        # Calculate current latitude for longitude correction
        # Use center of current row
        center_lat = current_y - (tile_size_m * deg_per_m_lat) / 2
        
        # Longitude: 1 deg ~= 111,320 * cos(lat) meters
        # Clamp cos to avoid division by zero (though unlikely for valid maps)
        cos_lat = np.cos(np.radians(center_lat))
        deg_per_m_lon = 1 / (111320.0 * max(0.1, abs(cos_lat)))
        
        # Steps in degrees
        tile_h_deg = tile_size_m * deg_per_m_lat
        tile_w_deg = tile_size_m * deg_per_m_lon
        stride_h_deg = stride_m * deg_per_m_lat
        stride_w_deg = stride_m * deg_per_m_lon
        
        current_x = minx
        col = 0
        
        while current_x < maxx:
            # Define tile bounds
            # Top-Left origin for generation loop, but bounds are (minx, miny, maxx, maxy)
            t_minx = current_x
            t_maxy = current_y
            t_maxx = current_x + tile_w_deg
            t_miny = current_y - tile_h_deg
            
            yield (t_minx, t_miny, t_maxx, t_maxy), col, row
            
            current_x += stride_w_deg
            col += 1
            
        current_y -= stride_h_deg
        row += 1

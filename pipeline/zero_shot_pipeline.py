"""
Zero-Shot Detection Pipeline
Uses DINOv3 for query-based object detection in satellite imagery.
"""

import numpy as np
import torch
import logging
from datetime import datetime
from PIL import Image

# Import existing utilities
from pipeline.tiling import generate_geo_grid, Tile
from data.preprocessing import download_image_as_array
from models.dinov3 import DINOv3Wrapper

def run_zero_shot_pipeline(
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    query_vector: torch.Tensor,
    threshold: float = 0.5,
    resolution: int = 10,
    hf_token: str = None,
    progress_callback=None,
    status_callback=None,
) -> list:
    """
    Execute Zero-Shot Detection on an AOI.
    
    Args:
        aoi_geojson: Target area geometry.
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        query_vector: (Embed_Dim,) tensor from reference patch.
        threshold: Similarity threshold (0.0 to 1.0).
        resolution: Meters per pixel.
        hf_token: Hugging Face token.
        
    Returns:
        List of detections [{'geometry': ..., 'score': ...}]
    """
    import ee
    
    # 1. Initialize GEE
    from data.gee_client import GEEClient
    if not GEEClient.is_initialized():
        GEEClient.initialize()
        
    logger = logging.getLogger(__name__)

    def log_status(message: str) -> None:
        if status_callback:
            status_callback(message)
        logger.info(message)

    def update_progress(value: float) -> None:
        if progress_callback:
            progress_callback(value)

    # 2. Initialize Model
    try:
        model = DINOv3Wrapper(token=hf_token)
        # Pre-load to fail fast if token invalid
        model._load_model()
    except Exception as e:
        logger.exception("Model initialization failed.")
        return []

    # 3. Fetch Imagery (Target Area)
    log_status("🛰️ Fetching target Sentinel-2 imagery...")
    from data.sentinel2 import Sentinel2Retriever
    retriever = Sentinel2Retriever()
    
    if aoi_geojson.get('type') == 'Polygon':
        aoi_ee = ee.Geometry.Polygon(aoi_geojson['coordinates'])
    else:
        aoi_ee = ee.Geometry(aoi_geojson)
        
    # CHECK DATA AVAILABILITY
    # Before tiling, ensure we actually have images for this whole area/time.
    try:
        col_check = retriever.get_collection(aoi_ee, start_date, end_date)
        count = col_check.size().getInfo()
        if count == 0:
            logger.warning(
                "No Sentinel-2 imagery found for this area between %s and %s.",
                start_date,
                end_date,
            )
            return []
        print(f"[DEBUG] Found {count} Sentinel-2 scenes for the AOI.")
    except Exception as e:
        print(f"[DEBUG] Collection check failed: {e}")
        # Proceed cautiously? Or stop?
        # If check failed (e.g. auth), subsequent steps will fail too.
        logger.exception("Failed to query Earth Engine.")
        return []
        
    # Get Bounds
    bounds_info = aoi_ee.bounds().getInfo()['coordinates'][0]
    west = min(p[0] for p in bounds_info)
    south = min(p[1] for p in bounds_info)
    east = max(p[0] for p in bounds_info)
    north = max(p[1] for p in bounds_info)
    bounds = (west, south, east, north)
    
    # Generate Tiles
    # Fixed tile size for inference (e.g. 512x512)
    # DINOv3 patch size is 14. 512 is not multiple of 14 (36.5). 
    # 518 is 14*37. 
    # Let's use 224 or 448 (14*32). 448 is good balance.
    TILE_SIZE = 448 
    
    # 3. Generate Grid
    # We use a sliding window over the AOI
    # Tiling strategy: Generate geospatial bounds for each tile, fetch, process.
    grid = list(generate_geo_grid(bounds, resolution, tile_size=TILE_SIZE))
    total_tiles = len(grid)
    print(f"Generated {len(grid)} tiles.")   
    if total_tiles > 200:
        logger.warning("Processing %s tiles. This may take time.", total_tiles)
    
    detections = []
    
    update_progress(0.0)
    log_status(f"Starting analysis of {total_tiles} tiles...")
    
    # Prepare Query Vector
    query_vector = query_vector.to(model.device)
    query_norm = query_vector / query_vector.norm()
    
    for i, (t_bounds, col, row) in enumerate(grid):
        update_progress((i + 1) / total_tiles)
        log_status(f"Processing Tile {i+1}/{total_tiles}...")
        
        # Geometry
        t_minx, t_miny, t_maxx, t_maxy = t_bounds
        tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])
        
        # Download
        # Using retriever to get normalize composite
        # IMPORTANT: We should verify if the composite actually has data.
        # But for speed, we assume the tile is valid if it's within the AOI.
        # However, GEE might return empty if no images intersect this specific small tile.
        
        try:
             # Create composite for just this tile
             tile_comp = retriever.get_composite(tile_geom, start_date, end_date)
             tile_comp = retriever.normalize_for_model(tile_comp)
             
             # DOWNLOAD OPTIMIZATION: Only fetch RGB bands (B4, B3, B2)
             rgb_bands = ['B4', 'B3', 'B2']
             arr = download_image_as_array(tile_comp, tile_geom, bands=rgb_bands, scale=resolution)
        except Exception as e:
             print(f"[DEBUG] Tile {i} download failed: {str(e)[:100]}...") # Log first 100 chars
             continue
        
        if arr.max() == 0: 
             print(f"[DEBUG] Tile {i} is empty (max=0). Skipping.")
             continue
        
        # Preprocess
        # download_image_as_array returns (C, H, W) = (3, H, W)
        # Hugging Face ImageProcessor usually expects (H, W, C) for numpy arrays.
        
        try:
            # Transpose: (C, H, W) -> (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
                
            # Convert to uint8 0-255 for Processor if currently float 0-1
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = np.clip(arr, 0, 1)
                arr_uint8 = (arr * 255).astype(np.uint8)
            else:
                arr_uint8 = arr
                
            # Extract Features
            # shape: (1, N_patches, D)
            # CRITICAL FIX: Enable center_features to match query vector centering
            features = model.extract_features(arr_uint8, center_features=True) 
            features = features.squeeze(0) # (N_patches, D)
        except Exception as e:
            print(f"[DEBUG] Tile {i} Feature Extraction Error: {e}")
            continue
        
        # Calculate Similarity
        # (N, D) @ (D, 1) -> (N, 1)
        # Normalize features
        feats_norm = features / features.norm(dim=1, keepdim=True)
        
        sim_scores = (feats_norm @ query_norm).cpu().numpy() # (N,)
        
        # Map back to spatial map
        # CRITICAL FIX: Dynamically determine grid dimension instead of assuming fixed size
        grid_dim = int(np.sqrt(len(sim_scores)))
        
        if grid_dim * grid_dim != len(sim_scores):
            # Fallback for non-square results if any (though usually square in the processor)
            logger.warning("Feature count %s is not a perfect square.", len(sim_scores))
            sim_map = sim_scores.reshape(1, -1) # Flattened fallback
        else:
            sim_map = sim_scores.reshape(grid_dim, grid_dim)
            
        # Thresholding
        # Finding connected components or just points
        # For simple version: Store tiles having max score > threshold
        
        # Thresholding
        max_score = sim_map.max()
        if max_score > threshold:
            # Store Result
            # Convert map to heatmap image
            # Replace cv2 with skimage to avoid dependency issues
            import skimage.transform
            
            # skimage resize expects (H, W)
            # It returns float 0-1
            heatmap_resized = skimage.transform.resize(
                sim_map, 
                (arr_uint8.shape[0], arr_uint8.shape[1]), 
                order=3, # Cubic
                mode='reflect', 
                anti_aliasing=True
            )
            
            # Create masked overlay
            detections.append({
                'image': arr_uint8,
                'heatmap': heatmap_resized,
                'score': float(max_score),
                'bounds': t_bounds,
                'dino_attention': model.get_attention_map(arr_uint8), # Add native DINO attention
                'pca_map': model.get_pca_map(arr_uint8, center_features=True) # Add PCA visualization
            })
            
    update_progress(1.0)
    
    # Sort by score
    detections.sort(key=lambda x: x['score'], reverse=True)
    
    return detections

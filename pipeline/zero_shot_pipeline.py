"""
Zero-Shot Detection Pipeline
Uses DINOv3 for query-based object detection in satellite imagery.
"""

import numpy as np
import torch
import streamlit as st

# Import existing utilities
from pipeline.tiling import generate_geo_grid
from data.preprocessing import download_image_as_array
from models.dinov3 import DINOv3Wrapper

def run_zero_shot_pipeline(
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    query_vector: torch.Tensor,
    sensor: str = "Sentinel-2",
    threshold: float = 0.5,
    resolution: int = 10,
    hf_token: str = None
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
        
    # 2. Initialize Model
    try:
        model = DINOv3Wrapper(token=hf_token)
        # Pre-load to fail fast if token invalid
        model._load_model()
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        return []

    # 3. Fetch Imagery (Target Area)
    st.info(f"🛰️ Fetching target {sensor} imagery...")
    if sensor == "Sentinel-1":
        from data.sentinel1 import Sentinel1Retriever
        retriever = Sentinel1Retriever()
        bands = ['VV', 'VH']
    else:
        from data.sentinel2 import Sentinel2Retriever
        retriever = Sentinel2Retriever()
        bands = ['B4', 'B3', 'B2']
    
    if aoi_geojson.get('type') == 'Polygon':
        aoi_ee = ee.Geometry.Polygon(aoi_geojson['coordinates'])
    else:
        aoi_ee = ee.Geometry(aoi_geojson)
        
    # Build ONE composite for the whole AOI up front. get_composite() also
    # guards against empty collections, so this doubles as the availability
    # check — without paying a blocking GEE round trip per tile.
    try:
        composite = retriever.get_composite(aoi_ee, start_date, end_date)
        composite = retriever.normalize_for_model(composite)
    except ValueError as e:
        st.error(
            f"❌ No {sensor} imagery found for this area between {start_date} and {end_date}. "
            f"Try increasing the date range or relaxing filters. ({e})"
        )
        return []
    except Exception as e:
        st.error(f"Failed to query Earth Engine: {e}")
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
        st.warning(f"Processing {total_tiles} tiles. This may take time.")
    
    detections = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Starting analysis of {total_tiles} tiles...")

    # Prepare Query Vector
    query_vector = query_vector.to(model.device)
    query_norm = query_vector / query_vector.norm()

    # Downloads are network-bound: run them in parallel against the single
    # shared composite. Feature extraction stays on the main thread.
    import concurrent.futures

    def download_tile(args):
        i, t_bounds = args
        t_minx, t_miny, t_maxx, t_maxy = t_bounds
        tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])
        try:
            arr = download_image_as_array(composite, tile_geom, bands=bands, scale=resolution)
        except Exception as e:
            print(f"[DEBUG] Tile {i} download failed: {str(e)[:100]}...")
            return i, None
        if arr.max() == 0:
            return i, None
        return i, arr

    MAX_WORKERS = 12
    download_args = [(i, t_bounds) for i, (t_bounds, col, row) in enumerate(grid)]
    tile_bounds_by_idx = {i: t_bounds for i, (t_bounds, col, row) in enumerate(grid)}

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, arr in executor.map(download_tile, download_args):
            completed += 1
            progress_bar.progress(completed / total_tiles)
            status_text.text(f"Processing Tile {completed}/{total_tiles}...")

            if arr is None:
                continue

            t_bounds = tile_bounds_by_idx[i]

            # Preprocess
            # download_image_as_array returns (C, H, W) = (3, H, W)
            # Hugging Face ImageProcessor usually expects (H, W, C) for numpy arrays.
            try:
                # Transpose: (C, H, W) -> (H, W, C)
                arr = np.transpose(arr, (1, 2, 0))

                # DINO expects RGB-like input. Expand Sentinel-1 (VV/VH) to 3 channels.
                if sensor == "Sentinel-1" and arr.shape[2] == 2:
                    arr = np.stack([arr[:, :, 0], arr[:, :, 1], arr[:, :, 0]], axis=-1)

                # Convert to uint8 0-255 for Processor if currently float 0-1
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = np.clip(arr, 0, 1)
                    arr_uint8 = (arr * 255).astype(np.uint8)
                else:
                    arr_uint8 = arr

                # Extract Features
                # shape: (1, N_patches, D)
                # center_features matches the query vector centering
                features = model.extract_features(arr_uint8, center_features=True)
                features = features.squeeze(0) # (N_patches, D)
            except Exception as e:
                print(f"[DEBUG] Tile {i} Feature Extraction Error: {e}")
                continue

            # Calculate Similarity: (N, D) @ (D,) -> (N,)
            feats_norm = features / features.norm(dim=1, keepdim=True)
            sim_scores = (feats_norm @ query_norm).cpu().numpy() # (N,)

            # Map back to spatial map, deriving the grid size dynamically
            grid_dim = int(np.sqrt(len(sim_scores)))

            if grid_dim * grid_dim != len(sim_scores):
                # Fallback for non-square results if any (though usually square in the processor)
                st.warning(f"Feature count {len(sim_scores)} is not a perfect square.")
                sim_map = sim_scores.reshape(1, -1) # Flattened fallback
            else:
                sim_map = sim_scores.reshape(grid_dim, grid_dim)

            # Thresholding: store tiles whose best patch beats the threshold
            max_score = sim_map.max()
            if max_score > threshold:
                # Convert map to heatmap image (skimage instead of cv2)
                import skimage.transform

                # skimage resize expects (H, W); returns float 0-1
                heatmap_resized = skimage.transform.resize(
                    sim_map,
                    (arr_uint8.shape[0], arr_uint8.shape[1]),
                    order=3, # Cubic
                    mode='reflect',
                    anti_aliasing=True
                )

                detections.append({
                    'image': arr_uint8,
                    'heatmap': heatmap_resized,
                    'score': float(max_score),
                    'bounds': t_bounds,
                })

    status_text.empty()
    progress_bar.empty()

    # Suppress overlapping duplicates (grid has 50% overlap), sort by score
    from pipeline.postprocessing import nms_results
    detections = nms_results(detections)

    # Extra visualizations (each costs a full model forward) only for the
    # detections that survived NMS.
    for det in detections:
        det['dino_attention'] = model.get_attention_map(det['image'])   # Native DINO attention
        det['pca_map'] = model.get_pca_map(det['image'], center_features=True)  # PCA visualization

    return detections

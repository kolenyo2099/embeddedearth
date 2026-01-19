import ee
import torch
import numpy as np
import io
from datetime import datetime
from typing import List, Dict, Union, Optional, Callable
import logging

from data.sentinel2 import Sentinel2Retriever
from data.sentinel1 import Sentinel1Retriever
from data.preprocessing import download_image_as_array, normalize_reflectance, get_rgb_visualization
from models.copernicus_fm import CopernicusFM
from config import sentinel2_bands, sentinel1_bands, model_config
from pipeline.tiling import generate_geo_grid

class CopernicusSearchPipeline:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = CopernicusFM(device=device)
        self.s2_retriever = Sentinel2Retriever()
        self.s1_retriever = Sentinel1Retriever()

    def _get_meta_info(self, 
                       lon: float, 
                       lat: float, 
                       date: datetime, 
                       resolution: float, 
                       patch_size: int = 16) -> torch.Tensor:
        """
        Construct meta_info tensor [lon, lat, time, area]
        
        Args:
            lon, lat: Center coordinates
            date: Acquisition date
            resolution: Meters per pixel
            patch_size: Patch size in pixels
            
        Returns:
            Tensor of shape [1, 4]
        """
        # Time: Day of year (0-365)
        day_of_year = date.timetuple().tm_yday
        
        # Area: Patch area in km2
        # resolution is in meters. patch_size usually 16.
        # side_km = (resolution * patch_size) / 1000.0
        # area_km2 = side_km * side_km
        side_km = (resolution * patch_size) / 1000.0
        area_km2 = side_km * side_km
        
        meta = torch.tensor([lon, lat, day_of_year, area_km2], dtype=torch.float32)
        return meta.unsqueeze(0) # [1, 4]

    def _prepare_input(self, 
                       image_array: np.ndarray, 
                       bbox: tuple, 
                       date: datetime,
                       sensor: str,
                       resolution: float) -> dict:
        """
        Convert numpy array to model inputs.
        """
        # Image shape: (C, H, W)
        # Normalize to 0-1 if not already (Sentinel-1 is already 0-1 from retriever)
        # Sentinel-2 from download_image_as_array is raw, so normalize.
        
        if sensor == "Sentinel-2":
            # download_image_as_array returns raw reflectance
            image_array = normalize_reflectance(image_array)
            bands_config = sentinel2_bands
        else: # Sentinel-1
            # Sentinel-1 retriever returns 0-1 (dB scaled)
            # But download_image_as_array might just download raw pixels if used directly on the image?
            # Wait, download_image_as_array calls getDownloadUrl on the image passed.
            # If the image passed is already normalized (which we do in pipeline), then it's 0-1.
            # We must ensure we pass the normalized image to download_image_as_array.
            bands_config = sentinel1_bands
            
        # Resize to model size (e.g. 224, or keep original if creating tiles)
        # For simplicity, we assume fixed size or handle in model (which has interpolation)
        # CopernicusFM handles arbitrary sizes via flexible interpolation?
        # model_vit.py uses `resize_abs_pos_embed`.
        # So we can pass (C, H, W).
        
        import torch.nn.functional as F
        
        # Create tensor [1, C, H, W]
        x = torch.from_numpy(image_array).float().unsqueeze(0) 
        
        # Resize to fixed 224x224 to ensure compatibility with ViT patch embedding (16x16)
        # This handles cases where the input is smaller than 16x16 (Kernel Size Error)
        # or inconsistent with the expected input dimensions.
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Meta info
        minx, miny, maxx, maxy = bbox
        lon = (minx + maxx) / 2
        lat = (miny + maxy) / 2
        meta_info = self._get_meta_info(lon, lat, date, resolution)
        
        # Wavelengths and Bandwidths
        if sensor == "Sentinel-2":
            wvs = bands_config.get_wavelength_tensor() # nm
            bws = bands_config.get_bandwidth_list()    # nm
        else: # Sentinel-1
            # S1 config has microns (55500). Model expects nm.
            # Convert um -> nm (x1000)
            wvs = [w * 1000 for w in bands_config.get_wavelength_tensor()]
            bws = [b * 1000 for b in bands_config.get_bandwidth_list()]
            
        return {
            "x": x,
            "meta_info": meta_info,
            "wavelengths": wvs,
            "bandwidths": bws
        }

    def run_search(self, 
                   query_geom: Dict, 
                   search_geom: Dict, 
                   start_date: str, 
                   end_date: str, 
                   sensor: str = "Sentinel-2",
                   resolution: int = 10,
                   threshold: float = 0.5,
                   progress_callback: Optional[Callable[[str], None]] = None) -> List[Dict]:
        """
        Run similarity search.
        
        Args:
            query_geom: GeoJSON geometry for query
            search_geom: GeoJSON geometry for search area
            start_date, end_date: ISO strings
            sensor: "Sentinel-2" or "Sentinel-1"
            resolution: meters per pixel
        """
        # Initialize GEE
        from data.gee_client import GEEClient
        GEEClient.initialize()
        
        # 1. Fetch Query Image
        # --------------------
        query_ee_geom = ee.Geometry(query_geom)
        retriever = self.s2_retriever if sensor == "Sentinel-2" else self.s1_retriever
        bands_config = sentinel2_bands if sensor == "Sentinel-2" else sentinel1_bands
        
        # Get composite
        query_comp = retriever.get_composite(query_ee_geom, start_date, end_date)
        if sensor == "Sentinel-1":
            query_comp = retriever.normalize_for_model(query_comp)
        
        # Check if empty (simple check via fetching small info?)
        # Just try downloading
        try:
            # Scale should match model expectations ideally, but we pass resolution.
            # Use 224x224 tile logic or just download bounds?
            # For query, we download the bbox of the geometry.
            query_arr = download_image_as_array(
                query_comp, 
                query_ee_geom, 
                bands=bands_config.band_names, 
                scale=resolution
            )
            # Check for valid data
            if np.max(query_arr) == 0:
                print("Query image is empty.")
                return []
                
        except Exception as e:
            print(f"Failed to download query image: {e}")
            return []

        # Get Query Embedding
        # -------------------
        bbox = query_ee_geom.bounds().getInfo()['coordinates'][0] # [[minx, miny], ...]
        minx = min(p[0] for p in bbox)
        maxx = max(p[0] for p in bbox)
        miny = min(p[1] for p in bbox)
        maxy = max(p[1] for p in bbox)
        
        # Date for meta info (use midpoint of range?)
        # Or extract from image metadata (hard with composite).
        # Use start_date as proxy.
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        
        q_inputs = self._prepare_input(
            query_arr, 
            (minx, miny, maxx, maxy), 
            date_obj, 
            sensor, 
            resolution
        )
        
        self.model.model.to(self.device)
        query_emb = self.model.forward(
            q_inputs['x'].to(self.device),
            q_inputs['meta_info'].to(self.device),
            q_inputs['wavelengths'],
            q_inputs['bandwidths']
        )
        
        # Flatten and normalize query embedding
        # Output shape [B, D] or [B, N, D]? 
        # forward returns feature vector (pooled) if not return_intermediate.
        # vit_base usually returns [B, embed_dim] if global pool is True.
        # CopernicusFM uses global pool by default (see wrapper).
        # So [1, 1024]
        
        query_vec = query_emb.detach().cpu()
        query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
        
        # 2. Process Search Area
        # ----------------------
        search_ee_geom = ee.Geometry(search_geom)
        search_rect = search_ee_geom.bounds().getInfo()['coordinates'][0]
        s_west = min(p[0] for p in search_rect)
        s_south = min(p[1] for p in search_rect)
        s_east = max(p[0] for p in search_rect)
        s_north = max(p[1] for p in search_rect)
        
        # Generate grid
        # Use simple fixed size matching query? Or 224x224 patches?
        # Let's use 224px tiles ~ 2.24km at 10m.
        tile_size_px = 224
        
        tiles = list(generate_geo_grid((s_west, s_south, s_east, s_north), resolution, tile_size_px))
        
        results = []
        
        if progress_callback:
            progress_callback(f"Processing {len(tiles)} tiles...")
            
        for i, (t_bounds, col, row) in enumerate(tiles):
            if progress_callback and i % 5 == 0:
                progress_callback(f"Processing tile {i+1}/{len(tiles)}...")
                
            # t_bounds: (minx, miny, maxx, maxy)
            t_geom = ee.Geometry.Rectangle(list(t_bounds))
            
            # Fetch tile
            try:
                # Reuse retriever (cached?? No, GEE is lazy)
                # But creating composite for each tile is slow.
                # Ideally create one composite for AOI and clip.
                # But download_image_as_array takes image.
                # So we can pass the same composite image, just different region.
                
                # Check intersection with search_geom to capture irregular shapes
                if not t_geom.intersects(search_ee_geom).getInfo():
                    continue
                    
                tile_arr = download_image_as_array(
                    query_comp, # Reuse composite!
                    t_geom,
                    bands=bands_config.band_names,
                    scale=resolution
                )
                
                if np.max(tile_arr) == 0:
                    continue
                    
                # Prepare Inputs
                t_inputs = self._prepare_input(
                    tile_arr,
                    t_bounds,
                    date_obj,
                    sensor,
                    resolution
                )
                
                # Inference
                t_emb = self.model.forward(
                    t_inputs['x'].to(self.device),
                    t_inputs['meta_info'].to(self.device),
                    t_inputs['wavelengths'],
                    t_inputs['bandwidths']
                )
                
                t_vec = t_emb.detach().cpu()
                t_vec = t_vec / t_vec.norm(dim=-1, keepdim=True)
                
                # Similarity
                score = torch.mm(query_vec, t_vec.T).item()
                
                # Filter by threshold (Replicating DINO pipeline logic)
                if score < threshold:
                    continue
                
                # Generate visualization image for UI
                display_img = None
                try:
                    norm_arr = normalize_reflectance(tile_arr)
                    if sensor == "Sentinel-2":
                        display_img = get_rgb_visualization(norm_arr, bands=bands_config.band_names)
                    else:
                        # Fallback for Sentinel-1 (2 bands) or others
                        # Simple grayscale using first band
                        band0 = np.clip(norm_arr[0], 0, 1)
                        gray = (band0 * 255).astype(np.uint8)
                        display_img = np.stack([gray, gray, gray], axis=-1)
                except Exception as e:
                    print(f"Vis generation failed: {e}")

                results.append({
                    'image': display_img,
                    'geometry': t_geom.getInfo(), #GeoJSON
                    'score': score,
                    'bounds': t_bounds
                })
                
            except Exception as e:
                import traceback
                print(f"Tile {i} error: {e}")
                # traceback.print_exc()
                continue
                
        # Sort results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

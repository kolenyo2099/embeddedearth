"""Semantic search service for the EmbeddedEarth API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from typing import Optional, Callable, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchParams:
    query: str
    start_date: Optional[date]
    end_date: Optional[date]
    top_k: int = 10
    similarity_threshold: float = 0.3
    resolution: float = 10.0
    search_type: str = "text"
    reference_image: Optional[bytes] = None


def run_semantic_search(
    aoi_geojson: dict,
    params: SemanticSearchParams,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """
    Execute the full semantic search pipeline with real satellite imagery.

    Args:
        aoi_geojson: GeoJSON geometry dict from map drawing.
        params: Search parameters from UI.
        progress_callback: Optional callback for progress updates.
        status_callback: Optional callback for status messages.

    Returns:
        List of result dicts with 'image', 'heatmap', 'score', 'bounds'.
    """
    import ee

    def log_status(message: str) -> None:
        if status_callback:
            status_callback(message)
        logger.info(message)

    def update_progress(value: float) -> None:
        if progress_callback:
            progress_callback(value)

    logger.info("[SEARCH] Starting search pipeline")
    logger.debug("[SEARCH] AOI GeoJSON: %s", aoi_geojson)
    logger.info("[SEARCH] Query: %s", params.query)

    results = []

    try:
        # Step 1: Initialize GEE if needed
        from data.gee_client import GEEClient
        if not GEEClient.is_initialized():
            log_status("🔐 Initializing Google Earth Engine...")
            GEEClient.initialize()

        # Step 2: Convert AOI GeoJSON to EE Geometry
        log_status("📍 Processing area of interest...")

        if aoi_geojson.get("type") == "Polygon":
            aoi_ee = ee.Geometry.Polygon(aoi_geojson["coordinates"])
        else:
            aoi_ee = ee.Geometry(aoi_geojson)

        logger.debug("[SEARCH] EE Geometry created: %s", aoi_ee.getInfo())

        # Step 3: Fetch Sentinel-2 imagery
        log_status("🛰️ Fetching Sentinel-2 imagery from Google Earth Engine...")

        from data.sentinel2 import Sentinel2Retriever
        retriever = Sentinel2Retriever()

        # Date range from params
        start_date = params.start_date.strftime("%Y-%m-%d") if params.start_date else None
        end_date = params.end_date.strftime("%Y-%m-%d") if params.end_date else None

        logger.info("[SEARCH] Date range: %s to %s", start_date, end_date)

        composite = retriever.get_composite(aoi_ee, start_date, end_date)
        composite = retriever.normalize_for_model(composite)

        # Step 4: Tile-First Strategy
        log_status("🗺️ Generating search grid...")

        from pipeline.tiling import generate_geo_grid, Tile
        from data.preprocessing import download_image_as_array, get_rgb_visualization
        from models.encoders import create_encoders

        # Get bounds
        bounds_info = aoi_ee.bounds().getInfo()["coordinates"][0]
        west = min(p[0] for p in bounds_info)
        south = min(p[1] for p in bounds_info)
        east = max(p[0] for p in bounds_info)
        north = max(p[1] for p in bounds_info)
        bounds = (west, south, east, north)

        # Generate grid with dynamic resolution (Smart Scaling)
        target_resolution = params.resolution

        # Estimate degrees width/height
        deg_width = east - west
        deg_height = north - south

        # Approx meters (at equator, simplistic but safe for estimation)
        meters_width = deg_width * 111320
        meters_height = deg_height * 111320

        # Tile size in meters at target res
        tile_m = 384 * target_resolution
        stride_m = tile_m * 0.5

        # Estimated tiles (Width / Stride) * (Height / Stride)
        est_cols = max(1, meters_width / stride_m)
        est_rows = max(1, meters_height / stride_m)
        total_est_tiles = est_cols * est_rows

        if total_est_tiles > 5000:
            logger.warning(
                "High-Resolution Search: Generating %s tiles. This might take a while!",
                int(total_est_tiles),
            )

        if total_est_tiles > 25000:
            raise ValueError(
                f"Too many tiles ({int(total_est_tiles)}). Reduce the area or increase resolution."
            )

        grid_tiles = list(generate_geo_grid(bounds, resolution=target_resolution))
        logger.info(
            "Created grid with %s tiles (Resolution: %sm/px).",
            len(grid_tiles),
            target_resolution,
        )

        if len(grid_tiles) > 20000:
            raise ValueError("Area is too big! Please select a smaller region.")

        # Initialize models once
        from models import dofa_clip
        import importlib

        importlib.reload(dofa_clip)
        dofa_clip._model_instance = None
        logger.info("[SEARCH] Forcing model reload")

        text_encoder, image_encoder = create_encoders()
        query_embedding = text_encoder.encode(params.query)

        logger.debug(
            "[SEARCH] Text embedding stats: min=%s, max=%s, mean=%s, norm=%s",
            query_embedding.min(),
            query_embedding.max(),
            query_embedding.mean(),
            np.linalg.norm(query_embedding),
        )

        def process_tile_task(args):
            idx, t_bounds, col, row = args
            try:
                t_minx, t_miny, t_maxx, t_maxy = t_bounds
                tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])

                tile_composite = retriever.get_composite(tile_geom, start_date, end_date)
                tile_composite = retriever.normalize_for_model(tile_composite)

                tile_data = download_image_as_array(tile_composite, tile_geom, scale=target_resolution)

                if tile_data.max() == 0:
                    return None

                from data.preprocessing import prepare_for_model
                tile_data = prepare_for_model(tile_data)

                d_min, d_max, d_mean = tile_data.min(), tile_data.max(), tile_data.mean()
                if idx < 5:
                    logger.debug(
                        "[DATA] Tile %s stats: shape=%s, min=%s, max=%s, mean=%s",
                        idx,
                        tile_data.shape,
                        d_min,
                        d_max,
                        d_mean,
                    )

                if d_max == 0:
                    return None

                emb = image_encoder.encode_batch([tile_data])

                tile_meta = Tile(
                    x=col * 192,
                    y=row * 192,
                    width=384,
                    height=384,
                    data=None,
                    bounds=t_bounds,
                )

                return (tile_meta, emb)

            except Exception as e:
                logger.debug("[WARN] Tile %s failed: %s", idx, e)
                return None

        import concurrent.futures

        processed_tiles = []
        tile_embeddings = []
        max_workers = 12

        update_progress(0.0)
        log_status(f"Processing {max_workers} tiles in parallel.")

        total_tiles = len(grid_tiles)
        completed = 0

        task_args = [(i, t[0], t[1], t[2]) for i, t in enumerate(grid_tiles)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {executor.submit(process_tile_task, arg): arg for arg in task_args}

            for future in concurrent.futures.as_completed(future_to_tile):
                result = future.result()
                completed += 1

                if completed % 5 == 0:
                    progress = min(1.0, completed / total_tiles)
                    update_progress(progress)

                if result:
                    t_meta, t_emb = result
                    processed_tiles.append(t_meta)
                    tile_embeddings.append(t_emb)

        update_progress(1.0)

        if not processed_tiles:
            logger.warning("No valid data found in the selected area.")
            return []

        tile_embeddings = np.vstack(tile_embeddings)

        log_status("🔍 Ranking results by similarity...")

        similarities = np.dot(tile_embeddings, query_embedding.T).flatten()

        top_k = min(params.top_k, len(processed_tiles))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        log_status(f"🔥 Fetching full details for top {top_k} matches...")

        from models.dofa_clip import get_model
        from xai.grad_eclip import generate_explanation

        model_wrapper = get_model()

        for rank, idx in enumerate(top_indices):
            tile = processed_tiles[idx]
            score = float(similarities[idx])

            if score < params.similarity_threshold:
                logger.debug(
                    "Skipping tile %s: score %s < threshold %s",
                    idx,
                    score,
                    params.similarity_threshold,
                )
                continue

            t_minx, t_miny, t_maxx, t_maxy = tile.bounds
            tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])

            tile_composite = retriever.get_composite(tile_geom, start_date, end_date)
            tile_composite = retriever.normalize_for_model(tile_composite)

            tile_data = download_image_as_array(tile_composite, tile_geom, scale=target_resolution)

            from data.preprocessing import prepare_for_model
            tile_data = prepare_for_model(tile_data)

            tile.data = tile_data

            rgb_image = get_rgb_visualization(tile.data)

            try:
                heatmap = generate_explanation(model_wrapper, tile.data, params.query)
            except Exception as e:
                logger.warning("Grad-CAM failed for tile %s: %s", idx, e)
                heatmap = np.ones((224, 224)) * score

            results.append({
                "image": rgb_image,
                "heatmap": heatmap,
                "score": score,
                "bounds": tile.bounds,
            })

        if len(results) == 0:
            logger.warning(
                "No results above similarity threshold (%s).",
                params.similarity_threshold,
            )

        return results

    except Exception as e:
        logger.exception("Search failed: %s", e)
        return []

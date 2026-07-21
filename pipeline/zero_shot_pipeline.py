"""
Zero-Shot Detection Pipeline
Uses DINOv3 for query-based object detection in satellite imagery.

Two data paths:
- Area-cached (preferred): reuses a LoadedArea's already-downloaded tiles.
  Per-tile DINOv3 patch features are query-independent (only the final
  similarity-vs-query dot product depends on the reference patch), so they
  are cached in area.embeddings["dinov3"] via embed_area_dinov3 — a repeat
  detection with a different reference patch costs a dot product per tile,
  not a new forward pass, and needs no GEE round trip at all.
- Fetch (fallback): the original behavior — fetches its own composite and
  grid when no LoadedArea is available.
"""

from typing import Callable, List, Optional

import numpy as np
import torch
import streamlit as st

# Import existing utilities
from pipeline.tiling import generate_geo_grid
from pipeline.area_store import LoadedArea
from data.preprocessing import download_image_as_array
from models.dinov3 import DINOv3Wrapper
from config import sentinel1_bands, sentinel2_bands

DINOV3_MODEL_KEY = "dinov3"

ProgressCB = Optional[Callable[[str, Optional[float]], None]]


def _report(progress_cb: ProgressCB, msg: str, frac: Optional[float] = None) -> None:
    if progress_cb:
        progress_cb(msg, frac)


def _bands_for_sensor(sensor: str) -> List[str]:
    return sentinel1_bands.band_names if sensor == "Sentinel-1" else sentinel2_bands.band_names


def _tile_to_uint8(tile: np.ndarray, sensor: str) -> np.ndarray:
    """(C, H, W) float [0,1] cached tile -> (H, W, 3) uint8, matching the
    original fetch path's normalization (no brightness stretch — DINOv3
    was not trained on stretched imagery)."""
    if sensor == "Sentinel-1":
        vv = np.clip(tile[0], 0, 1)
        vh = np.clip(tile[1], 0, 1) if tile.shape[0] > 1 else vv
        rgb = np.stack([vv, vh, vv], axis=-1)
        return (rgb * 255).astype(np.uint8)

    bands = sentinel2_bands.band_names

    def _find(candidates):
        for name in candidates:
            if name in bands:
                return bands.index(name)
        return None

    r_idx = _find(["B4", "B04"])
    g_idx = _find(["B3", "B03"])
    b_idx = _find(["B2", "B02"])
    if r_idx is None or g_idx is None or b_idx is None:
        r_idx, g_idx, b_idx = 0, 1, 2

    rgb = np.stack([tile[r_idx], tile[g_idx], tile[b_idx]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def embed_area_dinov3(
    area: LoadedArea,
    model: DINOv3Wrapper,
    progress_cb: ProgressCB = None,
) -> np.ndarray:
    """
    Extract centered DINOv3 patch features for every cached tile in `area`.

    No-op (returns cached array) if area.embeddings["dinov3"] already exists.
    Result shape (N_tiles, N_patches, D) — a per-tile 2D feature map, not a
    single vector, since scoring needs spatial detail for the heatmap.
    """
    if DINOV3_MODEL_KEY in area.embeddings:
        return area.embeddings[DINOV3_MODEL_KEY]

    total = area.num_tiles
    feats_list = []
    for i in range(total):
        tile = area.tile_arrays[i].astype(np.float32)
        arr_uint8 = _tile_to_uint8(tile, area.params.sensor)
        features = model.extract_features(arr_uint8, center_features=True)  # (1, N, D)
        feats_list.append(features.squeeze(0).cpu().numpy())
        if (i + 1) % 5 == 0 or i + 1 == total:
            _report(progress_cb, f"Extracting DINOv3 features {i + 1}/{total}...", (i + 1) / total)

    area.embeddings[DINOV3_MODEL_KEY] = np.stack(feats_list, axis=0).astype(np.float32)
    return area.embeddings[DINOV3_MODEL_KEY]


def score_area_dinov3(
    area: LoadedArea,
    query_vector: torch.Tensor,
    model_key: str = DINOV3_MODEL_KEY,
) -> List[np.ndarray]:
    """Per-tile similarity maps (grid_h, grid_w) against a reference query vector."""
    features = area.embeddings.get(model_key)
    if features is None:
        raise ValueError(f"Area has no '{model_key}' features — call embed_area_dinov3() first.")

    query_vector = query_vector.detach().cpu().float()
    query_norm = query_vector / query_vector.norm()

    sim_maps = []
    for tile_feats in features:
        t = torch.from_numpy(tile_feats)
        feats_norm = t / t.norm(dim=1, keepdim=True)
        sims = (feats_norm @ query_norm).numpy()

        grid_dim = int(np.sqrt(len(sims)))
        if grid_dim * grid_dim == len(sims):
            sim_map = sims.reshape(grid_dim, grid_dim)
        else:
            sim_map = sims.reshape(1, -1)
        sim_maps.append(sim_map)

    return sim_maps


def _run_zero_shot_from_area(
    area: LoadedArea,
    query_vector: torch.Tensor,
    threshold: float,
    hf_token: Optional[str],
    progress_cb: ProgressCB,
) -> list:
    model = DINOv3Wrapper(token=hf_token)
    model._load_model()

    embed_area_dinov3(area, model, progress_cb=progress_cb)
    sim_maps = score_area_dinov3(area, query_vector)

    import skimage.transform

    detections = []
    for idx, sim_map in enumerate(sim_maps):
        max_score = float(sim_map.max())
        if max_score <= threshold:
            continue

        arr_uint8 = _tile_to_uint8(area.tile_arrays[idx].astype(np.float32), area.params.sensor)
        heatmap_resized = skimage.transform.resize(
            sim_map,
            (arr_uint8.shape[0], arr_uint8.shape[1]),
            order=3,
            mode="reflect",
            anti_aliasing=True,
        )
        detections.append({
            "image": arr_uint8,
            "heatmap": heatmap_resized,
            "score": max_score,
            "bounds": area.tile_bounds[idx],
        })

    from pipeline.postprocessing import nms_results
    detections = nms_results(detections)

    for det in detections:
        det["dino_attention"] = model.get_attention_map(det["image"])
        det["pca_map"] = model.get_pca_map(det["image"], center_features=True)

    return detections


def _run_zero_shot_fetch(
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    query_vector: torch.Tensor,
    sensor: str,
    threshold: float,
    resolution: int,
    hf_token: Optional[str],
) -> list:
    """Original fetch-based path — used when no LoadedArea is available."""
    import ee

    from data.gee_client import GEEClient
    if not GEEClient.is_initialized():
        GEEClient.initialize()

    try:
        model = DINOv3Wrapper(token=hf_token)
        model._load_model()
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        return []

    st.info(f"🛰️ Fetching target {sensor} imagery...")
    if sensor == "Sentinel-1":
        from data.sentinel1 import Sentinel1Retriever
        retriever = Sentinel1Retriever()
        bands = ["VV", "VH"]
    else:
        from data.sentinel2 import Sentinel2Retriever
        retriever = Sentinel2Retriever()
        bands = ["B4", "B3", "B2"]

    if aoi_geojson.get("type") == "Polygon":
        aoi_ee = ee.Geometry.Polygon(aoi_geojson["coordinates"])
    else:
        aoi_ee = ee.Geometry(aoi_geojson)

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

    bounds_info = aoi_ee.bounds().getInfo()["coordinates"][0]
    west = min(p[0] for p in bounds_info)
    south = min(p[1] for p in bounds_info)
    east = max(p[0] for p in bounds_info)
    north = max(p[1] for p in bounds_info)
    bounds = (west, south, east, north)

    TILE_SIZE = 448  # 14 * 32 — divisible by DINOv3's patch size

    grid = list(generate_geo_grid(bounds, resolution, tile_size=TILE_SIZE))
    total_tiles = len(grid)
    if total_tiles > 200:
        st.warning(f"Processing {total_tiles} tiles. This may take time.")

    detections = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Starting analysis of {total_tiles} tiles...")

    query_vector = query_vector.to(model.device)
    query_norm = query_vector / query_vector.norm()

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

            try:
                arr = np.transpose(arr, (1, 2, 0))

                if sensor == "Sentinel-1" and arr.shape[2] == 2:
                    arr = np.stack([arr[:, :, 0], arr[:, :, 1], arr[:, :, 0]], axis=-1)

                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = np.clip(arr, 0, 1)
                    arr_uint8 = (arr * 255).astype(np.uint8)
                else:
                    arr_uint8 = arr

                features = model.extract_features(arr_uint8, center_features=True)
                features = features.squeeze(0)
            except Exception as e:
                print(f"[DEBUG] Tile {i} Feature Extraction Error: {e}")
                continue

            feats_norm = features / features.norm(dim=1, keepdim=True)
            sim_scores = (feats_norm @ query_norm).cpu().numpy()

            grid_dim = int(np.sqrt(len(sim_scores)))

            if grid_dim * grid_dim != len(sim_scores):
                st.warning(f"Feature count {len(sim_scores)} is not a perfect square.")
                sim_map = sim_scores.reshape(1, -1)
            else:
                sim_map = sim_scores.reshape(grid_dim, grid_dim)

            max_score = sim_map.max()
            if max_score > threshold:
                import skimage.transform

                heatmap_resized = skimage.transform.resize(
                    sim_map,
                    (arr_uint8.shape[0], arr_uint8.shape[1]),
                    order=3,
                    mode="reflect",
                    anti_aliasing=True,
                )

                detections.append({
                    "image": arr_uint8,
                    "heatmap": heatmap_resized,
                    "score": float(max_score),
                    "bounds": t_bounds,
                })

    status_text.empty()
    progress_bar.empty()

    from pipeline.postprocessing import nms_results
    detections = nms_results(detections)

    for det in detections:
        det["dino_attention"] = model.get_attention_map(det["image"])
        det["pca_map"] = model.get_pca_map(det["image"], center_features=True)

    return detections


def run_zero_shot_pipeline(
    query_vector: torch.Tensor,
    threshold: float = 0.5,
    hf_token: str = None,
    area: Optional[LoadedArea] = None,
    aoi_geojson: Optional[dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sensor: str = "Sentinel-2",
    resolution: int = 10,
    progress_cb: ProgressCB = None,
) -> list:
    """
    Execute Zero-Shot Detection.

    Pass `area` (a LoadedArea from the sidebar's Load & Embed Area) to reuse
    its cached tiles/features — no GEE round trip on a repeat query. Without
    an area, falls back to fetching aoi_geojson/start_date/end_date/sensor
    directly (the original behavior).

    Args:
        query_vector: (Embed_Dim,) tensor from reference patch.
        threshold: Similarity threshold (0.0 to 1.0).
        hf_token: Hugging Face token.
        area: Optional LoadedArea to search against instead of fetching.
        aoi_geojson/start_date/end_date/sensor/resolution: Fetch-path args,
            required when `area` is None.

    Returns:
        List of detections [{'image', 'heatmap', 'score', 'bounds', ...}]
    """
    if area is not None:
        return _run_zero_shot_from_area(area, query_vector, threshold, hf_token, progress_cb)

    if aoi_geojson is None:
        raise ValueError("Either `area` or `aoi_geojson` must be provided.")

    return _run_zero_shot_fetch(
        aoi_geojson, start_date, end_date, query_vector, sensor, threshold, resolution, hf_token
    )

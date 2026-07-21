"""
Ingest — Phase A of the two-phase workflow.

Fetches a GEE composite for an AOI, tiles it, downloads every tile in
parallel, and caches the model-ready pixel arrays in a LoadedArea. No
encoding happens here — that's `embed_area`'s job — so this module has no
model dependency and can be called well before a query exists.

No Streamlit calls: progress is reported via `progress_cb(msg, frac)` so the
module stays testable, matching the pattern used by CopernicusSearchPipeline.
"""

import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import model_config, sentinel1_bands, sentinel2_bands
from pipeline.area_store import AreaParams, LoadedArea, compute_area_id, new_area_id_timestamp

MAX_TILES = 25000
MAX_WORKERS = 12

# Minimum fraction of a tile that must carry real data. GEE returns 0 for
# masked pixels (clouds, swath gaps, past the coastline/imagery edge), so a
# tile that is a thin sliver of imagery on a black background passes the
# max()==0 check but is useless. Require at least this much valid coverage.
MIN_VALID_FRACTION = 0.5

ProgressCB = Optional[Callable[[str, Optional[float]], None]]

# Model family -> (human label, the search tab it unlocks). An area is embedded
# with exactly one family at load time; the UI then constrains search to the
# matching tab. Keep the keys in sync with AreaParams.model.
MODEL_FAMILIES = {
    "dofa": {
        "label": "DOFA-CLIP — text & reference-image semantic search",
        "tab": "semantic",
        "needs_hf_token": False,
    },
    "dinov3": {
        "label": "DINOv3 — zero-shot visual detection",
        "tab": "zeroshot",
        "needs_hf_token": True,
    },
    "copernicus": {
        "label": "CopernicusFM — foundation-model similarity",
        "tab": "copernicus",
        "needs_hf_token": False,
    },
}


def _report(progress_cb: ProgressCB, msg: str, frac: Optional[float] = None) -> None:
    if progress_cb:
        progress_cb(msg, frac)


def model_key_for_sensor(sensor: str) -> str:
    """Model cache key — S1/S2 embeddings are incompatible and must not share a key."""
    return "dofa_s1" if sensor == "Sentinel-1" else "dofa_s2"


def embed_area_with_model(
    area,
    model: str,
    hf_token: Optional[str] = None,
    progress_cb: ProgressCB = None,
) -> None:
    """
    Embed a freshly loaded area with the single model family chosen at load
    time. Heavy model imports stay lazy so unused backends aren't loaded.

    - "dofa"       -> DOFA-CLIP (sensor-keyed) via embed_area()
    - "dinov3"     -> DINOv3 patch features via embed_area_dinov3()
    - "copernicus" -> CopernicusFM tile vectors via embed_search_area()
    """
    if model == "dofa":
        embed_area(area, model_key=model_key_for_sensor(area.params.sensor), progress_cb=progress_cb)

    elif model == "dinov3":
        from models.dinov3 import DINOv3Wrapper
        from pipeline.zero_shot_pipeline import embed_area_dinov3

        wrapper = DINOv3Wrapper(token=hf_token)
        wrapper._load_model()
        embed_area_dinov3(area, wrapper, progress_cb=progress_cb)

    elif model == "copernicus":
        from datetime import datetime as _dt

        from pipeline.copernicus_pipeline import CopernicusSearchPipeline

        pipeline = CopernicusSearchPipeline()
        date_obj = _dt.strptime(area.params.start_date, "%Y-%m-%d")
        # embed_search_area's callback is (msg) only — adapt to (msg, frac).
        cb = (lambda m: _report(progress_cb, m)) if progress_cb else None
        pipeline.embed_search_area(area, date_obj, area.params.resolution, progress_callback=cb)

    else:
        raise ValueError(f"Unknown model family: {model!r}")


def load_area(
    params: AreaParams,
    progress_cb: ProgressCB = None,
    name: Optional[str] = None,
) -> LoadedArea:
    """
    Fetch + tile + download an AOI into a LoadedArea (no embeddings yet).

    Raises ValueError for user-actionable problems (area too large, no valid
    tiles found) so callers can surface a clean message.
    """
    import ee

    from data.gee_client import GEEClient
    if not GEEClient.is_initialized():
        _report(progress_cb, "Initializing Google Earth Engine...", 0.0)
        GEEClient.initialize()

    _report(progress_cb, "Processing area of interest...", 0.02)
    aoi_geojson = params.aoi_geojson
    if aoi_geojson.get("type") == "Polygon":
        aoi_ee = ee.Geometry.Polygon(aoi_geojson["coordinates"])
    else:
        aoi_ee = ee.Geometry(aoi_geojson)

    sensor = params.sensor
    _report(progress_cb, f"Fetching {sensor} imagery from Google Earth Engine...", 0.05)

    if sensor == "Sentinel-1":
        from data.sentinel1 import Sentinel1Retriever
        retriever = Sentinel1Retriever()
        bands_to_download = sentinel1_bands.band_names
    else:
        from data.sentinel2 import Sentinel2Retriever
        retriever = Sentinel2Retriever()
        bands_to_download = sentinel2_bands.band_names

    composite = retriever.get_composite(aoi_ee, params.start_date, params.end_date)
    composite = retriever.normalize_for_model(composite)

    _report(progress_cb, "Generating search grid...", 0.08)
    from pipeline.tiling import generate_geo_grid

    bounds_info = aoi_ee.bounds().getInfo()["coordinates"][0]
    west = min(p[0] for p in bounds_info)
    south = min(p[1] for p in bounds_info)
    east = max(p[0] for p in bounds_info)
    north = max(p[1] for p in bounds_info)
    bounds = (west, south, east, north)

    target_resolution = params.resolution
    chip_size = params.chip_size

    deg_width = east - west
    deg_height = north - south
    meters_width = deg_width * 111320
    meters_height = deg_height * 111320
    tile_m = chip_size * target_resolution
    stride_m = tile_m * 0.5
    est_cols = max(1, meters_width / stride_m)
    est_rows = max(1, meters_height / stride_m)
    total_est_tiles = est_cols * est_rows

    if total_est_tiles > MAX_TILES:
        raise ValueError(
            f"Too many tiles ({int(total_est_tiles)}). Reduce the area or "
            f"increase resolution above {target_resolution}m."
        )

    grid_tiles = list(generate_geo_grid(bounds, resolution=target_resolution, tile_size=chip_size))

    if len(grid_tiles) > MAX_TILES:
        raise ValueError("Area is too large. Please select a smaller region.")

    _report(progress_cb, f"Created grid with {len(grid_tiles)} tiles.", 0.1)

    from data.preprocessing import download_image_as_array, prepare_for_model

    def download_tile_task(args: Tuple[int, tuple]):
        idx, t_bounds = args
        try:
            t_minx, t_miny, t_maxx, t_maxy = t_bounds
            tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])

            tile_data = download_image_as_array(
                composite, tile_geom, bands=bands_to_download, scale=target_resolution
            )
            if tile_data.max() == 0:
                return None

            # Drop tiles that are mostly nodata (a pixel is valid if any band
            # is non-zero; masked pixels are 0 across all bands).
            valid_frac = float(np.mean(np.any(tile_data > 0, axis=0)))
            if valid_frac < MIN_VALID_FRACTION:
                return None

            # Data is already [0, 1]: composite was normalized server-side by
            # retriever.normalize_for_model(). prepare_for_model() only resizes.
            tile_data = prepare_for_model(tile_data)
            if tile_data.max() == 0:
                return None

            return (idx, t_bounds, tile_data.astype(np.float16))
        except Exception:
            return None

    task_args = [(i, t[0]) for i, t in enumerate(grid_tiles)]
    collected: Dict[int, Tuple[tuple, np.ndarray]] = {}
    total_tiles = len(grid_tiles)
    completed = 0

    _report(progress_cb, f"Downloading with {MAX_WORKERS} parallel workers...", 0.1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_tile = {executor.submit(download_tile_task, arg): arg for arg in task_args}
        for future in concurrent.futures.as_completed(future_to_tile):
            result = future.result()
            completed += 1

            if completed % 5 == 0 or completed == total_tiles:
                frac = 0.1 + 0.85 * (completed / total_tiles)
                _report(progress_cb, f"Downloading tile {completed}/{total_tiles}...", frac)

            if result:
                idx, t_bounds, tile_data = result
                collected[idx] = (t_bounds, tile_data)

    if not collected:
        raise ValueError("No valid data found in the selected area.")

    ordered_idx = sorted(collected.keys())
    tile_bounds = [collected[i][0] for i in ordered_idx]
    tile_arrays = np.stack([collected[i][1] for i in ordered_idx], axis=0)

    area_params = AreaParams(
        aoi_geojson=params.aoi_geojson,
        start_date=params.start_date,
        end_date=params.end_date,
        sensor=params.sensor,
        resolution=params.resolution,
        chip_size=params.chip_size,
        model=params.model,
    )
    area_id = compute_area_id(area_params)

    area = LoadedArea(
        area_id=area_id,
        name=name or f"Area {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        created_at=new_area_id_timestamp(),
        params=area_params,
        tile_bounds=tile_bounds,
        tile_arrays=tile_arrays,
        embeddings={},
    )

    _report(progress_cb, f"Area loaded: {len(tile_bounds)} tiles.", 1.0)
    return area


def embed_area(
    area: LoadedArea,
    image_encoder=None,
    model_key: Optional[str] = None,
    progress_cb: ProgressCB = None,
    area_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Batched encoding of area.tile_arrays. No-op (returns cached array) if
    `model_key` was already embedded for this area.

    Non-finite embedding rows are dropped along with their tile — index
    alignment across tile_bounds/tile_arrays/every cached embedding must be
    preserved, so a filter here is applied to all of them together.

    If `area_dir` is given (the area is already persisted), the resulting
    embedding is also written to `emb_{model_key}.npy` there.
    """
    if model_key is None:
        model_key = model_key_for_sensor(area.params.sensor)

    if model_key in area.embeddings:
        return area.embeddings[model_key]

    if image_encoder is None:
        from models.encoders import create_encoders
        wavelengths = (
            sentinel1_bands.get_wavelength_tensor()
            if area.params.sensor == "Sentinel-1"
            else sentinel2_bands.get_wavelength_tensor()
        )
        _, image_encoder = create_encoders(wavelengths=wavelengths)

    encode_batch = model_config.batch_size
    tiles = area.tile_arrays
    total = len(tiles)

    embedding_chunks = []
    for start in range(0, total, encode_batch):
        end = min(start + encode_batch, total)
        batch = tiles[start:end].astype(np.float32)
        embs = image_encoder.encode_batch(list(batch))
        embedding_chunks.append(embs)
        _report(progress_cb, f"Encoding tile {end}/{total}...", end / total if total else 1.0)

    all_embeddings = np.concatenate(embedding_chunks, axis=0)
    keep_mask = np.isfinite(all_embeddings).all(axis=1)

    if not keep_mask.all():
        area.tile_arrays = area.tile_arrays[keep_mask]
        area.tile_bounds = [b for b, ok in zip(area.tile_bounds, keep_mask) if ok]
        all_embeddings = all_embeddings[keep_mask]
        for key, emb in list(area.embeddings.items()):
            area.embeddings[key] = emb[keep_mask]

    area.embeddings[model_key] = all_embeddings.astype(np.float32)

    if area_dir is not None:
        area_dir = Path(area_dir)
        area_dir.mkdir(parents=True, exist_ok=True)
        np.save(area_dir / f"emb_{model_key}.npy", area.embeddings[model_key])

    return area.embeddings[model_key]

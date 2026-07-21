"""
Semantic Search — Phase B of the two-phase workflow.

Runs against an already-loaded, already-embedded LoadedArea: encode the
query, score with a single np.dot, threshold -> NMS -> top-k, then Grad-CAM
straight from the cached tile pixels (no re-download).

`rank_candidates` and `explain_candidates` are exposed separately from
`search_area` so a UI can re-rank live (dragging the threshold/top_k
sliders) using the cached `similarities` array without re-encoding anything,
reusing a per-query `{idx: heatmap}` cache for Grad-CAM.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import sentinel1_bands, sentinel2_bands
from pipeline.area_store import LoadedArea
from pipeline.ingest import model_key_for_sensor
from pipeline.postprocessing import nms_results


@dataclass
class ModelBundle:
    """Encoders/model shared across a search — image_encoder must match the
    area's sensor wavelengths (S1 vs S2 embeddings are incompatible)."""

    text_encoder: Any
    image_encoder: Any
    model_wrapper: Any = None


def _bands_for_sensor(sensor: str) -> List[str]:
    return sentinel1_bands.band_names if sensor == "Sentinel-1" else sentinel2_bands.band_names


def encode_query(params, model_bundle: ModelBundle) -> np.ndarray:
    """Encode the text query or reference image into a query embedding."""
    if params.search_type == "image":
        if not params.reference_image:
            raise ValueError("Reference image search selected, but no image was provided.")

        ref_img = Image.open(io.BytesIO(params.reference_image)).convert("RGB")
        ref_arr = np.asarray(ref_img, dtype=np.float32) / 255.0
        ref_arr = np.transpose(ref_arr, (2, 0, 1))

        from models.encoders import create_encoders
        # Approximate RGB wavelength mapping (R, G, B in nm).
        _, ref_encoder = create_encoders(
            model=model_bundle.text_encoder.model,
            wavelengths=[665.0, 560.0, 490.0],
        )
        return ref_encoder.encode(ref_arr)

    return model_bundle.text_encoder.encode(params.query)


def score_area(area: LoadedArea, query_embedding: np.ndarray, model_key: Optional[str] = None) -> np.ndarray:
    """Cosine-similarity score every cached tile embedding against the query."""
    if model_key is None:
        model_key = model_key_for_sensor(area.params.sensor)

    embeddings = area.embeddings.get(model_key)
    if embeddings is None:
        raise ValueError(f"Area has no '{model_key}' embeddings — call embed_area() first.")

    return np.dot(embeddings, query_embedding.T).flatten()


def build_diagnostics(area: LoadedArea, similarities: np.ndarray, params, query_embedding: np.ndarray) -> dict:
    """Similarity distribution stats for the collapsed diagnostics panel."""
    hist_counts, hist_edges = np.histogram(similarities, bins=10)

    return {
        "source": "semantic",
        "sensor": area.params.sensor,
        "query_type": params.search_type,
        "query_norm": float(np.linalg.norm(query_embedding)),
        "embedding_dim": int(query_embedding.shape[-1]) if query_embedding.ndim > 1 else int(query_embedding.shape[0]),
        "total_tiles": int(area.num_tiles),
        "valid_tiles": int(area.num_tiles),
        "scored_tiles": int(len(similarities)),
        "above_threshold": int(np.sum(similarities >= params.similarity_threshold)),
        "sim_min": float(np.min(similarities)),
        "sim_mean": float(np.mean(similarities)),
        "sim_median": float(np.median(similarities)),
        "sim_max": float(np.max(similarities)),
        "sim_std": float(np.std(similarities)),
        "similarity_histogram_counts": hist_counts.tolist(),
        "similarity_histogram_edges": hist_edges.tolist(),
        "top_scores": np.sort(similarities)[::-1][:10].tolist(),
        "threshold": float(params.similarity_threshold),
    }


def rank_candidates(area: LoadedArea, similarities: np.ndarray, threshold: float, top_k: int) -> List[dict]:
    """Threshold -> geographic NMS (grid has 50% overlap) -> top-k."""
    passing = np.where(similarities >= threshold)[0]
    candidates = [
        {"idx": int(i), "score": float(similarities[i]), "bounds": area.tile_bounds[i]}
        for i in passing
    ]
    deduplicated = nms_results(candidates)
    return deduplicated[: min(top_k, len(deduplicated))]


def explain_candidates(
    area: LoadedArea,
    candidates: List[dict],
    params,
    model_bundle: ModelBundle,
    heatmap_cache: Optional[Dict[int, Optional[np.ndarray]]] = None,
) -> List[dict]:
    """Build result dicts (RGB + Grad-CAM) straight from cached tile pixels.

    `heatmap_cache` is a per-query {idx: heatmap} dict; pass the same dict
    back in on a live re-rank so already-explained tiles skip Grad-CAM.
    """
    from data.preprocessing import get_rgb_visualization
    from xai.grad_eclip import generate_explanation

    bands = _bands_for_sensor(area.params.sensor)
    heatmap_cache = heatmap_cache if heatmap_cache is not None else {}

    results = []
    for candidate in candidates:
        idx = candidate["idx"]
        tile_data = area.tile_arrays[idx].astype(np.float32)
        rgb_image = get_rgb_visualization(tile_data, bands=bands)

        if idx not in heatmap_cache and params.search_type == "text":
            try:
                heatmap = generate_explanation(
                    model_bundle.model_wrapper,
                    tile_data,
                    params.query,
                    wavelengths=model_bundle.image_encoder.wavelengths,
                )
            except Exception as e:
                print(f"[WARN] Grad-CAM failed for tile {idx}: {e}")
                heatmap = None
            heatmap_cache[idx] = heatmap
        else:
            heatmap = heatmap_cache.get(idx)

        results.append({
            "image": rgb_image,
            "heatmap": heatmap,
            "score": candidate["score"],
            "bounds": candidate["bounds"],
        })

    return results


def search_area(
    area: LoadedArea,
    params,
    model_bundle: ModelBundle,
    heatmap_cache: Optional[Dict[int, Optional[np.ndarray]]] = None,
) -> Tuple[List[dict], dict]:
    """One-shot search: encode -> score -> rank -> explain.

    diagnostics['similarities'] carries the raw per-tile similarity array so
    a caller can stash it (e.g. st.session_state.last_similarities) and later
    call rank_candidates/explain_candidates directly for instant re-ranking.
    """
    query_embedding = encode_query(params, model_bundle)
    similarities = score_area(area, query_embedding)

    diagnostics = build_diagnostics(area, similarities, params, query_embedding)
    candidates = rank_candidates(area, similarities, params.similarity_threshold, params.top_k)
    diagnostics["after_nms"] = len(candidates)

    results = explain_candidates(area, candidates, params, model_bundle, heatmap_cache=heatmap_cache)
    diagnostics["similarities"] = similarities

    return results, diagnostics

"""Helpers for serializing search results for the API."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from xai.visualization import create_heatmap_overlay


def _encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _encode_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray) -> str:
    overlay = create_heatmap_overlay(image, heatmap)
    return _encode_png(overlay)


def _encode_heatmap_raw(heatmap: np.ndarray) -> str:
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap_img = (heatmap_norm * 255).astype(np.uint8)
    return _encode_png(heatmap_img)


def serialize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized = []
    for result in results:
        image = result.get("image")
        heatmap = result.get("heatmap")
        dino_attention = result.get("dino_attention")
        pca_map = result.get("pca_map")

        payload = {
            "score": float(result.get("score", 0.0)),
            "bounds": result.get("bounds"),
            "geometry": result.get("geometry"),
            "image": _encode_png(image) if image is not None else None,
            "heatmap": _encode_heatmap_overlay(image, heatmap)
            if image is not None and heatmap is not None
            else None,
            "heatmap_raw": _encode_heatmap_raw(heatmap) if heatmap is not None else None,
            "dino_attention": _encode_heatmap_overlay(image, dino_attention)
            if image is not None and dino_attention is not None
            else None,
            "pca_map": _encode_png(pca_map) if pca_map is not None else None,
        }

        serialized.append(payload)

    return serialized


def _decode_png(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image)


def _decode_heatmap(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    image = Image.open(io.BytesIO(raw)).convert("L")
    return np.array(image) / 255.0


def deserialize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deserialized = []
    for result in results:
        image = _decode_png(result["image"]) if result.get("image") else None
        heatmap_data = result.get("heatmap_raw") or result.get("heatmap")
        heatmap = _decode_heatmap(heatmap_data) if heatmap_data else None
        deserialized.append(
            {
                "image": image,
                "heatmap": heatmap,
                "score": result.get("score"),
                "bounds": result.get("bounds"),
                "geometry": result.get("geometry"),
            }
        )

    return deserialized

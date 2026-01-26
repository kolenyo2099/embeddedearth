"""FastAPI backend for the EmbeddedEarth Svelte frontend."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from app.export_utils import generate_geojson, generate_kmz, generate_pdf_report, generate_zip_package
from app.services.result_serialization import deserialize_results, serialize_results
from app.services.semantic_search import SemanticSearchParams, run_semantic_search
from pipeline.copernicus_pipeline import CopernicusSearchPipeline
from pipeline.zero_shot_pipeline import run_zero_shot_pipeline
from models.dinov3 import DINOv3Wrapper


class GEEConnectRequest(BaseModel):
    project_id: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    aoi_geojson: Dict[str, Any]
    query: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    top_k: int = Field(10, ge=1)
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0)
    resolution: float = Field(10.0, ge=1.0)
    search_type: str = "text"
    reference_image_base64: Optional[str] = None


class ZeroShotSearchRequest(BaseModel):
    aoi_geojson: Dict[str, Any]
    start_date: str
    end_date: str
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    resolution: int = Field(10, ge=1)
    hf_token: Optional[str] = None
    reference_image_base64: str
    bbox: List[int]


class CopernicusSearchRequest(BaseModel):
    query_geom: Dict[str, Any]
    search_geom: Dict[str, Any]
    start_date: str
    end_date: str
    sensor: str = "Sentinel-2"
    resolution: int = Field(10, ge=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class ExportRequest(BaseModel):
    results: List[Dict[str, Any]]
    query: str


app = FastAPI(title="EmbeddedEarth API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/gee/connect")
def connect_gee(payload: GEEConnectRequest) -> Dict[str, Optional[str]]:
    from data.gee_client import GEEClient

    try:
        GEEClient.initialize(project_id=payload.project_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "connected", "project_id": GEEClient.get_project_id()}


@app.post("/api/search/semantic")
def semantic_search(payload: SemanticSearchRequest) -> Dict[str, Any]:
    start_date = datetime.strptime(payload.start_date, "%Y-%m-%d").date() if payload.start_date else None
    end_date = datetime.strptime(payload.end_date, "%Y-%m-%d").date() if payload.end_date else None

    params = SemanticSearchParams(
        query=payload.query,
        start_date=start_date,
        end_date=end_date,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
        resolution=payload.resolution,
        search_type=payload.search_type,
        reference_image=base64.b64decode(payload.reference_image_base64)
        if payload.reference_image_base64
        else None,
    )

    results = run_semantic_search(payload.aoi_geojson, params)
    return {"results": serialize_results(results)}


@app.post("/api/search/zero-shot")
def zero_shot_search(payload: ZeroShotSearchRequest) -> Dict[str, Any]:
    try:
        image_bytes = base64.b64decode(payload.reference_image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid reference image.") from exc

    image_array = np.array(image)
    bbox = payload.bbox

    try:
        wrapper = DINOv3Wrapper(token=payload.hf_token)
        query_vector = wrapper.get_patch_embedding(image_array, bbox)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = run_zero_shot_pipeline(
        aoi_geojson=payload.aoi_geojson,
        start_date=payload.start_date,
        end_date=payload.end_date,
        query_vector=query_vector,
        threshold=payload.threshold,
        resolution=payload.resolution,
        hf_token=payload.hf_token,
    )

    return {"results": serialize_results(results)}


@app.post("/api/search/copernicus")
def copernicus_search(payload: CopernicusSearchRequest) -> Dict[str, Any]:
    pipeline = CopernicusSearchPipeline()
    results = pipeline.run_search(
        query_geom=payload.query_geom,
        search_geom=payload.search_geom,
        start_date=payload.start_date,
        end_date=payload.end_date,
        sensor=payload.sensor,
        resolution=payload.resolution,
        threshold=payload.threshold,
    )

    return {"results": serialize_results(results)}


@app.post("/api/export/pdf")
def export_pdf(payload: ExportRequest) -> Response:
    results = deserialize_results(payload.results)
    pdf_bytes = generate_pdf_report(results, payload.query)
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=embeddedearth_report.pdf"},
    )


@app.post("/api/export/kmz")
def export_kmz(payload: ExportRequest) -> Response:
    results = deserialize_results(payload.results)
    kmz_bytes = generate_kmz(results)
    return Response(
        content=kmz_bytes,
        media_type="application/vnd.google-earth.kmz",
        headers={"Content-Disposition": "attachment; filename=results.kmz"},
    )


@app.post("/api/export/geojson")
def export_geojson(payload: ExportRequest) -> Response:
    results = deserialize_results(payload.results)
    geojson_text = generate_geojson(results)
    return Response(
        content=geojson_text,
        media_type="application/geo+json",
        headers={"Content-Disposition": "attachment; filename=results.geojson"},
    )


@app.post("/api/export/zip")
def export_zip(payload: ExportRequest) -> Response:
    results = deserialize_results(payload.results)
    zip_bytes = generate_zip_package(results, payload.query)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=results.zip"},
    )


frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

"""
Area Store

The LoadedArea model and its disk persistence. A LoadedArea holds everything
downloaded and (lazily) embedded for one AOI/date/sensor/resolution/chip
combination, so Phase B (search) never has to touch Google Earth Engine again.

Key invariant: tile embeddings depend only on
(aoi, dates, sensor, bands, resolution, chip_size, model) — never on the
query. `AreaParams` captures exactly that; query/threshold/top_k are not
part of the area identity.

An area is embedded with exactly ONE model family (chosen at load time), so
`model` is part of the area identity: loading the same AOI with a different
model yields a distinct area. See `MODEL_FAMILIES` in pipeline/ingest.py.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import area_store_config


@dataclass
class AreaParams:
    """Everything that identifies a loaded area's downloaded imagery."""

    aoi_geojson: dict
    start_date: str
    end_date: str
    sensor: str  # "Sentinel-2" | "Sentinel-1"
    resolution: float
    chip_size: int
    model: str = "dofa"  # model family: "dofa" | "dinov3" | "copernicus"

    def to_canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict) -> "AreaParams":
        return cls(
            aoi_geojson=data["aoi_geojson"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            sensor=data["sensor"],
            resolution=data["resolution"],
            chip_size=data["chip_size"],
            # Areas saved before model-first embedding predate this field and
            # were all DOFA-CLIP.
            model=data.get("model", "dofa"),
        )


@dataclass
class LoadedArea:
    """A downloaded, tiled area with lazily-computed embeddings."""

    area_id: str
    name: str
    created_at: str
    params: AreaParams
    tile_bounds: List[Tuple[float, float, float, float]]
    tile_arrays: np.ndarray  # (N, C, H, W) float16, model-ready
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)  # model_key -> (N, D) float32

    @property
    def num_tiles(self) -> int:
        return len(self.tile_bounds)


def compute_area_id(params: AreaParams) -> str:
    """Hash of the canonical JSON of AreaParams. Changing any load parameter
    (including the AOI) yields a different id — query, threshold, and top_k
    are not part of the id."""
    digest = hashlib.sha1(params.to_canonical_json().encode("utf-8")).hexdigest()
    return digest


def _area_dir(area_id: str, base_dir: Optional[Path] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else area_store_config.areas_dir
    return base / area_id


def save_area(area: LoadedArea, base_dir: Optional[Path] = None) -> Path:
    """Write meta.json, tiles.npz, and any cached embeddings to disk."""
    directory = _area_dir(area.area_id, base_dir)
    directory.mkdir(parents=True, exist_ok=True)

    meta = {
        "area_id": area.area_id,
        "name": area.name,
        "created_at": area.created_at,
        "params": asdict(area.params),
        "tile_bounds": area.tile_bounds,
        "num_tiles": area.num_tiles,
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    np.savez_compressed(directory / "tiles.npz", tiles=area.tile_arrays)

    for model_key, emb in area.embeddings.items():
        np.save(directory / f"emb_{model_key}.npy", emb)

    return directory


def load_area_from_disk(area_id: str, base_dir: Optional[Path] = None) -> LoadedArea:
    """Load a previously saved area — no GEE required."""
    directory = _area_dir(area_id, base_dir)
    meta = json.loads((directory / "meta.json").read_text())

    tiles_npz = np.load(directory / "tiles.npz")
    tile_arrays = tiles_npz["tiles"]

    embeddings = {}
    for emb_path in directory.glob("emb_*.npy"):
        model_key = emb_path.stem[len("emb_"):]
        embeddings[model_key] = np.load(emb_path)

    return LoadedArea(
        area_id=meta["area_id"],
        name=meta["name"],
        created_at=meta["created_at"],
        params=AreaParams.from_dict(meta["params"]),
        tile_bounds=[tuple(b) for b in meta["tile_bounds"]],
        tile_arrays=tile_arrays,
        embeddings=embeddings,
    )


def list_saved_areas(base_dir: Optional[Path] = None) -> List[dict]:
    """List metadata for every saved area, newest first."""
    base = Path(base_dir) if base_dir is not None else area_store_config.areas_dir
    if not base.exists():
        return []

    metas = []
    for entry in base.iterdir():
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            metas.append(json.loads(meta_path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue

    metas.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return metas


def delete_area(area_id: str, base_dir: Optional[Path] = None) -> None:
    """Permanently remove a saved area's directory."""
    directory = _area_dir(area_id, base_dir)
    if directory.exists():
        shutil.rmtree(directory)


def new_area_id_timestamp() -> str:
    """ISO-8601 UTC timestamp for `created_at`."""
    return datetime.now(timezone.utc).isoformat()

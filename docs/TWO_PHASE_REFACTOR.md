# Two-Phase Workflow Refactor Guide

Goal: split the monolithic "draw → search → download → embed → rank" flow into
**Phase A: Load an area** (fetch + tile + embed once) and **Phase B: Search**
(instant, repeatable queries over cached embeddings). Additionally: make the map
the main panel of the app, and let users save, list, and reload areas.

---

## 1. Target architecture

```
Phase A (slow, once per area)          Phase B (instant, many times)
─────────────────────────────          ─────────────────────────────
AOI + dates + sensor + res/chip        text or reference-image query
        │                                      │
   load_area()                          encode query (ms)
   ├─ GEE composite                            │
   ├─ grid + parallel download          np.dot over cached embeddings
   ├─ cache tile arrays                        │
   └─ embed_area(model) [lazy]          threshold → NMS → top-k
        │                                      │
   LoadedArea ──── save/load ────►      Grad-CAM from cached pixels
   (session state + disk)               (no re-download)
```

Key invariant that makes this work: **tile embeddings depend only on
(AOI, dates, sensor, bands, resolution, chip_size, model) — never on the
query.** Everything query-dependent is cheap.

A saved area that has embeddings on disk can be searched **without GEE at
all** — loading from disk skips Earth Engine entirely.

---

## 2. New files

### `pipeline/area_store.py` — the LoadedArea model + persistence

```python
@dataclass
class AreaParams:            # everything that identifies an area
    aoi_geojson: dict
    start_date: str
    end_date: str
    sensor: str              # "Sentinel-2" | "Sentinel-1"
    resolution: float
    chip_size: int

@dataclass
class LoadedArea:
    area_id: str             # sha1 of canonical-JSON AreaParams
    name: str
    created_at: str
    params: AreaParams
    tile_bounds: list[tuple] # (minx, miny, maxx, maxy) per tile
    tile_arrays: np.ndarray  # (N, C, H, W) float16, model-ready (post prepare_for_model)
    embeddings: dict[str, np.ndarray]  # model_key -> (N, D) float32, lazy
```

Functions:

- `compute_area_id(params) -> str` — hash of the canonical JSON of `AreaParams`.
  Changing any load parameter ⇒ different id ⇒ different area. Query,
  threshold, and top_k are **not** part of the id.
- `save_area(area, base_dir)` / `load_area_from_disk(area_id)` /
  `list_saved_areas() -> list[meta]` / `delete_area(area_id)`

On-disk layout (add `areas_dir` to `config.py`, default
`~/.embeddedearth/areas/`):

```
areas/<area_id>/
  meta.json            # name, params, aoi geojson, tile bounds, created_at
  tiles.npz            # compressed float16 stack — the disk-heavy part
  emb_dofa_s2.npy      # one file per model_key, written lazily
  emb_dinov3.npy
```

Disk-size note: ~300 S2 tiles at 384×384×10 bands float16 ≈ 0.8 GB compressed
less. Embeddings are negligible (300 × 1152 × 4 B ≈ 1.4 MB). If this becomes a
problem, add a "save embeddings only" toggle later — search stays instant,
only Grad-CAM/thumbnails for a *reloaded* area would need re-download.
Don't build that first; save everything in v1.

### `pipeline/ingest.py` — Phase A logic (extracted from `run_search`)

- `load_area(params: AreaParams, progress_cb) -> LoadedArea`
  Move here, verbatim where possible, from `app/main.py:run_search`:
  - GEE init + AOI→`ee.Geometry` (lines ~405–419)
  - retriever selection + composite + `normalize_for_model` (lines ~421–443)
  - grid generation + MAX_TILES guard (lines ~445–498)
  - the 12-worker download pool (lines ~527–626), **but keep `tile_data`
    instead of dropping it** — stack into `tile_arrays` (float16)
  - Returns a `LoadedArea` with `embeddings={}`.
  - No Streamlit calls inside — report via `progress_cb(msg, frac)` so the
    module stays testable (same pattern as `CopernicusSearchPipeline`).
- `embed_area(area, model_key="dofa") -> np.ndarray`
  Batched encoding (`ENCODE_BATCH` loop, reuse `encode_buffer` logic including
  the `np.isfinite` filter — drop non-finite rows *and their tiles* so indices
  stay aligned). Caches into `area.embeddings[model_key]` and writes the
  `.npy` if the area is saved. Model key must encode the wavelength config
  (e.g. `"dofa_s2"`, `"dofa_s1"`) since S1/S2 embeddings are incompatible.

### `pipeline/semantic_search.py` — Phase B logic

- `search_area(area, query_params, model_bundle) -> (results, diagnostics)`
  Extracted from the second half of `run_search`:
  - query embedding (text or reference image, lines ~507–523)
  - `np.dot`, diagnostics dict, threshold, `nms_results`, top-k
    (lines ~633–683)
  - Grad-CAM loop (lines ~685–746) — **delete the re-download block**
    (lines ~700–715); use `area.tile_arrays[idx].astype(np.float32)` directly.
    RGB thumbnails via the existing `get_rgb_visualization`.
  - Return `(results, diagnostics)`; the caller writes session state.

### `app/components/area_panel.py` — area management UI

Three sections (this lives in the sidebar, see §4):

1. **Load form** (`st.form`): name field (default e.g. "Area 2026-07-21"),
   start/end date, sensor, resolution, chip coverage — i.e. everything being
   *removed* from `search_form.py` — plus a "📥 Load & Embed Area" button.
   On submit: require an AOI, run `load_area` + `embed_area` with a progress
   bar, put the result in `st.session_state.current_area`, offer/auto
   `save_area`.
2. **Current area status**: "✅ *Sonora coast* — 312 tiles · Sentinel-2 ·
   10 m · Jun 1–Jul 21 2026", plus an "Unload" button.
3. **Saved areas list**: `list_saved_areas()`, one row per area with name,
   tile count, sensor, date range, and **Load** / **Delete** buttons.
   Load reads from disk into `current_area` (no GEE needed). Delete asks for
   confirmation via a two-click pattern.

---

## 3. Modified files

### `app/main.py` (biggest change)

- `run_search()` — gutted: body moves to `pipeline/ingest.py` +
  `pipeline/semantic_search.py` as described above.
- `initialize_session_state()` — add `current_area: None`,
  `last_similarities: None`.
- Semantic-search submit handler: replace the `run_search(aoi, params)` call
  with a guard + thin call:

  ```python
  area = st.session_state.current_area
  if area is None:
      st.error("Load an area first (sidebar → Load & Embed Area).")
  else:
      embed_area(area, model_key_for(area.params.sensor))  # no-op if cached
      results, diag = search_area(area, search_params, ...)
  ```

- Layout: see §4.
- Result signature: results/diagnostics reset when `current_area` changes
  (compare `area_id`), not only on new searches.

### `app/components/map_viewer.py`

- Map becomes the main panel: raise default `height` (e.g. 650) and render at
  full container width (already `width=None`).
- New parameter `overlays`: after building `m`, add
  - the **current area** footprint (`folium.GeoJson`, solid outline),
  - each **saved area** footprint (dashed/muted outline,
    `tooltip=name`) so users see what they already have,
  - optionally the current **result bounds** as rectangles.
  `m` is rebuilt every rerun, so overlays need no remount tricks — the nonce
  mechanism stays only for clearing *drawings*.
- On "load saved area", center/zoom the map on its bounds by setting
  `st.session_state.map_center` / `map_zoom` before the widget renders.

### `app/components/search_form.py`

- **Remove** from the form: start/end date, sensor, resolution, chip
  coverage (all move to the area panel — they're load-time parameters).
- **Keep**: search method (text/image), query, reference upload, top_k,
  similarity threshold.
- Move `top_k` and `similarity threshold` **outside the `st.form`**: with
  cached embeddings a rerun costs milliseconds, so these can re-rank live.
  Store the raw `similarities` array in `st.session_state.last_similarities`
  and re-apply threshold → NMS → top-k on every rerun without re-encoding
  anything (skip Grad-CAM regeneration for tiles that already have heatmaps —
  keep a small `{idx: heatmap}` dict per query).
- `SearchParameters`: drop `sensor`, `start_date`, `end_date`, `resolution`,
  `chip_size`; `validate_search_params` loses the date check.

### `app/components/result_grid.py`

- The result-signature reset (selection/export state) must incorporate
  `area_id` + query + threshold so switching areas clears stale selections.

### `config.py`

- Add `areas_dir` (with env override, consistent with `EMBEDDEDEARTH_*`
  convention) and optionally `tile_cache_dtype = "float16"`.

### `pipeline/zero_shot_pipeline.py` and `pipeline/copernicus_pipeline.py` (phase 2, optional)

Both re-fetch imagery today. Adapt them to accept an optional
`area: LoadedArea`:

- **Zero-shot (DINOv3)**: tile features are query-independent → cache under
  `model_key="dinov3"` via the same `embed_area` mechanism (needs a small
  encoder-adapter since it produces patch features, not one vector).
  Reuse `area.tile_arrays` if the loaded bands cover what DINOv3 needs
  (RGB from S2); otherwise fall back to the current fetch path.
- **CopernicusFM**: same pattern, `model_key="copernicus_fm"`. Its query is a
  *geometry* (query area), so only the search-area side is cacheable.
- Don't block the main refactor on these — ship semantic search first and
  leave both tabs on their current code path with a note.

### Tests

- New `tests/test_area_store.py`: `compute_area_id` stability/uniqueness,
  save→load roundtrip (small synthetic arrays), list/delete.
- New `tests/test_ingest_search_split.py`: `search_area` against a fake
  `LoadedArea` with synthetic embeddings — verifies threshold/NMS/top-k
  without GEE or models (fast suite).
- Existing fast tests should be unaffected; `run_search`-related behavior is
  now covered at the module level instead of through Streamlit.

---

## 4. Layout: map as main panel

Replace the current 50/50 `st.columns` split in `render_main_content`:

```
┌─ sidebar ────────────┐ ┌─ main ────────────────────────────────┐
│ 🌍 EmbeddedEarth      │ │  🗺️ MAP (full width, ~650px)          │
│ GEE connect          │ │   · draw AOI                          │
│ ──────────────────── │ │   · saved-area outlines (tooltips)    │
│ 📥 LOAD AREA          │ │   · current-area footprint            │
│  name / dates /      │ │   · result rectangles                 │
│  sensor / res / chip │ ├───────────────────────────────────────┤
│  [Load & Embed]      │ │  🔍 Search bar row (tabs: semantic /   │
│ ──────────────────── │ │     zero-shot / copernicus)           │
│ ✅ Current area info  │ ├───────────────────────────────────────┤
│ ──────────────────── │ │  📊 Result grid + diagnostics          │
│ 💾 Saved areas        │ │                                       │
│  · Sonora [Load][✕]  │ │                                       │
└──────────────────────┘ └───────────────────────────────────────┘
```

- Sidebar (`render_sidebar`): keep GEE connect; add
  `render_area_panel()`; demote the About/Help text into expanders at the
  bottom.
- Main: map full-width on top, then a compact search row (the slimmed
  search form fits horizontally now that dates/sensor are gone), then
  results. The three search tabs stay, but they all share the loaded area.
- The intro/prompting-tips expander stays collapsed above the map.

---

## 5. Migration order (each step leaves the app working)

1. **Pure extraction, no behavior change**: create `pipeline/ingest.py` and
   `pipeline/semantic_search.py` by splitting `run_search`; `run_search`
   becomes `search_area(load_area(...))` glue. Run fast tests + one manual
   search.
2. **Session-level caching (the big win, ~1 day)**: `LoadedArea` in
   `st.session_state`, "Load & Embed" button, search hits the cache. No disk
   persistence yet. Verify: second query on same area returns in <1 s;
   Grad-CAM works from cached pixels (compare a heatmap against the old
   re-download path once).
3. **Persistence + saved areas**: `area_store.py` save/load/list/delete +
   `area_panel.py`. Verify: restart the app, load a saved area, search with
   GEE disconnected.
4. **Layout**: map-as-main-panel restructure + map overlays for saved/current
   areas + live threshold/top_k sliders.
5. **(Optional) zero-shot + CopernicusFM adoption** of the shared area cache.

## 6. Gotchas

- **Index alignment**: after the finite-filter in `embed_area`, embeddings row
  *i* must correspond to `tile_bounds[i]` and `tile_arrays[i]`. Filter all
  three together.
- **S1 vs S2**: sensor is a *load-time* choice now. An S2 area cannot serve an
  S1 search — the UI must make this visible in the current-area status rather
  than silently searching the wrong embeddings.
- **float16 storage**: cast back to float32 before `encode_batch` /
  Grad-CAM. Sanity-check once that float16 roundtrip doesn't move similarity
  scores meaningfully (it shouldn't at [0,1] reflectance scale; the model
  input is what matters, embeddings themselves stay float32).
- **Streamlit reruns**: `LoadedArea` with a few hundred MB of arrays in
  `st.session_state` is fine (in-process object, not serialized), but load it
  once — never rebuild it inside the render path.
- **Grad-CAM correctness**: pass `image_encoder.wavelengths` exactly as today;
  the cached array is post-`prepare_for_model`, same as what was embedded, so
  explanations now explain *precisely* the embedded pixels (today's
  re-download could theoretically differ if GEE returned different data).

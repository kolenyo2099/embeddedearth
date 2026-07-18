# EmbeddedEarth — Function Map & Summary

## 250-Word Summary

EmbeddedEarth is an AI-powered semantic search engine for satellite imagery, built as a Streamlit web app. Users draw a geographic area of interest (AOI) on an interactive map and type a natural language query (e.g., "industrial facility near river") or upload a reference image. The system fetches multispectral satellite data from Google Earth Engine (Sentinel-1 SAR or Sentinel-2 optical), tiling the AOI into overlapping chips (~384x384px each). Every tile is encoded into a 1152-dimensional embedding using DOFA-CLIP — a Vision Transformer fine-tuned on GeoLangBind-2M with wavelength-aware encoding that understands both optical (490–2190 nm) and C-band SAR (~55,500,000 nm). The query is likewise encoded, then cosine similarity (inner product on L2-normalized vectors) ranks all tiles. Results above a configurable threshold are returned as ranked detections with scores. For top results, Grad-ECLIP (a Grad-CAM adaptation for ViTs) generates heatmaps highlighting which visual regions drove each match — gradients w.r.t. the final transformer layer are pooled into per-token weights that multiply activations to produce a spatial importance map. Three search modes exist: semantic text/image search (DOFA-CLIP), zero-shot detection (DINOv3 patch embeddings with centering), and Copernicus Foundation Model similarity search (which ingests spatiotemporal metadata like lat/lon, day-of-year, and patch area alongside the image). Because the tiling grid uses 50% overlap, every pipeline applies non-maximum suppression (pipeline/postprocessing.py) so one hotspot doesn't dominate the ranking as several near-duplicate tiles.

---

## Architecture: Call Graph

```
main()
 └── configure_page(), initialize_session_state(), inject_accessibility_css()
      └── render_sidebar()
           └── GEEClient.initialize()          [user clicks Connect]
      └── render_main_content()
           ├── render_map_viewer()              → returns AOI GeoJSON dict
           │    └── st_folium map + Draw plugin; AOI kept in session state
           │        ("Clear AOI" button remounts the widget)
           │
           ├── [Tab 1: Semantic Search]
           │    ├── render_search_form()         → SearchParameters
           │    ├── validate_search_params()     → (bool, str)
           │    └── run_search(aoi, params)      → List[Dict]  [MAIN PIPELINE]
           │         └── GEEClient.initialize()
           │         ├── Sentinel1Retriever / Sentinel2Retriever.get_composite()
           │         │     └── normalize_for_model()  [server-side scaling]
           │         ├── generate_geo_grid()      → iterator of (bounds, col, row)
           │         ├── [parallel: download_tile_task] per tile (12 workers)
           │         │    ├── download_image_as_array()  [GEE GeoTIFF download + rasterio parse]
           │         │    └── prepare_for_model()        [PIL resize to target_size]
           │         ├── image_encoder.encode_batch()   [main thread, real batches of 32]
           │         │    └── DOFACLIPWrapper.encode_image()
           │         │         └── preprocess_tensor() + vision.trunk(image, waves)
           │         ├── Cosine similarity: np.dot(embeddings, query)
           │         ├── filter by threshold → nms_results() [suppress overlap dupes] → top-k
           │         └── For each winner: re-download + generate_explanation() [Grad-CAM]
           │              └── GradECLIP.generate_gradcam()
           │                   └── hooks on vision.trunk.norm, backward pass
           │
           ├── [Tab 2: Zero-Shot Detection]
           │    ├── render_zero_shot_form()        → ZS parameters
           │    └── run_zero_shot_pipeline()         → List[Dict]
           │         └── GEEClient.initialize()
           │         ├── retriever.get_composite() [ONE composite for the AOI,
           │         │    doubles as the data-availability check]
           │         ├── generate_geo_grid()
           │         ├── [parallel downloads, sequential DINOv3 features]
           │         │    └── DINOv3Wrapper.extract_features(center_features=True)
           │         │         └── cosine similarity → threshold → heatmap resize
           │         └── nms_results() → attention/PCA maps only for survivors
           │
           └── [Tab 3: Copernicus FM]
                ├── render_copernicus_form(aoi)     → CopernicusParameters
                └── CopernicusSearchPipeline.run_search()
                     └── CopernicusFM model with metadata encoding
                          └── shapely intersect filter → cosine similarity
                               → threshold → nms_results()

           [Results Display]
            └── render_result_grid(results)
                 ├── _render_result_card() per result  [shows image, heatmap, score, download buttons]
                 ├── _get_score_color()               [green→red gradient]
                 ├── _download_result()                [saves tile as PNG]
                 └── _render_search_diagnostics_panel() [similarity stats + histogram]

   [Export — from result grid buttons]
    ├── generate_pdf_report()      [PDF with results table, map screenshot, query info]
    ├── generate_geojson()         [FeatureCollection of tile bounding boxes]
    ├── generate_kmz()             [KML/zip for Google Earth]
    └── generate_zip_package()     [all formats bundled]
```

---

## Module: `app/main.py` — Streamlit App Entry Point

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `main()` | None | Orchestrates page config, init, sidebar, and content rendering | `if __name__` |
| `configure_page()` | None | Sets Streamlit page title, icon, layout, menu items | `main()` |
| `initialize_session_state()` | None | Seeds `st.session_state` with defaults (`gee_initialized`, `search_results`, etc.) | `main()` |
| `inject_accessibility_css()` | None | Injects ARIA/accessibility CSS (from `app.accessibility`) | `main()` |
| `render_sidebar()` | None | Sidebar with GEE connect button, help text, keyboard shortcuts | `main()` |
| `render_main_content()` | None | Main layout: map column + 3 search tab columns + results below | `main()` |
| `_render_search_diagnostics_panel()` | None | Collapsible panel showing similarity stats, histogram, top scores | called after `run_search` if results exist |
| `run_search(aoi_geojson, params)` → list | List[Dict] | **Core search pipeline**: fetch imagery → tile → encode → rank → Grad-CAM | `render_main_content()` tab-1 |

---

## Module: `app/components/map_viewer.py` — Interactive Map

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `debug_log(message, data)` | None | Prints debug info (dev only) | internal helpers |
| `initialize_map_state()` | None | Seeds map-related session state variables | `render_map_viewer` |
| `extract_geometry_from_draw_data(draw_data)` → dict/None | Optional[Dict] | Parses draw event data into GeoJSON-like geometry dict | `render_map_viewer` |
| `render_map_viewer()` → dict/None | Optional[Dict] | Renders streamlit-folium map with Draw plugin, captures draw events, persists AOI in session state, offers a Clear-AOI button (remounts the widget via key nonce) | `render_main_content()` |

---

## Module: `app/components/search_form.py` — Semantic Search Form

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `render_search_form(key_prefix)` → SearchParameters | Dataclass | Text input, sensor selector (S1/S2), date range, resolution slider, image upload, submit button | `render_main_content()` tab-1 |
| `validate_search_params(params)` → (bool, str) | Tuple[bool, str] | Checks query not empty, date order valid, image provided if image search type selected | `render_main_content()` tab-1 |

---

## Module: `app/components/result_grid.py` — Results Display

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `render_result_grid(results)` | None | Renders grid of result cards with images, heatmaps, scores, download/export buttons | `render_main_content()` after search completes |
| `_render_result_card(result, index, key_prefix)` | None | Single card: RGB image, heatmap overlay (if any), similarity score bar, export buttons | `render_result_grid` |
| `_get_score_color(score)` → str | str | Returns CSS color string (green=high, red=low) for score bar | `_render_result_card` |
| `_download_result(image, index, key_prefix)` | None | Saves tile image as PNG via Streamlit button | `_render_result_card` |
| `render_no_results()` | None | Shows "no results" message in result grid area | `render_result_grid` |
| `render_loading_state()` | None | Shows spinner/loading indicator | `render_result_grid` |
| `update_loading_progress(progress, status, current, total, message)` | None | Updates progress bar text for ongoing operations | pipeline callers |

---

## Module: `app/components/zero_shot_form.py`

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `render_zero_shot_form()` → dict | Dict | Query image upload, sensor selector, threshold slider, HF token input, submit | `render_main_content()` tab-2 |

---

## Module: `app/components/copernicus_form.py`

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `render_copernicus_form(current_map_aoi)` → CopernicusParameters | Dataclass | Step 1: capture query AOI + search AOI from map; sensor, date range, resolution, threshold selectors | `render_main_content()` tab-3 |

---

## Module: `app/accessibility.py` — ARIA/Screen Reader Support

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `inject_accessibility_css()` | None | Injects CSS for focus indicators, contrast, touch targets, reduced motion | `main()` |
| `announce_to_screen_reader(message)` | None | Writes to Streamlit widget with ARIA-live region for screen reader announcement | `render_main_content()` on search start/end |

---

## Module: `app/export_utils.py` — Export Functionality

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `generate_pdf_report(results, query, aoi_geojson)` | bytes | PDF with results table, AOI map screenshot, query text | `_render_result_card` (export button) |
| `generate_geojson(results)` | bytes | GeoJSON FeatureCollection: each result is a polygon feature with score metadata | `_render_result_card` |
| `generate_kmz(results)` | bytes | KML placemarks zipped as KMZ for Google Earth | `_render_result_card` |
| `generate_zip_package(results, query)` | bytes | Bundles PDF + GeoJSON + KMZ into a single ZIP | `_render_result_card` |

---

## Module: `data/gee_client.py` — GEE Authentication

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `GEEClient.initialize(project_id)` → bool | bool | Authenticates with GEE, sets class-level `_initialized` flag | Multiple: sidebar, run_search, pipelines |
| `GEEClient.is_initialized()` → bool | bool | Checks if GEE is already connected | run_search, pipelines |
| `GEEClient.get_project_id()` → str/None | Optional[str] | Returns current GEE project ID | UI display in sidebar |

---

## Module: `data/preprocessing.py` — Image I/O & Normalization

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `download_image_as_array(image, aoi, bands, scale)` → np.ndarray | ndarray [C,H,W] | Downloads GEE image as GeoTIFF via HTTP, reads with rasterio. Auto-scales up if request too large (recursive fallback) | `run_search`, `CopernicusPipeline.run_search` |
| `normalize_reflectance(data, scale_factor)` → np.ndarray | ndarray [0,1] | Divides raw reflectance by scale factor (10000 for S2), clips to [0,1] | `CopernicusPipeline._prepare_input` for S2 |
| `prepare_for_model(data, target_size)` → np.ndarray | ndarray [C,T,T] | Resizes each channel via PIL BILINEAR to model input size (384) | `run_search` |
| `get_rgb_visualization(data, bands, brightness_factor)` → np.ndarray | ndarray [H,W,3] uint8 | Extracts B4/B3/B2 (or B04/B03/B02) as RGB, applies brightness boost, returns displayable image | `run_search` (for result display), Copernicus pipeline |

---

## Module: `data/sentinel2.py` — Sentinel-2 Retriever

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `Sentinel2Retriever.get_composite(aoi, start_date, end_date)` → ee.Image | ee.Image | Creates cloud-free median composite from S2 collection within AOI + date range | `run_search`, `CopernicusPipeline.run_search` |
| `Sentinel2Retriever.normalize_for_model(image)` → ee.Image | ee.Image | Applies .divide(10000) on server side (raw S2 reflectance is 0-10000 Uint16) | `run_search` (S2 path), `CopernicusPipeline.run_search` (conditional) |
| `Sentinel2Retriever.get_collection(aoi, start, end)` → ee.ImageCollection | ee.ImageCollection | Returns the unprocessed collection (for data availability checks) | `run_zero_shot_pipeline` |

---

## Module: `data/sentinel1.py` — Sentinel-1 Retriever

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `Sentinel1Retriever.get_composite(aoi, start_date, end_date)` → ee.Image | ee.Image | Creates median composite from S1 collection (VV + VH bands); raises ValueError on empty collections | `run_search` (S1 path), zero-shot & Copernicus pipelines |
| `Sentinel1Retriever.normalize_for_model(image)` → ee.Image | ee.Image | Rescales sigma-naught dB values: clips [-25, 0] dB and maps linearly to [0, 1] server-side | `run_search`, `CopernicusPipeline.run_search` |

---

## Module: `pipeline/tiling.py` — Image Tiling & Grid Generation

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `TileGenerator.__init__(tile_size, overlap_ratio)` | None | Sets up stride = tile_size * (1 - overlap) | internal |
| `TileGenerator.generate(image, transform, bounds)` → Iterator[Tile] | Iterator[Tile] | Sliding window tiling with configurable overlap; computes geospatial bounds per tile | `tile_image()` |
| `tile_image(image, tile_size, overlap, transform, bounds)` → list | List[Tile] | Convenience: creates generator and materializes all tiles | utility (pixel-space tiling) |
| `generate_geo_grid(bounds, resolution, tile_size, overlap_ratio)` → Iterator[Tuple[bounds, col, row]] | Iterator[Tuple] | **Core tiling function**: converts geographic bounds to grid of overlapping tile extents in degrees. Uses lat-dependent longitude correction for accuracy. | `run_search`, `run_zero_shot_pipeline`, `CopernicusPipeline.run_search` |

---

## Module: `pipeline/postprocessing.py` — Deduplication & NMS

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `compute_iou(box1, box2)` → float | float | Intersection over Union between two (minx,miny,maxx,maxy) boxes | `nms_results` |
| `nms_results(results, iou_threshold=0.3)` → list | List[Dict] | Geographic NMS on result dicts ({score, bounds, ...}): keeps highest-scoring result per overlap cluster, returns sorted by score. Applied by all three search pipelines because the tiling grid has 50% overlap. | `run_search`, `run_zero_shot_pipeline`, `CopernicusSearchPipeline.run_search` |

---

## Module: `pipeline/zero_shot_pipeline.py` — DINOv3 Zero-Shot Detection

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `run_zero_shot_pipeline(aoi, start_date, end_date, query_vector, sensor, threshold, resolution, hf_token)` → list | List[Dict] | **Zero-shot detection**: build ONE normalized composite for the AOI → tile → parallel downloads (12 workers) → DINOv3 features per tile → cosine similarity to query patch embedding → threshold → heatmap resize via `skimage.transform.resize` → NMS → attention/PCA maps computed only for surviving detections. Returns [{image, heatmap, score, bounds, dino_attention, pca_map}]. | `render_main_content()` tab-2 |
| *(no other named functions)* | — | Entire file is a single pipeline function + inline imports | — |

---

## Module: `pipeline/copernicus_pipeline.py` — Copernicus Foundation Model Search

| Class/Function | Returns | Purpose | Called By |
|----------------|---------|---------|-----------|
| `CopernicusSearchPipeline.__init__(device)` | None | Initializes CopernicusFM model + S1/S2 retrievers | tab-3 pipeline instantiation |
| `_get_meta_info(lon, lat, date, resolution, patch_size)` → tensor | Tensor [1,4] | Constructs metadata: [lon, lat, day_of_year, area_km2] for spatiotemporal encoding | `_prepare_input` |
| `_prepare_input(image_array, bbox, date, sensor, resolution)` → dict | Dict[str,Any] | Normalizes S2 reflectance (S1 already 0-1), resizes to 224x224, builds tensor [1,C,H,W], creates meta_info, converts wavelengths to nm. Returns {x, meta_info, wavelengths, bandwidths} | `run_search` per tile |
| `CopernicusSearchPipeline.run_search(query_geom, search_geom, start_date, end_date, sensor, resolution, threshold, progress_callback)` → list | List[Dict] | **Copernicus FM pipeline**: fetch query image → encode as embedding → tile search area → client-side shapely intersect filter → encode each tile → cosine similarity → threshold filter → NMS. Returns [{image, geometry, score, bounds}] with GeoJSON built client-side. | `render_main_content()` tab-3 |

---

## Module: `models/encoders.py` — Text & Image Encoders

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `TextEncoder.encode(text, normalize)` → np.ndarray | ndarray [B,D] | Tokenizes text, calls `model.encode_text()`, returns embedding vector | `run_search` (line 543: query encoding), `SemanticSearchEngine.search_by_text` |
| `ImageEncoder.encode(images, normalize)` → np.ndarray | ndarray [B,D] | Calls `model.encode_image()` with wavelengths from config or caller-provided tensor | `run_search` (line 584: per-tile encoding) |
| `ImageEncoder.encode_batch(images, batch_size)` → np.ndarray | ndarray [N,D] | Iterates over images in batches, stacks results | `run_search` (line 584) |
| `ImageEncoder.wavelengths` (property) → Tensor | torch.Tensor [N] | Lazy-loads wavelengths via `to_micrometers()` conversion. Supports nm or μm input. | `generate_explanation`, `ImageEncoder.encode` |
| `create_encoders(model, bands, wavelengths)` → (TextEncoder, ImageEncoder) | Tuple | Factory function that creates both encoder instances sharing a model | `run_search` (lines 521-525) |

---

## Module: `models/dofa_clip.py` — DOFA-CLIP Model Wrapper

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `DOFACLIPWrapper._load_model()` | None | Loads model from HF hub via `open_clip.create_model_from_pretrained()`. Sets `_loaded` flag. | All lazy property accessors |
| `DOFACLIPWrapper.preprocess_tensor(images, normalize)` → tensor | Tensor [B,C,H,W] | Converts numpy→torch, handles (H,W,C) layout, resizes to 384x384, applies SigLIP normalization (mean=0.5, std=0.5 → [-1,1]) | `encode_image`, `generate_gradcam`, `verify_explanation_perturbation` |
| `DOFACLIPWrapper.encode_text(text, normalize)` → Tensor | Tensor [B,D] | Tokenizes → `model.encode_text()` → L2-normalize | `TextEncoder.encode`, `GradECLIP.generate_gradcam` (text embedding for similarity) |
| `DOFACLIPWrapper.encode_image(images, wavelengths, normalize)` → Tensor | Tensor [B,D] | Preprocesses → calls `visual.trunk(images, wavelengths)` → L2-normalize. **Core image→embedding function.** | `ImageEncoder.encode`, `verify_explanation_perturbation` |
| `DOFACLIPWrapper.get_visual_tokens(images, wavelengths, normalize)` → Tensor | Tensor [B,N,D] | Same as encode_image but returns pre-pooled per-patch tokens (not global pooled) | `generate_similarity_map` in GradECLIP |
| `get_model()` → DOFACLIPWrapper | DOFACLIPWrapper | Returns singleton model instance (cached globally) | `TextEncoder.model`, `ImageEncoder.model`, `generate_explanation`, main pipeline |

---

## Module: `models/wavelengths.py` — Wavelength Configuration

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `get_wavelengths_for_bands(bands)` → list | List[int] nm | Returns wavelength values in nanometers for given band names | historical/deprecated |
| `get_wavelength_tensor(bands, device)` → Tensor | Tensor [N] | Returns PyTorch tensor of wavelengths (nm) for the band config | `ImageEncoder` lazy property construction |
| `to_micrometers(wavelengths)` → Tensor | Tensor [N] μm | Converts nm tensor to micrometers by dividing by 1000 | `ImageEncoder.wavelengths`, `GradECLIP.generate_gradcam` fallback path |

---

## Module: `xai/grad_eclip.py` — Explainable AI (Grad-CAM)

| Function/Class | Returns | Purpose | Called By |
|----------------|---------|---------|-----------|
| `GradECLIP.__init__(model_wrapper, target_layer)` | None | Registers forward hook (captures activations) and backward hook (captures gradients) on `vision_model.trunk.norm` layer. | `generate_explanation` |
| `GradECLIP._register_hooks()` | None | Attaches `register_forward_hook` and `register_full_backward_hook` to target layer. | `__init__` |
| `GradECLIP.remove_hooks()` | None | Detaches all registered hooks to free memory. | `generate_explanation` (finally block) |
| `GradECLIP._get_layer(layer_path)` → module/None | Optional[nn.Module] | Traverses model attributes via dot-separated path. Supports index-based access like `-1`. | `_register_hooks`, internal |
| `GradECLIP.generate_gradcam(image, text, image_size, wavelengths)` → np.ndarray | ndarray [H,W] heatmap | **Core XAI function**: forward pass with grad enabled on image tensor → encode image + text → cosine similarity score → backward pass → capture gradients/activations from hooks → average gradients over spatial dims to get weights → weight activations → ReLU → reshape to grid (handles CLS/register tokens) → bilinear upsample to image_size → normalize to [0,1]. | `run_search` (per top result), external XAI tools |
| `generate_explanation(model_wrapper, image, text, device, image_size, wavelengths)` → np.ndarray | ndarray [H,W] heatmap | **High-level entry point**: converts numpy image to torch tensor with grad → creates GradECLIP instance → calls generate_gradcam → cleanup. The function called by `run_search`. | `run_search` (line 740) |

---

## Module: `xai/visualization.py` — Heatmap Visualization Helpers

| Function | Returns | Purpose | Called By |
|----------|---------|---------|-----------|
| `create_heatmap_overlay(rgb_image, heatmap, alpha)` → ndarray | ndarray [H,W,3] uint8 | Blends grayscale heatmap over RGB image with configurable transparency (viridis colormap) | downstream UI code |
| `create_colorbar(heatmap_range, height)` → bytes | bytes PNG | Creates a labeled color scale bar for embedding alongside results | result grid display |
| `generate_caption(score, query)` → str | str | Generates a text description like "95% match for 'solar panels'" | UI display helpers |
| `save_figure_to_bytes(fig)` → bytes | bytes | Saves a matplotlib figure to PNG bytes in memory | visualization pipeline |
| `create_comparison_figure(before, after, labels)` → None | None | Side-by-side comparison of two images with annotations | not actively used |

---

## Data Flow: End-to-End (Semantic Search)

```
User draws AOI on map
  → render_map_viewer() returns GeoJSON dict
  → st.session_state.aoi_geojson set (global)

User types query + clicks Search
  → render_search_form() returns SearchParameters
  → validate_search_params() confirms valid input
  → announce_to_screen_reader("Search started")

run_search(aoi, params):
  │
  ├─ GEEClient.initialize()         [authenticate with Google Earth Engine]
  │
  ├─ Sentinel2Retriever.get_composite(aoi, dates)
  │   └── ee.ImageCollection.filterBounds().filterDate().median().clip()
  │   └── .normalize_for_model()   [divide by 10000 on server]
  │
  ├─ generate_geo_grid(bounds, res, chip_size)
  │   └── yields (minx,miny,maxx,maxy), col, row    [sliding window, 50% overlap]
  │
  ├─ for each tile (ThreadPoolExecutor, 12 workers — download only):
  │     download_image_as_array(composite, tile_geom)
  │       └── image.getDownloadUrl(GEO_TIFF) → requests.get() → rasterio.read()
  │     prepare_for_model(tile_data)
  │       └── PIL resize per channel to 384x384
  │
  ├─ image_encoder.encode_batch(buffer)   [main thread, batches of 32]
  │     └── DOFACLIPWrapper.encode_image(images, wavelengths)
  │         └── preprocess_tensor() → SigLIP norm [0,1]→[-1,1]
  │         └── visual.trunk(images, waves)  [ViT forward pass]
  │             └── returns [B, 1152] embeddings
  │
  ├─ similarities = np.dot(tile_embeddings, query_embedding.T).flatten()
  │
  ├─ filter: similarities >= params.similarity_threshold
  │  → nms_results()  [suppress 50%-overlap duplicates]
  │  → take top_k
  │
  └─ for each winning tile (ranked):
       download_image_as_array()     [re-fetch full resolution]
       get_rgb_visualization()       [B4,B3,B2 → RGB display image]
       generate_explanation(model, tile_data, query)   [Grad-CAM heatmap]
         └── GradECLIP hooks on trunk.norm
             └── forward + backward pass
                 └── gradients pooled × activations → ReLU → upsample to 384×384

st.session_state.search_results = results
render_result_grid(results)         [grid of cards with images, heatmaps, scores, export buttons]
_render_search_diagnostics_panel()  [similarity stats + histogram chart]
```

---

## Key Configuration (from `config.py`)

| Config Object | Purpose | Key Values |
|---------------|---------|------------|
| `model_config` | ML model settings | model_name, embedding_dim=1152, image_size=384, device='cpu', batch_size |
| `sentinel2_bands` | S2 band config | band_names=['B2','B3','B4','B8','B11','B12'], scale_factor=10000, wavelengths=[490,560,665,842,1610,2190] nm |
| `sentinel1_bands` | S1 band config | band_names=['VV','VH'], wavelengths=[55500] μm (C-band) |
| `tiling_config` | Tiling defaults | tile_size=384, overlap_ratio=0.5, stride=192 |
| `search_config` | Search defaults | top_k=10, similarity_threshold=0.1 |
| `gee_config` | GEE settings | project_id (user-provided), s2_collection='COPERNICUS/S2_SR_HARMONIZED' |
| `ui_config` | UI display settings | page_title, page_icon, layout='wide', default_center=(40.0, -3.7), default_zoom=6 |

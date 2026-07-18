"""
EmbeddedEarth - Main Streamlit Application

AI-Driven Remote Sensing Semantic Search Engine

This is the main entry point for the Streamlit application.
Run with: streamlit run app/main.py
"""

import streamlit as st
import numpy as np
from datetime import datetime
import io
from PIL import Image

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import ui_config, gee_config

# Import app components
from app.accessibility import inject_accessibility_css, announce_to_screen_reader
from app.components.search_form import render_search_form, validate_search_params
from app.components.map_viewer import render_map_viewer
from app.components.result_grid import render_result_grid


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=ui_config.page_title,
        page_icon=ui_config.page_icon,
        layout=ui_config.layout,
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/ElLocoGIS/docs',
            'Report a bug': 'https://github.com/ElLocoGIS/issues',
            'About': """
            # EmbeddedEarth
            
            AI-Driven Remote Sensing Semantic Search Engine
            
            Search satellite imagery using natural language or visual references.
            Powered by DOFA-CLIP and Google Earth Engine.
            """
        }
    )


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'gee_initialized': False,
        'search_results': None,
        'current_query': None,
        'processing': False,
        'search_diagnostics': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar with info and settings."""
    with st.sidebar:
        st.title("🌍 EmbeddedEarth")
        
        st.markdown("---")
        
        st.markdown("""
        ### About
        
        **EmbeddedEarth** helps you find features in satellite imagery using natural language.
        
        **Powered by:**
        - DOFA-CLIP (Vision-Language Model)
        - Google Earth Engine
        - Sentinel-2 Imagery
        """)
        
        st.markdown("---")
        
        # GEE Status
        if st.session_state.get('gee_initialized'):
            st.success("✅ GEE Connected")
            if st.session_state.get('gee_project_id'):
                st.caption(f"Project: {st.session_state.gee_project_id}")
        else:
            st.warning("⚠️ GEE Not Connected")
            
            project_id_input = st.text_input(
                "GEE Project ID",
                value=gee_config.project_id or "",
                placeholder="e.g., my-gee-project-123",
                help="Required for GEE authentication. Check your Google Cloud Console."
            )
            
            if st.button("🔐 Connect to GEE"):
                try:
                    from data.gee_client import GEEClient
                    # Initialize with provided project ID
                    GEEClient.initialize(project_id=project_id_input)
                    st.session_state.gee_initialized = True
                    st.session_state.gee_project_id = project_id_input
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    
            with st.expander("❓ How to get a Project ID"):
                st.markdown("""
                1. Go to [Google Cloud Console](https://console.cloud.google.com).
                2. Create a new project.
                3. Search for **"Earth Engine API"** and enable it.
                4. Copy the **Project ID** from the dashboard.
                """)
        
        st.markdown("---")
        
        # Help section
        with st.expander("❓ Help"):
            st.markdown("""
            **Keyboard Navigation:**
            - `Tab` - Move between elements
            - `Enter` - Activate buttons
            - `Arrow keys` - Pan map
            - `+/-` - Zoom map
            """)


def render_main_content():
    """Render the main application content."""
    # CSS hack: hide the random '0' if it's a progress bar artifact
    st.markdown("""
        <style>
            .stProgress > div > div > div > div { background-color: transparent; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div id="main-content">', unsafe_allow_html=True)
    
    st.title("🌍 EmbeddedEarth")
    st.caption("Semantic Search for Satellite Imagery")
    
    with st.expander("📖 User Guide & Prompting Tips (Based on Research)", expanded=False):
        st.markdown("""
        ### 🧠 How to Speak "Satellite"
        The AI model (DOFA-CLIP) was trained on the **GeoLangBind-2M** dataset, which pairs satellite images with professional analyst descriptions. To get the best results, try to mimic this style.
        
        #### 1. Use Standard Remote Sensing Vocabulary
        The model understands technical land cover terms better than casual speech.
        *   ✅ **Preferred**: *"High-density residential", "Industrial storage tanks", "Coniferous forest", "Meandering river", "Circular irrigation pivots"*
        *   ❌ **Avoid**: *"Busy town", "Factory place", "Zig-zag water", "Circles"*
        
        #### 2. Describe Spatial Patterns & Geometry
        Satellite analysis is all about how objects are arranged.
        *   ✅ *"**Scattered** trees in a dry field"*
        *   ✅ *"**Clustered** buildings along a linear road"*
        *   ✅ *"**Rectangular** agricultural plots"*
        *   ✅ *"**Grid-like** urban fabric"*

        #### 3. Leverage Material & Texture
        Because the model "sees" spectral wavelengths (not just color), it can distinguish materials.
        *   ✅ *"**Concrete** runway vs **Dirt** road"*
        *   ✅ *"**Metal** warehouse roof vs **Tile** residential roof"*
        *   ✅ *"**Turbid** water vs **Clear** deep water"*

        #### 4. The "Is there...?" Approach
        Phrase your prompt as if you are describing the **answer** to: *"Is there [this feature] in this area?"*
        *   *Example*: "A large coal power plant with multiple cooling towers and coal storage piles."
        
        #### 5. Search by Image (Experimental) 🧪
        You can upload a reference image to find visually similar areas.
        *   ⚠️ **Experimental**: This feature is highly sensitive to resolution differences.
        *   **Limitation**: Matching a high-res Google Maps screenshot (0.5m/px) against Sentinel-2 data (10m/px) may yield unexpected results.
        *   **Best Practice**: Use reference images that match the "blurry" look of Sentinel-2 for best accuracy.
        """)
    
    st.markdown("---")
    
    # Create two-column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Map viewer - returns GeoJSON geometry dict
        aoi = render_map_viewer()
    
    with col_right:

        # Search Tabs
        tab_semantic, tab_zeroshot, tab_copernicus = st.tabs(["💬 Semantic Search", "🎯 Zero-Shot Detection", "🛰️ Copernicus FM"])
        
        # --- TAB 1: Semantic Search ---
        with tab_semantic:
            search_params = render_search_form()
            
            # Debug info
            # print(f"[DEBUG MAIN] search_params.submitted: {search_params.submitted}") # Commented out
            
            # Process search
            if search_params.submitted:
                is_valid, error = validate_search_params(search_params)
                
                # Check Global AOI
                if aoi is None:
                    aoi = st.session_state.get('aoi_geojson')
                
                if not is_valid:
                    st.error(error)
                elif aoi is None:
                    st.error("Please draw an area of interest on the map first.")
                else:
                    announce_to_screen_reader("Search started. Please wait for results.")
                    with st.spinner("🔍 Searching..."):
                        results = run_search(aoi, search_params)
                        st.session_state.search_results = results
                        if search_params.search_type == "image":
                            st.session_state.current_query = "Reference Image Search"
                        else:
                            st.session_state.current_query = search_params.query
                    
                    if results:
                        announce_to_screen_reader(f"Found {len(results)} results.")

        # --- TAB 2: Zero-Shot Detection ---
        with tab_zeroshot:
            from app.components.zero_shot_form import render_zero_shot_form
            from pipeline.zero_shot_pipeline import run_zero_shot_pipeline
            
            zs_params = render_zero_shot_form()
            
            if zs_params and zs_params.get("submitted"):
                # Check Global AOI
                if aoi is None:
                    aoi = st.session_state.get('aoi_geojson')
                    
                if aoi is None:
                     st.error("Please draw an area of interest on the map first.")
                else:
                    with st.spinner("🎯 Running Zero-Shot Detection..."):
                        results = run_zero_shot_pipeline(
                            aoi_geojson=aoi,
                            start_date=zs_params['start_date'].strftime('%Y-%m-%d'),
                            end_date=zs_params['end_date'].strftime('%Y-%m-%d'),
                            query_vector=zs_params['query_vector'],
                            sensor=zs_params['sensor'],
                            threshold=zs_params['threshold'],
                            hf_token=zs_params['token']
                        )
                        st.session_state.search_results = results
                        st.session_state.current_query = "Zero-Shot Pattern"
                        st.session_state.search_diagnostics = None
                        
                        if results:
                             st.success(f"Found {len(results)} matches!")
                        else:
                             st.warning(f"⚠️ No matches found above {zs_params['threshold']:.0%} similarity. Try lowering the threshold or checking your query patch.")

        # --- TAB 3: Copernicus FM ---
        with tab_copernicus:
            try:
                from app.components.copernicus_form import render_copernicus_form
                from pipeline.copernicus_pipeline import CopernicusSearchPipeline
                
                # Pass current map AOI to form for capture
                cop_params = render_copernicus_form(aoi)
                
                if cop_params.submitted:
                    if not cop_params.query_geom:
                        st.error("Please capture a Query Area first (Step 1).")
                    elif not cop_params.search_geom:
                        st.error("Please capture a Search Area first (Step 1).")
                    else:
                        status_container = st.status(f"🛰️ CopernicusFM ({cop_params.sensor})", expanded=True)
                        with status_container:
                            st.write("Initializing pipeline...")
                            
                            def update_progress(msg):
                                st.write(msg)
                                
                            pipeline = CopernicusSearchPipeline()
                            results = pipeline.run_search(
                                query_geom=cop_params.query_geom,
                                search_geom=cop_params.search_geom,
                                start_date=cop_params.start_date.strftime('%Y-%m-%d'),
                                end_date=cop_params.end_date.strftime('%Y-%m-%d'),
                                sensor=cop_params.sensor,
                                resolution=cop_params.resolution,
                                threshold=cop_params.threshold,
                                progress_callback=update_progress
                            )
                            
                            st.write("Search complete!")
                            status_container.update(label="✅ Search Complete", state="complete", expanded=False)

                        st.session_state.search_results = results
                        st.session_state.current_query = f"CopernicusFM ({cop_params.sensor})"
                        st.session_state.search_diagnostics = None
                        
                        if results:
                            st.success(f"Found {len(results)} matches!")
                        else:
                            st.warning("No matches found.")
            except Exception as e:
                st.error(f"Error loading Copernicus FM tab: {e}")
                print(f"[ERROR] Copernicus Tab: {e}")
                import traceback
                traceback.print_exc()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results
    st.markdown("---")
    
    if st.session_state.search_results:
        results = st.session_state.search_results
        render_result_grid(results)
        _render_search_diagnostics_panel()
    else:
        st.info(
            "👋 **Getting Started**\n\n"
            "1. Draw an area of interest on the map\n"
            "2. Enter a search query (e.g., 'solar panels', 'deforestation')\n"
            "3. Click Search to find matching locations"
        )


def _render_search_diagnostics_panel():
    """Render collapsed diagnostics for semantic search pipeline."""
    diag = st.session_state.get('search_diagnostics')
    if not diag:
        return

    with st.expander("🧪 Search Diagnostics", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sensor", diag.get('sensor', 'N/A'))
        c2.metric("Query Type", diag.get('query_type', 'N/A'))
        c3.metric("Query Norm", f"{diag.get('query_norm', 0.0):.4f}")
        c4.metric("Embedding Dim", str(diag.get('embedding_dim', 'N/A')))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Tiles", str(diag.get('total_tiles', 0)))
        c6.metric("Valid Tiles", str(diag.get('valid_tiles', 0)))
        c7.metric("Scored Tiles", str(diag.get('scored_tiles', 0)))
        c8.metric("Above Threshold", str(diag.get('above_threshold', 0)))

        if 'after_nms' in diag:
            st.caption(f"After overlap suppression (NMS): {diag['after_nms']} distinct locations")

        st.markdown(
            f"**Similarity stats**  "
            f"min={diag.get('sim_min', 0.0):.4f}, "
            f"mean={diag.get('sim_mean', 0.0):.4f}, "
            f"median={diag.get('sim_median', 0.0):.4f}, "
            f"max={diag.get('sim_max', 0.0):.4f}, "
            f"std={diag.get('sim_std', 0.0):.4f}"
        )

        hist_counts = diag.get('similarity_histogram_counts', [])
        hist_edges = diag.get('similarity_histogram_edges', [])
        if hist_counts:
            st.caption(
                f"Similarity histogram ({len(hist_counts)} bins). "
                f"Range: {hist_edges[0]:.4f} -> {hist_edges[-1]:.4f}"
            )
            st.bar_chart(np.array(hist_counts, dtype=np.int32))

        top_scores = diag.get('top_scores', [])
        if top_scores:
            score_preview = ", ".join(f"{s:.4f}" for s in top_scores)
            st.caption(f"Top scores: {score_preview}")


def run_search(aoi_geojson: dict, params) -> list:
    """
    Execute the full search pipeline with real satellite imagery.
    
    Pipeline:
    1. Convert AOI GeoJSON to EE Geometry
    2. Fetch Sentinel-2 composite from GEE
    3. Download and tile the imagery
    4. Encode tiles with CLIP
    5. Rank by text query similarity
    6. Generate Grad-CAM explanations for top results
    
    Args:
        aoi_geojson: GeoJSON geometry dict from map drawing.
        params: Search parameters from form.
        
    Returns:
        List of result dicts with 'image', 'heatmap', 'score', 'bounds'.
    """
    import ee

    results = []
    st.session_state.search_diagnostics = None
    
    try:
        # Step 1: Initialize GEE if needed
        from data.gee_client import GEEClient
        if not GEEClient.is_initialized():
            st.info("🔐 Initializing Google Earth Engine...")
            GEEClient.initialize()
            st.session_state.gee_initialized = True
        
        # Step 2: Convert AOI GeoJSON to EE Geometry
        st.info("📍 Processing area of interest...")

        if aoi_geojson.get('type') == 'Polygon':
            aoi_ee = ee.Geometry.Polygon(aoi_geojson['coordinates'])
        else:
            # Generic conversion
            aoi_ee = ee.Geometry(aoi_geojson)

        # Step 3: Fetch imagery for selected sensor
        sensor = getattr(params, "sensor", "Sentinel-2")
        st.info(f"🛰️ Fetching {sensor} imagery from Google Earth Engine...")

        if sensor == "Sentinel-1":
            from data.sentinel1 import Sentinel1Retriever
            from config import sentinel1_bands
            retriever = Sentinel1Retriever()
            bands_to_download = sentinel1_bands.band_names
            encoder_wavelengths = sentinel1_bands.get_wavelength_tensor()
        else:
            from data.sentinel2 import Sentinel2Retriever
            from config import sentinel2_bands
            retriever = Sentinel2Retriever()
            bands_to_download = sentinel2_bands.band_names
            encoder_wavelengths = sentinel2_bands.get_wavelength_tensor()
        
        # Date range from params
        start_date = params.start_date.strftime('%Y-%m-%d') if params.start_date else None
        end_date = params.end_date.strftime('%Y-%m-%d') if params.end_date else None

        composite = retriever.get_composite(aoi_ee, start_date, end_date)
        composite = retriever.normalize_for_model(composite)

        # Step 4: Tile-First Strategy
        st.info("🗺️ Generating search grid...")
        
        from pipeline.tiling import generate_geo_grid, Tile
        from data.preprocessing import download_image_as_array, get_rgb_visualization
        from models.encoders import create_encoders
        
        # Get bounds
        bounds_info = aoi_ee.bounds().getInfo()['coordinates'][0]
        west = min(p[0] for p in bounds_info)
        south = min(p[1] for p in bounds_info)
        east = max(p[0] for p in bounds_info)
        north = max(p[1] for p in bounds_info)
        bounds = (west, south, east, north)
        
        # Generate grid with dynamic resolution (Smart Scaling)
        # Default resolution for Sentinel-2
        # User-defined resolution (Multi-Scale Search)
        target_resolution = params.resolution
        
        # Estimate degrees width/height
        deg_width = east - west
        deg_height = north - south
        
        # Approx meters (at equator, simplistic but safe for estimation)
        meters_width = deg_width * 111320
        meters_height = deg_height * 111320
        
        # Tile size in meters at target res (chip_size controls geo coverage per chip)
        chip_size = getattr(params, 'chip_size', 384)
        tile_m = chip_size * target_resolution
        stride_m = tile_m * 0.5 # 50% overlap
        
        # Estimated tiles (Width / Stride) * (Height / Stride)
        est_cols = max(1, meters_width / stride_m)
        est_rows = max(1, meters_height / stride_m)
        total_est_tiles = est_cols * est_rows
        
        MAX_TILES = 25000

        if total_est_tiles > 5000:
            st.warning(f"⚠️ High-Resolution Search: Generating {int(total_est_tiles)} tiles. This might take a while!")

        if total_est_tiles > MAX_TILES:
             st.error(f"🛑 Too many tiles ({int(total_est_tiles)}). Please reduce the area or increase resolution to >{target_resolution}m.")
             return []

        grid_tiles = list(generate_geo_grid(bounds, resolution=target_resolution, tile_size=chip_size))
        chip_coverage_m = int(chip_size * target_resolution)
        st.write(f"Created grid with {len(grid_tiles)} tiles (Resolution: {target_resolution}m/px, Chip: ~{chip_coverage_m}×{chip_coverage_m}m).")

        if len(grid_tiles) > MAX_TILES:
             st.error("Area is still too big! Please select a smaller region.")
             return []
        
        # Initialize models once
        text_encoder, _ = create_encoders()
        _, image_encoder = create_encoders(
            model=text_encoder.model,
            wavelengths=encoder_wavelengths
        )

        if params.search_type == "image":
            if not params.reference_image:
                st.error("Reference image search selected, but no image was provided.")
                return []

            ref_img = Image.open(io.BytesIO(params.reference_image)).convert("RGB")
            ref_arr = np.asarray(ref_img, dtype=np.float32) / 255.0
            ref_arr = np.transpose(ref_arr, (2, 0, 1))

            # Approximate RGB wavelength mapping (R,G,B in nm).
            _, ref_encoder = create_encoders(
                model=text_encoder.model,
                wavelengths=[665.0, 560.0, 490.0]
            )
            query_embedding = ref_encoder.encode(ref_arr)
        else:
            query_embedding = text_encoder.encode(params.query)

        from data.preprocessing import prepare_for_model

        # Workers only download + preprocess (network/IO-bound); encoding happens
        # on the main thread in real batches so the model's batch dimension is
        # actually used instead of 12 threads contending over batch-of-1 calls.
        def download_tile_task(args):
            idx, t_bounds, col, row = args
            try:
                t_minx, t_miny, t_maxx, t_maxy = t_bounds
                tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])

                # Download (expensive network IO)
                tile_data = download_image_as_array(
                    composite,
                    tile_geom,
                    bands=bands_to_download,
                    scale=target_resolution
                )

                if tile_data.max() == 0:
                    return None

                # Data is already in [0, 1]: composite was normalized server-side by
                # retriever.normalize_for_model() (GEE .divide(scale_factor)).
                # prepare_for_model() only resizes to the model's input resolution.
                tile_data = prepare_for_model(tile_data)

                if tile_data.max() == 0:
                     return None

                # Metadata only — pixel data is dropped after encoding to save RAM
                tile_meta = Tile(
                    x=col * (chip_size // 2),
                    y=row * (chip_size // 2),
                    width=chip_size,
                    height=chip_size,
                    data=None,
                    bounds=t_bounds
                )

                return (tile_meta, tile_data)

            except Exception:
                return None

        # Execute in parallel
        import concurrent.futures
        from config import model_config

        processed_tiles = []
        embedding_chunks = []

        # Max workers: 12 is generally safe for GEE REST API without hitting QPS limits too hard
        MAX_WORKERS = 12
        ENCODE_BATCH = model_config.batch_size

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_tiles = len(grid_tiles)
        completed = 0

        # Prepare args
        task_args = [(i, t[0], t[1], t[2]) for i, t in enumerate(grid_tiles)]

        st.info(f"🚀 Downloading with {MAX_WORKERS} parallel workers, encoding in batches of {ENCODE_BATCH}.")

        def encode_buffer(buffer):
            """Encode buffered (meta, data) pairs; keep only finite embeddings."""
            metas, arrays = zip(*buffer)
            embs = image_encoder.encode_batch(list(arrays))
            finite = np.isfinite(embs).all(axis=1)
            for meta, emb, ok in zip(metas, embs, finite):
                if ok:
                    processed_tiles.append(meta)
                    embedding_chunks.append(emb)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_tile = {executor.submit(download_tile_task, arg): arg for arg in task_args}

            batch_buffer = []
            for future in concurrent.futures.as_completed(future_to_tile):
                result = future.result()
                completed += 1

                # Update UI every 5 tiles to reduce overhead
                if completed % 5 == 0:
                    progress = min(1.0, completed / total_tiles)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing tile {completed}/{total_tiles}...")

                if result:
                    batch_buffer.append(result)
                    if len(batch_buffer) >= ENCODE_BATCH:
                        encode_buffer(batch_buffer)
                        batch_buffer = []

            if batch_buffer:
                encode_buffer(batch_buffer)

        status_text.empty()
        progress_bar.empty()

        if not processed_tiles:
            st.warning("No valid data found in the selected area.")
            return []

        # Stack embeddings
        tile_embeddings = np.vstack(embedding_chunks)
        
        # Step 7: Compute similarities and rank
        st.info("🔍 Ranking results by similarity...")
        
        # Cosine similarity
        similarities = np.dot(tile_embeddings, query_embedding.T).flatten()

        sim_min = float(np.min(similarities))
        sim_max = float(np.max(similarities))
        sim_mean = float(np.mean(similarities))
        sim_median = float(np.median(similarities))
        sim_std = float(np.std(similarities))
        above_threshold = int(np.sum(similarities >= params.similarity_threshold))
        hist_counts, hist_edges = np.histogram(similarities, bins=10)

        st.session_state.search_diagnostics = {
            'source': 'semantic',
            'sensor': sensor,
            'query_type': params.search_type,
            'query_norm': float(np.linalg.norm(query_embedding)),
            'embedding_dim': int(query_embedding.shape[-1]) if query_embedding.ndim > 1 else int(query_embedding.shape[0]),
            'total_tiles': int(total_tiles),
            'valid_tiles': int(len(processed_tiles)),
            'scored_tiles': int(len(similarities)),
            'above_threshold': above_threshold,
            'sim_min': sim_min,
            'sim_mean': sim_mean,
            'sim_median': sim_median,
            'sim_max': sim_max,
            'sim_std': sim_std,
            'similarity_histogram_counts': hist_counts.tolist(),
            'similarity_histogram_edges': hist_edges.tolist(),
            'top_scores': np.sort(similarities)[::-1][:10].tolist(),
            'threshold': float(params.similarity_threshold),
        }
        
        # Filter by threshold, suppress overlapping duplicates (the grid has 50%
        # overlap, so one hotspot shows up in several adjacent tiles), then top-k.
        from pipeline.postprocessing import nms_results

        passing = np.where(similarities >= params.similarity_threshold)[0]
        candidates = [
            {'idx': int(i), 'score': float(similarities[i]), 'bounds': processed_tiles[i].bounds}
            for i in passing
        ]
        deduplicated = nms_results(candidates)
        top_k = min(params.top_k, len(deduplicated))
        top_indices = [c['idx'] for c in deduplicated[:top_k]]

        st.session_state.search_diagnostics['after_nms'] = len(deduplicated)

        # Step 8: Re-fetch and Generate Explanations
        st.info(f"🔥 Fetching full details for top {top_k} matches...")
        
        from models.dofa_clip import get_model
        from xai.grad_eclip import generate_explanation
        from xai.visualization import create_heatmap_overlay # Kept from original
        from data.preprocessing import get_rgb_visualization
        
        model_wrapper = get_model()
        
        for rank, idx in enumerate(top_indices):
            # We need to recover the original tile index from the processed list
            tile = processed_tiles[idx]
            score = float(similarities[idx])

            # Re-download the specific tile data!
            # We need the geometry again
            t_minx, t_miny, t_maxx, t_maxy = tile.bounds
            tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])
            
            # Download again (only for these few winners)
            tile_data = download_image_as_array(
                composite,
                tile_geom,
                bands=bands_to_download,
                scale=target_resolution
            )
            
            # Data is already [0, 1] — composite was normalized via normalize_for_model().
            # prepare_for_model() only resizes.
            tile_data = prepare_for_model(tile_data)
            
            # Update tile with data
            tile.data = tile_data
            
            # Visualization
            rgb_image = get_rgb_visualization(tile.data, bands=bands_to_download)
            
            # Heatmap
            try:
                if params.search_type == "text":
                    # Pass the encoder's wavelength tensor (already in μm, correct sensor)
                    # so Grad-CAM uses the same wavelengths that were used for embedding.
                    heatmap = generate_explanation(
                        model_wrapper,
                        tile.data,
                        params.query,
                        wavelengths=image_encoder.wavelengths
                    )
                else:
                    heatmap = None  # No Grad-CAM for image-reference search
            except Exception as e:
                print(f"[WARN] Grad-CAM failed for tile {idx}: {e}")
                st.warning(f"Explanation unavailable for result {rank+1}: {e}")
                heatmap = None
            
            results.append({
                'image': rgb_image,
                'heatmap': heatmap,
                'score': score,
                'bounds': tile.bounds,
            })
        
        if len(results) == 0:
            st.warning(f"⚠️ No results above similarity threshold ({params.similarity_threshold:.0%}). Try lowering the threshold.")
        
        return results
        
    except Exception as e:
        import traceback
        error_msg = f"Search failed: {e}"
        print(f"[DEBUG SEARCH] ERROR: {error_msg}")
        print(f"[DEBUG SEARCH] Traceback:\n{traceback.format_exc()}")
        st.error(error_msg)
        st.session_state.search_diagnostics = None
        
        # Show debug info
        with st.expander("🔧 Error Details"):
            st.code(traceback.format_exc())
        
        return []


def main():
    """Main application entry point."""
    # Configure page
    configure_page()
    
    # Initialize state
    initialize_session_state()
    
    # Inject accessibility CSS
    inject_accessibility_css()
    
    # Render components
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()

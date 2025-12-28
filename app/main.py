"""
ElLocoGIS - Main Streamlit Application

AI-Driven Remote Sensing Semantic Search Engine

This is the main entry point for the Streamlit application.
Run with: streamlit run app/main.py
"""

import streamlit as st
import numpy as np
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import ui_config, gee_config

# Import app components
from app.accessibility import inject_accessibility_css, add_skip_link, announce_to_screen_reader
from app.components.search_form import render_search_form, validate_search_params
from app.components.map_viewer import render_map_viewer
from app.components.result_grid import render_result_grid, render_loading_state
from pipeline.refinement import get_refiner


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
    # print("[DEBUG MAIN] render_main_content called") # Commented out to reduce noise
    
    # CSS Hacks to clean UI
    st.markdown("""
        <style>
            /* Hide the annoying 'Skip to main content' button */
            a[href="#skip-to-content"] { display: none !important; }
            /* Hide the random '0' if it's a progress bar artifact */
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
        print("[DEBUG MAIN] Calling render_map_viewer...")
        aoi = render_map_viewer()
        print(f"[DEBUG MAIN] render_map_viewer returned: {aoi}")
        print(f"[DEBUG MAIN] aoi type: {type(aoi)}")
        print(f"[DEBUG MAIN] aoi is None: {aoi is None}")
    
    with col_right:
        # Search form
        search_params = render_search_form()
        
        # Debug info
        print(f"[DEBUG MAIN] search_params.submitted: {search_params.submitted}")
        print(f"[DEBUG MAIN] search_params.query: {search_params.query}")
        
        # Process search
        if search_params.submitted:
            print("[DEBUG MAIN] Form was submitted!")
            
            is_valid, error = validate_search_params(search_params)
            print(f"[DEBUG MAIN] Validation: is_valid={is_valid}, error={error}")
            
            # Debug: Check session state for AOI
            print(f"[DEBUG MAIN] Session state keys: {list(st.session_state.keys())}")
            print(f"[DEBUG MAIN] Session state aoi_geojson: {st.session_state.get('aoi_geojson')}")
            
            # Also check session state directly if aoi is None
            if aoi is None:
                aoi = st.session_state.get('aoi_geojson')
                print(f"[DEBUG MAIN] Fallback to session state aoi: {aoi}")
            
            if not is_valid:
                st.error(error)
                print(f"[DEBUG MAIN] Validation failed: {error}")
            elif aoi is None:
                st.error("Please draw an area of interest on the map first.")
                print("[DEBUG MAIN] ERROR: AOI is still None after all checks")
                
                # Extra debug info
                with st.expander("🔧 AOI Debug Info"):
                    st.markdown("**Session State:**")
                    for key in st.session_state.keys():
                        if 'aoi' in key.lower() or 'draw' in key.lower() or 'geo' in key.lower():
                            st.write(f"{key}: {st.session_state[key]}")
            else:
                print(f"[DEBUG MAIN] AOI is valid, proceeding with search. AOI: {aoi}")
                
                # Announce to screen readers
                announce_to_screen_reader("Search started. Please wait for results.")
                
                # Run search
                with st.spinner("🔍 Searching..."):
                    results = run_search(aoi, search_params)
                    st.session_state.search_results = results
                    st.session_state.current_query = search_params.query
                
                if results:
                    announce_to_screen_reader(f"Found {len(results)} results.")
                    print(f"[DEBUG MAIN] Search completed with {len(results)} results")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results
    st.markdown("---")
    
    if st.session_state.search_results:
        results = st.session_state.search_results
        render_result_grid(results)
        
        # Stage 2: Semantic Refinement (After Results)
        if results:
             st.markdown("---")
             with st.expander("✨ Advanced: Super-Resolution Refinement", expanded=False):
                st.markdown("""
                **Micro-Object Detection**:
                1. Upscale top 20 candidate tiles by 4x (10m -> 2.5m).
                2. Re-scan using an RGB-Optimized model (SigLIP).
                """)
                
                # New Prompt for Refinement
                default_prompt = st.session_state.get('current_query', '')
                refine_query = st.text_input(
                    "Refinement Prompt (Be specific, e.g., 'white airplane', 'red roof house')",
                    value=default_prompt,
                    help="Describe the specific object you are looking for inside the candidate tiles."
                )
                
                if st.button("🚀 Run Semantic Refinement (Slow)"):
                     with st.spinner("Initializing Visual Cortex (SigLIP) & Super-Resolution (ESRGAN)..."):
                         refiner = get_refiner()
                         
                     with st.spinner(f"Enhancing candidates & searching for '{refine_query}'..."):
                         # Prepare candidates (idx, image, metadata)
                         candidates = []
                         # Filter out None images just in case
                         valid_results = [r for r in results if r.get('image') is not None]
                         
                         for i, res in enumerate(valid_results[:20]): # Refine top 20 valid
                             img = res['image'] # HWC RGB
                             candidates.append((i, img, None))
                             
                         refined_results = refiner.refine_candidates(candidates, refine_query, top_k=20)
                         
                         # Format for grid
                         display_refined = []
                         for rr in refined_results:
                             # Convert to (H, W, C) uint8 for display
                             # rr.sub_tile.data is (C, H, W) float 0-1
                             vis_img = rr.sub_tile.data.transpose(1, 2, 0)
                             vis_img = (np.clip(vis_img, 0, 1) * 255).astype(np.uint8)
                             
                             display_refined.append({
                                 'image': vis_img,
                                 'score': rr.score,
                                 'bounds': rr.sub_tile.bounds,
                                 'heatmap': None
                             })
                             
                         st.success(f"Found {len(display_refined)} refined micro-targets matching '{refine_query}'!")
                         st.markdown("### 🔬 Refinement Results")
                         render_result_grid(display_refined, show_heatmaps=False, key_prefix="refined")
    else:
        st.info(
            "👋 **Getting Started**\n\n"
            "1. Draw an area of interest on the map\n"
            "2. Enter a search query (e.g., 'solar panels', 'deforestation')\n"
            "3. Click Search to find matching locations"
        )


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
    from datetime import datetime, timedelta
    
    print(f"[DEBUG SEARCH] Starting search pipeline...")
    print(f"[DEBUG SEARCH] AOI GeoJSON: {aoi_geojson}")
    print(f"[DEBUG SEARCH] Query: {params.query}")
    
    results = []
    
    try:
        # Step 1: Initialize GEE if needed
        from data.gee_client import GEEClient
        if not GEEClient.is_initialized():
            st.info("🔐 Initializing Google Earth Engine...")
            GEEClient.initialize()
            st.session_state.gee_initialized = True
        
        # Step 2: Convert AOI GeoJSON to EE Geometry
        st.info("📍 Processing area of interest...")
        print(f"[DEBUG SEARCH] Converting GeoJSON to EE Geometry...")
        
        if aoi_geojson.get('type') == 'Polygon':
            aoi_ee = ee.Geometry.Polygon(aoi_geojson['coordinates'])
        else:
            # Generic conversion
            aoi_ee = ee.Geometry(aoi_geojson)
        
        print(f"[DEBUG SEARCH] EE Geometry created: {aoi_ee.getInfo()}")
        
        # Step 3: Fetch Sentinel-2 imagery
        st.info("🛰️ Fetching Sentinel-2 imagery from Google Earth Engine...")
        
        from data.sentinel2 import Sentinel2Retriever
        retriever = Sentinel2Retriever()
        
        # Date range from params
        start_date = params.start_date.strftime('%Y-%m-%d') if params.start_date else None
        end_date = params.end_date.strftime('%Y-%m-%d') if params.end_date else None
        
        print(f"[DEBUG SEARCH] Date range: {start_date} to {end_date}")
        
        # Save query for Verification tools
        st.session_state.last_query = params.query
        st.session_state.last_search_dates = (start_date, end_date)
        
        composite = retriever.get_composite(aoi_ee, start_date, end_date)
        composite = retriever.normalize_for_model(composite)
        
        print(f"[DEBUG SEARCH] Composite created")
        
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
        
        # Tile size in meters at target res
        tile_m = 384 * target_resolution
        stride_m = tile_m * 0.5 # 50% overlap
        
        # Estimated tiles (Width / Stride) * (Height / Stride)
        est_cols = max(1, meters_width / stride_m)
        est_rows = max(1, meters_height / stride_m)
        total_est_tiles = est_cols * est_rows
        
        if total_est_tiles > 5000:
            st.warning(f"⚠️ High-Resolution Search: Generating {int(total_est_tiles)} tiles. This might take a while!")
        
        if total_est_tiles > 25000:
             st.error(f"🛑 Too many tiles ({int(total_est_tiles)}). Please reduce the area or increase resolution to >{target_resolution}m.")
             return []
        
        grid_tiles = list(generate_geo_grid(bounds, resolution=target_resolution))
        st.write(f"Created grid with {len(grid_tiles)} tiles (Resolution: {target_resolution}m/px).")
        
        if len(grid_tiles) > 20000:
             st.error("Area is still too big! Please select a smaller region.")
             return []
        
        # Initialize models once
        # DEBUG: Force reload to pick up new code/patches
        from models import dofa_clip
        import importlib
        importlib.reload(dofa_clip)
        dofa_clip._model_instance = None
        print("[DEBUG] Forcing model reload (Module Reloaded)...")
        
        text_encoder, image_encoder = create_encoders()
        query_embedding = text_encoder.encode(params.query)
        
        # DEBUG: Check text stats
        print(f"[DEBUG SEARCH] Text embedding stats: min={query_embedding.min():.4f}, max={query_embedding.max():.4f}, mean={query_embedding.mean():.4f}, norm={np.linalg.norm(query_embedding):.4f}")
        
        # Optimize: Define a processing function for parallel execution
        # We process-and-forget: Download -> Encode -> Discard Image -> Keep Embedding
        def process_tile_task(args):
            idx, t_bounds, col, row = args
            try:
                # Create EE geometry
                t_minx, t_miny, t_maxx, t_maxy = t_bounds
                # print(f"[DEBUG WORKER] Tile {idx} bounds: {t_bounds}") # Commented out to reduce noise, enable if needed
                tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])
                
                # Fetch composite
                # Note: We must create a new retriever instance or ensure it's thread-safe?
                # The retriever just calls GEE methods so it should be fine.
                tile_composite = retriever.get_composite(tile_geom, start_date, end_date)
                tile_composite = retriever.normalize_for_model(tile_composite)
                
                # Download (expensive network IO)
                tile_data = download_image_as_array(tile_composite, tile_geom, scale=target_resolution)
                
                if tile_data.max() == 0:
                    return None
                    
                # Double-norm fix: REMOVED manual normalization here.
                # prepare_for_model handles the scaling from 10000 -> 1.
                
                from data.preprocessing import prepare_for_model
                tile_data = prepare_for_model(tile_data)
                
                # DEBUG: Check data stats
                d_min, d_max, d_mean = tile_data.min(), tile_data.max(), tile_data.mean()
                if idx < 5: # Only print first few to avoid spam
                    print(f"[DEBUG DATA] Tile {idx} stats: shape={tile_data.shape}, min={d_min:.4f}, max={d_max:.4f}, mean={d_mean:.4f}")
                
                if d_max == 0:
                     return None
                     
                # Encode (expensive CPU/GPU)
                emb = image_encoder.encode_batch([tile_data])
                
                # Metadata only, NO DATA to save RAM
                tile_meta = Tile(
                    x=col*192,
                    y=row*192,
                    width=384,
                    height=384,
                    data=None, # Process-and-Forget!
                    bounds=t_bounds
                )
                
                return (tile_meta, emb)
                
            except Exception as e:
                # print(f"[WARN] Tile {idx} failed: {e}") 
                return None

        # Execute in parallel
        import concurrent.futures
        
        processed_tiles = []
        tile_embeddings = []
        
        # Max workers: 12 is generally safe for GEE REST API without hitting QPS limits too hard
        MAX_WORKERS = 12
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_tiles = len(grid_tiles)
        completed = 0
        
        # Prepare args
        task_args = [(i, t[0], t[1], t[2]) for i, t in enumerate(grid_tiles)]
        
        st.info(f"🚀 Speeding up... Processing {MAX_WORKERS} tiles in parallel.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all
            future_to_tile = {executor.submit(process_tile_task, arg): arg for arg in task_args}
            
            for future in concurrent.futures.as_completed(future_to_tile):
                result = future.result()
                completed += 1
                
                # Update UI every 5 tiles to reduce overhead
                if completed % 5 == 0:
                    progress = min(1.0, completed / total_tiles)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing tile {completed}/{total_tiles}...")
                
                if result:
                    t_meta, t_emb = result
                    processed_tiles.append(t_meta)
                    tile_embeddings.append(t_emb)

        status_text.empty()
        progress_bar.empty()
        
        if not processed_tiles:
            st.warning("No valid data found in the selected area.")
            return []
            
        # Concatenate embeddings
        tile_embeddings = np.vstack(tile_embeddings)
        
        # Step 7: Compute similarities and rank
        st.info("🔍 Ranking results by similarity...")
        
        # Cosine similarity
        similarities = np.dot(tile_embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_k = min(params.top_k, len(processed_tiles))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
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
            
            print(f"[DEBUG SEARCH] Processing result {rank+1}: tile {idx}, score {score:.3f}")
            
            if score < params.similarity_threshold:
                print(f"[DEBUG SEARCH] Skipping tile {idx}: score {score:.3f} < threshold {params.similarity_threshold}")
                continue
                
            # Re-download the specific tile data!
            # We need the geometry again
            t_minx, t_miny, t_maxx, t_maxy = tile.bounds
            t_minx, t_miny, t_maxx, t_maxy = tile.bounds
            tile_geom = ee.Geometry.Rectangle([t_minx, t_miny, t_maxx, t_maxy])
            
            # Fetch fresh composite
            tile_composite = retriever.get_composite(tile_geom, start_date, end_date)
            tile_composite = retriever.normalize_for_model(tile_composite)
            
            # Download again (only for these few winners)
            tile_data = download_image_as_array(tile_composite, tile_geom, scale=target_resolution)
            
            # REMOVED double normalization check. prepare_for_model does it.
            
            from data.preprocessing import prepare_for_model
            tile_data = prepare_for_model(tile_data)
            
            # Update tile with data
            tile.data = tile_data
            
            # Visualization
            rgb_image = get_rgb_visualization(tile.data)
            
            # Heatmap
            try:
                heatmap = generate_explanation(model_wrapper, tile.data, params.query)
            except Exception as e:
                print(f"[WARN] Grad-CAM failed for tile {idx}: {e}")
                heatmap = np.ones((224, 224)) * score
            
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

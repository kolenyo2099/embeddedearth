"""
EmbeddedEarth - Main Streamlit Application

AI-Driven Remote Sensing Semantic Search Engine

This is the main entry point for the Streamlit application.
Run with: streamlit run app/main.py

Two-phase workflow: Phase A ("Load & Embed Area" in the sidebar) fetches +
tiles + embeds an AOI once; Phase B (the search tabs below the map) queries
the cached embeddings instantly, as many times as you like, without
re-touching Google Earth Engine.
"""

import streamlit as st
import numpy as np

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import ui_config, gee_config

# Import app components
from app.accessibility import inject_accessibility_css, announce_to_screen_reader
from app.theme import inject_theme_css, render_hero, render_section
from app.components.search_form import render_search_form, validate_search_params
from app.components.map_viewer import render_map_viewer
from app.components.result_grid import render_result_grid
from app.components.area_panel import render_area_panel
from pipeline.ingest import embed_area, model_key_for_sensor, MODEL_FAMILIES
from pipeline.semantic_search import ModelBundle, search_area, rank_candidates, explain_candidates


def _family_short(model: str) -> str:
    """Short model name, e.g. 'DOFA-CLIP'."""
    return MODEL_FAMILIES.get(model, {}).get("label", model).split(" — ")[0]


def _tab_gate(area, expected_family: str, tab_name: str) -> bool:
    """Gate a search tab on the loaded area's model family.

    An area is embedded with a single model, so only its matching tab can search
    it. Returns True when this tab may run; otherwise renders a message
    explaining what to load and returns False.
    """
    want = _family_short(expected_family)
    if area is None:
        st.info(
            f"👋 Load an area with the **{want}** model first "
            f"(sidebar → **Load & Embed Area**) to use {tab_name}."
        )
        return False

    actual = getattr(area.params, "model", "dofa")
    if actual != expected_family:
        st.warning(
            f"🔒 **{area.name}** was embedded with **{_family_short(actual)}**, so it "
            f"can only be searched in that model's tab. To use {tab_name}, load an "
            f"area with the **{want}** model (sidebar → Load & Embed Area)."
        )
        return False

    return True


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
        'current_area': None,
        'last_similarities': None,
        'last_search_params': None,
        'heatmap_cache': {},
        'last_result_bounds': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar: GEE connect, area panel, then About/Help."""
    with st.sidebar:
        st.title("EmbeddedEarth")

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

        render_area_panel()

        st.markdown("---")

        with st.expander("ℹ️ About"):
            st.markdown("""
            **EmbeddedEarth** helps you find features in satellite imagery using natural language.

            **Powered by:**
            - DOFA-CLIP (Vision-Language Model)
            - Google Earth Engine
            - Sentinel-1 / Sentinel-2 Imagery
            """)

        with st.expander("❓ Help"):
            st.markdown("""
            **Workflow:**
            1. Draw an AOI on the map.
            2. Pick an embedding model + Load & Embed Area (sidebar) — fetches
               and embeds imagery once. The model you pick decides which search
               tab you can use (DOFA-CLIP → Semantic, DINOv3 → Zero-Shot,
               CopernicusFM → Copernicus FM).
            3. Search as many times as you like below the map — instant, no GEE.

            **Keyboard Navigation:**
            - `Tab` - Move between elements
            - `Enter` - Activate buttons
            - `Arrow keys` - Pan map
            - `+/-` - Zoom map
            """)


def render_main_content():
    """Render the main application content: map on top, search below."""
    render_hero()

    with st.expander("User Guide & Prompting Tips", expanded=False):
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

    # Map is the main panel now — full width, on top.
    aoi = render_map_viewer()

    if aoi is None:
        aoi = st.session_state.get('aoi_geojson')

    area = st.session_state.get('current_area')

    render_section("02", "Search", note="Query the embedded area")
    tab_semantic, tab_zeroshot, tab_copernicus = st.tabs(["Semantic Search", "Zero-Shot Detection", "Copernicus FM"])

    # --- TAB 1: Semantic Search (DOFA-CLIP) ---
    with tab_semantic:
        if not _tab_gate(area, "dofa", "Semantic Search"):
            search_params = None
        else:
            st.caption(f"Searching **{area.name}** — {area.num_tiles} tiles · {area.params.sensor}")

            search_params = render_search_form()

        if search_params is not None and search_params.submitted:
            is_valid, error = validate_search_params(search_params)

            if not is_valid:
                st.error(error)
            else:
                announce_to_screen_reader("Search started. Please wait for results.")
                with st.spinner("🔍 Searching..."):
                    results, diag = run_semantic_search(area, search_params)
                    st.session_state.search_results = results
                    st.session_state.search_diagnostics = diag
                    st.session_state.last_search_params = search_params
                    st.session_state.current_query = (
                        "Reference Image Search" if search_params.search_type == "image" else search_params.query
                    )
                    st.session_state.last_result_bounds = [{'bounds': r['bounds'], 'score': r.get('score', 0.0)} for r in results if r.get('bounds')]

                if results:
                    announce_to_screen_reader(f"Found {len(results)} results.")
        elif search_params is not None and area is not None \
                and st.session_state.get('last_similarities') is not None \
                and st.session_state.get('last_search_params') is not None:
            # Live re-rank: threshold/top_k changed without a submit click —
            # re-apply threshold -> NMS -> top-k against the cached similarity
            # array instead of re-encoding anything.
            live_params = st.session_state.last_search_params
            if (live_params.top_k != search_params.top_k
                    or live_params.similarity_threshold != search_params.similarity_threshold):
                candidates = rank_candidates(
                    area,
                    st.session_state.last_similarities,
                    search_params.similarity_threshold,
                    search_params.top_k,
                )
                model_bundle = build_model_bundle(area)
                results = explain_candidates(
                    area, candidates, search_params, model_bundle,
                    heatmap_cache=st.session_state.setdefault('heatmap_cache', {}),
                )
                st.session_state.search_results = results
                st.session_state.last_search_params = search_params
                st.session_state.last_result_bounds = [{'bounds': r['bounds'], 'score': r.get('score', 0.0)} for r in results if r.get('bounds')]

    # --- TAB 2: Zero-Shot Detection (DINOv3) ---
    with tab_zeroshot:
        from app.components.zero_shot_form import render_zero_shot_form
        from pipeline.zero_shot_pipeline import run_zero_shot_pipeline

        if _tab_gate(area, "dinov3", "Zero-Shot Detection"):
            zs_params = render_zero_shot_form(area=area)

            if zs_params and zs_params.get("submitted"):
                with st.spinner("🎯 Running Zero-Shot Detection..."):
                    results = run_zero_shot_pipeline(
                        query_vector=zs_params['query_vector'],
                        threshold=zs_params['threshold'],
                        hf_token=zs_params['token'],
                        area=area,
                    )
                    st.session_state.search_results = results
                    st.session_state.current_query = "Zero-Shot Pattern"
                    st.session_state.search_diagnostics = None
                    st.session_state.last_result_bounds = [{'bounds': r['bounds'], 'score': r.get('score', 0.0)} for r in results if r.get('bounds')]

                    if results:
                        st.success(f"Found {len(results)} matches!")
                    else:
                        st.warning(f"⚠️ No matches found above {zs_params['threshold']:.0%} similarity. Try lowering the threshold or checking your query patch.")

    # --- TAB 3: Copernicus FM ---
    with tab_copernicus:
        try:
            from app.components.copernicus_form import render_copernicus_form
            from pipeline.copernicus_pipeline import CopernicusSearchPipeline

            if not _tab_gate(area, "copernicus", "Copernicus FM"):
                cop_params = None
            else:
                cop_params = render_copernicus_form(aoi, area=area)

            if cop_params is not None and cop_params.submitted:
                if not cop_params.query_geom:
                    st.error("Please capture a Query Area first (Step 1).")
                else:
                    def _as_date_str(value):
                        return value.strftime('%Y-%m-%d') if hasattr(value, 'strftime') else value

                    status_container = st.status(f"🛰️ CopernicusFM ({cop_params.sensor})", expanded=True)
                    with status_container:
                        st.write("Initializing pipeline...")

                        def update_progress(msg):
                            st.write(msg)

                        pipeline = CopernicusSearchPipeline()
                        results = pipeline.run_search(
                            query_geom=cop_params.query_geom,
                            search_geom=cop_params.search_geom,
                            search_area=cop_params.search_area,
                            start_date=_as_date_str(cop_params.start_date),
                            end_date=_as_date_str(cop_params.end_date),
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
                    st.session_state.last_result_bounds = [{'bounds': r['bounds'], 'score': r.get('score', 0.0)} for r in results if r.get('bounds')]

                    if results:
                        st.success(f"Found {len(results)} matches!")
                    else:
                        st.warning("No matches found.")
        except Exception as e:
            st.error(f"Error loading Copernicus FM tab: {e}")
            print(f"[ERROR] Copernicus Tab: {e}")
            import traceback
            traceback.print_exc()

    # Display results
    render_section("03", "Results")

    if st.session_state.search_results:
        results = st.session_state.search_results
        last_params = st.session_state.get('last_search_params')
        render_result_grid(
            results,
            area_id=getattr(area, 'area_id', None),
            query=st.session_state.get('current_query'),
            threshold=getattr(last_params, 'similarity_threshold', None),
        )
        _render_search_diagnostics_panel()
    else:
        st.info(
            "👋 **Getting Started**\n\n"
            "1. Draw an area of interest on the map\n"
            "2. Load & Embed Area in the sidebar (once per area)\n"
            "3. Enter a search query and hit Search — re-run as many times as you like"
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


def build_model_bundle(area) -> ModelBundle:
    """Build encoders matching the loaded area's sensor wavelengths."""
    from config import sentinel1_bands, sentinel2_bands
    from models.encoders import create_encoders
    from models.dofa_clip import get_model

    wavelengths = (
        sentinel1_bands.get_wavelength_tensor()
        if area.params.sensor == "Sentinel-1"
        else sentinel2_bands.get_wavelength_tensor()
    )
    text_encoder, image_encoder = create_encoders(wavelengths=wavelengths)
    model_wrapper = get_model()
    return ModelBundle(text_encoder=text_encoder, image_encoder=image_encoder, model_wrapper=model_wrapper)


def run_semantic_search(area, params) -> tuple:
    """
    Phase B: query an already-loaded, already-embedded area.

    embed_area() is a no-op if this area/sensor combo was already embedded
    (e.g. loaded from disk with cached embeddings), so this never touches
    GEE — only np.dot + NMS + Grad-CAM from cached pixels.
    """
    model_key = model_key_for_sensor(area.params.sensor)
    embed_area(area, model_key=model_key)

    model_bundle = build_model_bundle(area)
    heatmap_cache = st.session_state.setdefault('heatmap_cache', {})
    # New query -> stale per-tile heatmaps from the previous query don't apply.
    heatmap_cache.clear()

    results, diagnostics = search_area(area, params, model_bundle, heatmap_cache=heatmap_cache)
    st.session_state.last_similarities = diagnostics.pop('similarities', None)

    if len(results) == 0:
        st.warning(f"⚠️ No results above similarity threshold ({params.similarity_threshold:.0%}). Try lowering the threshold.")

    return results, diagnostics


def main():
    """Main application entry point."""
    configure_page()
    initialize_session_state()
    inject_accessibility_css()
    inject_theme_css()  # after accessibility so theme colors win

    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()

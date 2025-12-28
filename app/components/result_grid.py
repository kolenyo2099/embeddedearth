"""
Result Grid Component

Displays search results in an accessible grid layout
with heatmap overlays and download options.
"""

import streamlit as st
import numpy as np
from typing import List, Optional
from io import BytesIO
from PIL import Image

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import ui_config
from xai.visualization import create_heatmap_overlay, generate_caption


def render_result_grid(
    results: List[dict],
    show_heatmaps: bool = True,
    columns: int = None,
    key_prefix: str = "res"
):
    """
    Render search results in an accessible grid.
    
    Args:
        results: List of result dicts.
        show_heatmaps: Whether to show heatmap overlays.
        columns: Number of columns (default from config).
        key_prefix: Unique prefix for streamlit element keys.
    """
    if not results:
        st.info("No results to display. Try a different query or expand your search area.")
        return
    
    columns = columns or ui_config.results_per_row
    
    st.markdown("### 🎯 Search Results")
    
    # Header row with count and master heatmap toggle
    header_col1, header_col2 = st.columns([2, 1])
    with header_col1:
        st.markdown(f"Found **{len(results)}** matching tiles")
    with header_col2:
        # Master heatmap toggle (stored in session state)
        master_heatmap_key = f"{key_prefix}_master_heatmap"
        master_heatmap = st.toggle(
            "🔥 Show All Heatmaps", 
            value=st.session_state.get(master_heatmap_key, False),
            key=master_heatmap_key,
            help="Toggle heatmap overlay on all results"
        )
        show_heatmaps = master_heatmap  # Override with master toggle
    
    # Interpretation Guide
    with st.expander("ℹ️ How to interpret these results", expanded=False):
        st.markdown("""
        **Similarity Score**: 
        - Shows how semantically close the image is to your text query.
        - **>20%**: Strong match.
        - **<10%**: Weak match.
        
        **Heatmap Overlay**:
        - Use the "Show All Heatmaps" toggle above to enable/disable all heatmaps.
        - You can also toggle individual heatmaps on each card.
        - **Red/Yellow Regions**: Parts of the image that contributed most to the match.
        - **Blue/Transparent**: Irrelevant background.
        """)
    
    # Create grid
    for row_start in range(0, len(results), columns):
        row_results = results[row_start:row_start + columns]
        cols = st.columns(columns)
        
        for idx, (col, result) in enumerate(zip(cols, row_results)):
            global_idx = row_start + idx
            
            with col:
                _render_result_card(
                    result=result,
                    index=global_idx,
                    default_heatmap_on=show_heatmaps,
                    key_prefix=key_prefix
                )


def _render_result_card(
    result: dict,
    index: int,
    default_heatmap_on: bool = False,
    key_prefix: str = "res"
):
    """
    Render a single result card.
    
    Args:
        result: Result dict.
        index: Result index.
        default_heatmap_on: Default state for heatmap toggle.
        key_prefix: Unique key prefix.
    """
    image = result.get('image')
    heatmap = result.get('heatmap')
    score = result.get('score', 0.0)
    bounds = result.get('bounds')
    
    if image is None:
        st.warning("No image data")
        return
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
    # Controls Row: Score and Heatmap Toggle
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # Score badge
        score_color = _get_score_color(score)
        st.markdown(
            f'<div style="background-color: {score_color}; '
            f'padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold; '
            f'text-align: center; display: inline-block;">'
            f'{score:.1%} match</div>',
            unsafe_allow_html=True
        )
        
    show_heatmap = default_heatmap_on  # Start with master toggle state
    with c2:
        if heatmap is not None:
            # Individual toggle (but master override takes precedence)
            individual_toggle = st.toggle("Heatmap", value=default_heatmap_on, key=f"{key_prefix}_heatmap_{index}")
            # If master is ON, always show. Otherwise use individual toggle.
            show_heatmap = default_heatmap_on or individual_toggle
    
    # Apply heatmap if enabled
    if show_heatmap and heatmap is not None:
        display_image = create_heatmap_overlay(image, heatmap)
    else:
        display_image = image
    
    # Generate caption
    caption = generate_caption(score, bounds, index)
    
    # Display image
    st.image(
        display_image,
        caption=caption,
        use_container_width=True
    )
    
    # Actions Row: Maps and Download
    c_act1, c_act2 = st.columns([1, 1])
    
    with c_act1:
        # Google Maps Link
        if bounds:
            minx, miny, maxx, maxy = bounds
            center_lon = (minx + maxx) / 2
            center_lat = (miny + maxy) / 2
            maps_url = f"https://www.google.com/maps/search/?api=1&query={center_lat},{center_lon}&z=16"
            st.markdown(f'[📍 Google Maps]({maps_url})')
            
    with c_act2:
        _download_result(image, index, key_prefix)


def _get_score_color(score: float) -> str:
    """Get color based on similarity score."""
    if score >= 0.25:
        return "#16a34a"  # Green
    elif score >= 0.15:
        return "#ca8a04"  # Dark Yellow
    else:
        return "#ea580c"  # Orange


def _download_result(image: np.ndarray, index: int, key_prefix: str):
    """Create downloadable image."""
    img_pil = Image.fromarray(image)
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    
    st.download_button(
        label="📥 Click to Save",
        data=buffer,
        file_name=f"result_{index + 1}.png",
        mime="image/png",
        key=f"{key_prefix}_dl_btn_{index}"
    )


def render_no_results():
    """Display message when no results are found."""
    st.warning(
        """
        ### No Matching Results
        
        Try the following:
        - Use different keywords in your query
        - Expand your search area
        - Lower the similarity threshold
        - Choose a different date range
        """
    )


def render_loading_state():
    """Display loading state during search."""
    with st.spinner("🔍 Searching satellite imagery..."):
        progress = st.progress(0)
        status = st.empty()
        
        return progress, status


def update_loading_progress(
    progress,
    status,
    current: int,
    total: int,
    message: str = "Processing tiles..."
):
    """Update loading progress display."""
    pct = current / total if total > 0 else 0
    progress.progress(pct)
    status.text(f"{message} ({current}/{total})")

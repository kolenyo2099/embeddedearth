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
from app.export_utils import generate_pdf_report, generate_kmz, generate_geojson, generate_zip_package

def render_result_grid(
    results: List[dict],
    show_heatmaps: bool = True,
    columns: int = None,
    key_prefix: str = "res"
):
    """
    Render search results in an accessible grid with export capabilities.
    """
    if not results:
        st.info("No results to display. Try a different query or expand your search area.")
        return
    
    columns = columns or ui_config.results_per_row
    
    # --- State Management for Selection ---
    selection_key = f"{key_prefix}_selection"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = set(range(len(results))) # Default all selected
        
    selected_indices = st.session_state[selection_key]
    
    # Validation: Ensure indices are within bounds of current results
    # This prevents IndexError if results list shrinks between searches
    valid_indices = {i for i in selected_indices if i < len(results)}
    if len(valid_indices) != len(selected_indices):
        selected_indices = valid_indices
        st.session_state[selection_key] = selected_indices
    
    # --- Header & Controls ---
    st.markdown("### 🎯 Search Results")
    
    control_col1, control_col2, control_col3 = st.columns([1.5, 1, 1])
    
    with control_col1:
        st.markdown(f"Found **{len(results)}** tiles. Selected: **{len(selected_indices)}**")
        
    with control_col2:
        # Selection Controls
        if st.button("Select All", key=f"{key_prefix}_sel_all"):
            # Update the main tracking set
            st.session_state[selection_key] = set(range(len(results)))
            # Force update of individual widget keys to reflect in UI
            for i in range(len(results)):
                st.session_state[f"{key_prefix}_chk_{i}"] = True
            st.rerun()
            
        if st.button("Deselect All", key=f"{key_prefix}_desel_all"):
            st.session_state[selection_key] = set()
            # Force update of individual widget keys to reflect in UI
            for i in range(len(results)):
                st.session_state[f"{key_prefix}_chk_{i}"] = False
            st.rerun()
            
    with control_col3:
         # Master heatmap toggle
        # Master heatmap toggle
        master_heatmap_key = f"{key_prefix}_master_heatmap"
        show_heatmaps = st.toggle(
            "🔥 Show Heatmaps", 
            value=st.session_state.get(master_heatmap_key, True), # Default True
            key=master_heatmap_key
        )
        
        # Heatmap Type Selector (New)
        # If DINO attention is available in the first result, show the option
        has_attn = results and 'dino_attention' in results[0]
        heatmap_mode = "Similarity"
        if has_attn:
             heatmap_mode = st.radio(
                 "Visualization Mode",
                 ["Similarity Scan", "DINO Attention"],
                 horizontal=True,
                 key=f"{key_prefix}_vis_mode"
             )
        
    # --- Export Menu ---
    with st.expander("📤 Export Options", expanded=False):
        st.caption("Select format to download the SELECTED results.")
        
        # Filter results based on selection
        export_subset = [results[i] for i in sorted(list(selected_indices))]
        
        if not export_subset:
            st.warning("⚠️ No results selected for export.")
        else:
            exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
            
            # Retrieve query from session state for reports
            current_query = st.session_state.get('current_query', "Unknown Query")
            
            with exp_c1:
                if st.button("📄 PDF Report"):
                    with st.spinner("Generating PDF..."):
                        pdf_bytes = generate_pdf_report(export_subset, current_query)
                        st.download_button(
                            "⬇️ Download PDF", 
                            data=bytes(pdf_bytes), 
                            file_name=f"mission_report.pdf", 
                            mime="application/pdf"
                        )
            
            with exp_c2:
                if st.button("🌍 Google Earth (KMZ)"):
                    with st.spinner("Generating KMZ..."):
                        kmz_bytes = generate_kmz(export_subset)
                        st.download_button(
                            "⬇️ Download KMZ",
                            data=kmz_bytes,
                            file_name="results.kmz",
                            mime="application/vnd.google-earth.kmz"
                        )
                        
            with exp_c3:
                if st.button("🗺️ GIS (GeoJSON)"):
                     json_str = generate_geojson(export_subset)
                     st.download_button(
                         "⬇️ Download GeoJSON",
                         data=json_str,
                         file_name="results.geojson",
                         mime="application/geo+json"
                     )
            
            with exp_c4:
                if st.button("💻 Raw Data (ZIP)"):
                     with st.spinner("Zipping files..."):
                         zip_bytes = generate_zip_package(export_subset, current_query)
                         st.download_button(
                             "⬇️ Download ZIP",
                             data=zip_bytes,
                             file_name="raw_data.zip",
                             mime="application/zip"
                         )

    # Interpretation Guide
    with st.expander("ℹ️ How to interpret these results", expanded=False):
        st.markdown("...") # Kept brief for diff clarity, can stay properly
    
    # --- Grid Rendering ---
    for row_start in range(0, len(results), columns):
        row_results = results[row_start:row_start + columns]
        cols = st.columns(columns)
        
        for idx, (col, result) in enumerate(zip(cols, row_results)):
            global_idx = row_start + idx
            
            with col:
                # Check if selected
                is_selected = global_idx in selected_indices
                
                # Render Selection Checkbox ABOVE the card content or as part of it
                # Native checkboxes inside columns work fine
                
                # Capture selection change
                new_selected = st.checkbox(
                    f"Select Result #{global_idx+1}", 
                    value=is_selected, 
                    key=f"{key_prefix}_chk_{global_idx}",
                    label_visibility="visible" 
                )
                
                # Update state immediately if changed (requires rerun usually, but st handles it)
                if new_selected != is_selected:
                    if new_selected:
                        selected_indices.add(global_idx)
                    else:
                        selected_indices.discard(global_idx)
                    # We might need to force rerun if we want the "Selected count" to update instantly
                    # For now rely on next interact
                
                _render_result_card(
                    result=result,
                    index=global_idx,
                    default_heatmap_on=show_heatmaps,
                    key_prefix=key_prefix,
                    is_selected=new_selected,
                    heatmap_mode=heatmap_mode
                )


def _render_result_card(
    result: dict,
    index: int,
    default_heatmap_on: bool = False,
    key_prefix: str = "res",
    is_selected: bool = True,
    heatmap_mode: str = "Similarity"
):
    """
    Render a single result card.
    """
    image = result.get('image')
    
    # Select heatmap based on mode
    if "Attention" in heatmap_mode:
        heatmap = result.get('dino_attention')
    elif "PCA" in heatmap_mode:
        # PCA returns a 3-channel RGB image, not a heatmap
        heatmap = result.get('pca_map')
    else:
        heatmap = result.get('heatmap')
        
    score = result.get('score', 0.0)
    bounds = result.get('bounds')
    
    if image is None: return
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
    # Highlight border if selected? (Streamlit styling hard, but maybe markdown text)
    if is_selected:
        st.markdown("✅ **Selected**")
    
    # Controls Row: Score
    c1, c2 = st.columns([1, 1])
    with c1:
        score_color = _get_score_color(score)
        st.markdown(
            f'<div style="background-color: {score_color}; padding: 4px; border-radius: 4px; color: white;">{score:.1%}</div>',
            unsafe_allow_html=True
        )
    
    show_heatmap = default_heatmap_on
    with c2:
        if heatmap is not None:
            show_heatmap = st.toggle("Heatmap", value=default_heatmap_on, key=f"{key_prefix}_heatmap_{index}") or default_heatmap_on
    
    # Apply heatmap
    display_image = create_heatmap_overlay(image, heatmap) if (show_heatmap and heatmap is not None) else image
    
    # Caption
    caption = generate_caption(score, bounds, index)
    
    st.image(display_image, caption=caption, use_column_width=True)
    
    # Actions (Maps + Single DL)
    c_act1, c_act2 = st.columns([1, 1])
    with c_act1:
        if bounds:
           # Maps link code...
           minx, miny, maxx, maxy = bounds
           center_lat = (miny + maxy) / 2
           center_lon = (minx + maxx) / 2
           maps_url = f"https://www.google.com/maps/search/?api=1&query={center_lat},{center_lon}"
           st.markdown(f'[📍 Map]({maps_url})')
           
    with c_act2:
        _download_result(image, index, key_prefix)


def _get_score_color(score: float) -> str:
    """Get color based on similarity score."""
    if score >= 0.25: return "#16a34a"
    elif score >= 0.15: return "#ca8a04"
    else: return "#ea580c"


def _download_result(image: np.ndarray, index: int, key_prefix: str):
    """Create downloadable image."""
    img_pil = Image.fromarray(image)
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    
    st.download_button(
        label="📥 Png",
        data=buffer,
        file_name=f"result_{index + 1}.png",
        mime="image/png",
        key=f"{key_prefix}_dl_btn_{index}"
    )

def render_no_results():
    """Display message when no results are found."""
    st.warning("No Matching Results found. Try adjusting filters.")

def render_loading_state():
    """Display loading state."""
    with st.spinner("🔍 Searching..."):
        return st.progress(0), st.empty()

def update_loading_progress(progress, status, current, total, message="Processing..."):
    pct = current / total if total > 0 else 0
    progress.progress(pct)
    status.text(f"{message} ({current}/{total})")

"""
Map Viewer Component

Interactive map using folium/streamlit-folium for AOI selection,
with session state persistence.

Set the environment variable EMBEDDEDEARTH_DEBUG=1 to enable verbose
console logging and the in-app debug expander.
"""

import os
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import ui_config

DEBUG = os.getenv("EMBEDDEDEARTH_DEBUG", "").lower() in ("1", "true", "yes")


def debug_log(message: str, data: Any = None):
    """Log debug messages to the console when EMBEDDEDEARTH_DEBUG is set."""
    if not DEBUG:
        return
    print(f"[DEBUG MAP] {message}")
    if data is not None:
        print(f"[DEBUG MAP] Data: {data}")


def initialize_map_state():
    """Initialize map-related session state variables."""
    defaults = {
        'map_center': list(ui_config.default_center),
        'map_zoom': ui_config.default_zoom,
        'drawn_features': None,
        'aoi_geojson': None,
        'last_draw_data': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def extract_geometry_from_draw_data(draw_data: Dict) -> Optional[Dict]:
    """
    Extract geometry from streamlit-folium draw data.
    
    Args:
        draw_data: Data returned by st_folium.
        
    Returns:
        GeoJSON geometry dict or None.
    """
    if draw_data is None:
        return None

    # Check for all_drawings (list of all drawn features)
    all_drawings = draw_data.get('all_drawings')

    if all_drawings and len(all_drawings) > 0:
        # Use the last drawn feature
        last_feature = all_drawings[-1]
        if 'geometry' in last_feature:
            return last_feature['geometry']

    if all_drawings == []:
        # The user deleted every drawing; don't resurrect a stale geometry
        # from last_active_drawing.
        return None

    # Check for last_active_drawing
    last_active = draw_data.get('last_active_drawing')
    if last_active and 'geometry' in last_active:
        return last_active['geometry']

    return None


def render_map_viewer(
    height: int = 500,
    key: str = "main_map"
) -> Optional[Dict]:
    """
    Render interactive map for AOI selection.
    
    Uses streamlit-folium with Draw plugin for reliable geometry capture.
    
    Args:
        height: Map height in pixels.
        key: Unique key for the map widget.
        
    Returns:
        GeoJSON geometry dict if user has drawn an AOI, None otherwise.
    """
    initialize_map_state()

    st.markdown("### 🗺️ Select Area of Interest")
    st.markdown(
        "**Instructions:** Use the rectangle or polygon tools on the left to draw your search area."
    )

    # Create base map with folium
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="OpenStreetMap"
    )
    
    # Add Draw control with specific options
    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': True,
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'rectangle': True,
        },
        edit_options={
            'edit': True,
            'remove': True,
        }
    )
    draw.add_to(m)
    
    # Render map and capture interactions.
    # The widget key includes a nonce so "Clear AOI" can remount the map with
    # an empty drawing layer (folium drawings can't be cleared in place).
    nonce_key = f"{key}_nonce"
    nonce = st.session_state.setdefault(nonce_key, 0)
    output = st_folium(
        m,
        height=height,
        width=None,  # Full width
        key=f"{key}_{nonce}",
        returned_objects=["all_drawings", "last_active_drawing"],
    )

    debug_log("st_folium output:", output)

    # Store the raw output for debugging
    st.session_state.last_draw_data = output

    # Extract geometry
    geometry = extract_geometry_from_draw_data(output)

    if geometry:
        st.session_state.aoi_geojson = geometry
    elif output is not None and output.get('all_drawings') == []:
        # User deleted every drawing with the map's trash tool: the stored AOI
        # is stale and must be cleared, not silently reused.
        st.session_state.aoi_geojson = None

    # Use session state geometry if available
    aoi = st.session_state.get('aoi_geojson')

    # Debug expander (opt-in via EMBEDDEDEARTH_DEBUG)
    if DEBUG:
        with st.expander("🔧 Debug Info (click to expand)"):
            st.markdown("**Session State:**")
            st.json({
                'aoi_geojson': st.session_state.get('aoi_geojson'),
                'drawn_features': st.session_state.get('drawn_features'),
            })

            st.markdown("**Last Draw Data:**")
            if output:
                st.json(output)
            else:
                st.write("No draw data")

    # Show status
    if aoi is None:
        st.info("👆 Draw a rectangle or polygon on the map to select your search area.")
    else:
        status_col, clear_col = st.columns([3, 1])
        with status_col:
            st.success("✅ Area of interest selected!")
        with clear_col:
            if st.button("🗑️ Clear AOI", key=f"{key}_clear_aoi"):
                st.session_state.aoi_geojson = None
                st.session_state[nonce_key] = nonce + 1
                st.rerun()

        # Show bounds
        try:
            if aoi.get('type') == 'Polygon':
                coords = aoi['coordinates'][0]
                lons = [p[0] for p in coords]
                lats = [p[1] for p in coords]
                st.markdown(
                    f"**Bounds:** W: {min(lons):.4f}°, E: {max(lons):.4f}°, "
                    f"S: {min(lats):.4f}°, N: {max(lats):.4f}°"
                )
        except Exception as e:
            debug_log(f"Error displaying bounds: {e}")

    return aoi

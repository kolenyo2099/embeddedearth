"""
Map Viewer Component

Interactive map using folium/streamlit-folium for AOI selection,
with session state persistence and extensive debugging.

DEBUG VERSION - Console logging enabled
"""

import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import json
from typing import Optional, Tuple, Dict, Any

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import ui_config


def debug_log(message: str, data: Any = None):
    """
    Log debug messages to both console and Streamlit expander.
    """
    print(f"[DEBUG MAP] {message}")
    if data is not None:
        print(f"[DEBUG MAP] Data: {data}")


def initialize_map_state():
    """Initialize map-related session state variables."""
    debug_log("Initializing map state...")
    
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
            debug_log(f"  Initialized {key} = {value}")
        else:
            debug_log(f"  Existing {key} = {st.session_state[key]}")


def extract_geometry_from_draw_data(draw_data: Dict) -> Optional[Dict]:
    """
    Extract geometry from streamlit-folium draw data.
    
    Args:
        draw_data: Data returned by st_folium.
        
    Returns:
        GeoJSON geometry dict or None.
    """
    debug_log("Extracting geometry from draw data...")
    debug_log("Full draw_data:", draw_data)
    
    if draw_data is None:
        debug_log("  draw_data is None")
        return None
    
    # Check for all_drawings (list of all drawn features)
    all_drawings = draw_data.get('all_drawings')
    debug_log(f"  all_drawings: {all_drawings}")
    
    if all_drawings and len(all_drawings) > 0:
        # Use the last drawn feature
        last_feature = all_drawings[-1]
        debug_log(f"  Using last feature: {last_feature}")
        
        if 'geometry' in last_feature:
            geometry = last_feature['geometry']
            debug_log(f"  Extracted geometry: {geometry}")
            return geometry
    
    # Check for last_active_drawing
    last_active = draw_data.get('last_active_drawing')
    debug_log(f"  last_active_drawing: {last_active}")
    
    if last_active and 'geometry' in last_active:
        geometry = last_active['geometry']
        debug_log(f"  Using last_active geometry: {geometry}")
        return geometry
    
    # Check for last_object_clicked_popup (might have geometry)
    last_clicked = draw_data.get('last_object_clicked')
    debug_log(f"  last_object_clicked: {last_clicked}")
    
    debug_log("  No geometry found in draw data")
    return None


def geojson_to_ee_geometry(geojson: Dict):
    """
    Convert GeoJSON geometry to Earth Engine Geometry.
    
    Args:
        geojson: GeoJSON geometry dict.
        
    Returns:
        ee.Geometry or None.
    """
    debug_log("Converting GeoJSON to EE Geometry...")
    debug_log("Input GeoJSON:", geojson)
    
    try:
        import ee
        
        geom_type = geojson.get('type')
        coords = geojson.get('coordinates')
        
        debug_log(f"  Geometry type: {geom_type}")
        debug_log(f"  Coordinates: {coords}")
        
        if geom_type == 'Polygon':
            ee_geom = ee.Geometry.Polygon(coords)
        elif geom_type == 'Rectangle':
            ee_geom = ee.Geometry.Rectangle(coords)
        elif geom_type == 'Point':
            ee_geom = ee.Geometry.Point(coords)
        else:
            debug_log(f"  Unknown geometry type: {geom_type}")
            # Try generic constructor
            ee_geom = ee.Geometry(geojson)
        
        debug_log(f"  Created EE Geometry: {ee_geom}")
        return ee_geom
        
    except Exception as e:
        debug_log(f"  Error converting to EE: {e}")
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
    debug_log("=" * 50)
    debug_log("render_map_viewer called")
    debug_log(f"  height={height}, key={key}")
    
    initialize_map_state()
    
    st.markdown("### 🗺️ Select Area of Interest")
    st.markdown(
        "**Instructions:** Use the rectangle or polygon tools on the left to draw your search area."
    )
    
    # Create base map with folium
    debug_log("Creating folium Map...")
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="OpenStreetMap"
    )
    
    # Add Draw control with specific options
    debug_log("Adding Draw control...")
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
    
    # Render map and capture interactions
    debug_log("Rendering map with st_folium...")
    output = st_folium(
        m,
        height=height,
        width=None,  # Full width
        key=key,
        returned_objects=["all_drawings", "last_active_drawing"],
    )
    
    debug_log("st_folium output:", output)
    
    # Store the raw output for debugging
    st.session_state.last_draw_data = output
    
    # Extract geometry
    geometry = extract_geometry_from_draw_data(output)
    
    if geometry:
        st.session_state.aoi_geojson = geometry
        debug_log("Geometry saved to session state")
    
    # Use session state geometry if available
    aoi = st.session_state.get('aoi_geojson')
    
    # Debug expander
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
        debug_log("No AOI - returning None")
    else:
        st.success("✅ Area of interest selected!")
        
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
        
        debug_log(f"Returning AOI: {aoi}")
    
    return aoi


def get_ee_geometry_from_aoi(aoi_geojson: Dict):
    """
    Convert stored AOI GeoJSON to Earth Engine Geometry.
    
    Args:
        aoi_geojson: GeoJSON geometry dict.
        
    Returns:
        ee.Geometry or None.
    """
    if aoi_geojson is None:
        return None
    return geojson_to_ee_geometry(aoi_geojson)


def create_bounds_geometry(
    bounds: Tuple[float, float, float, float]
):
    """
    Create ee.Geometry from bounds tuple.
    
    Args:
        bounds: (west, south, east, north) in degrees.
        
    Returns:
        ee.Geometry.Rectangle.
    """
    import ee
    west, south, east, north = bounds
    return ee.Geometry.Rectangle([west, south, east, north])

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
from typing import List, Optional, Dict, Any, Tuple

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import ui_config
from app.components.result_grid import get_score_color

DEBUG = os.getenv("EMBEDDEDEARTH_DEBUG", "").lower() in ("1", "true", "yes")


def debug_log(message: str, data: Any = None):
    """Log debug messages to the console when EMBEDDEDEARTH_DEBUG is set."""
    if not DEBUG:
        return
    print(f"[DEBUG MAP] {message}")
    if data is not None:
        print(f"[DEBUG MAP] Data: {data}")


def _fix_folium_tiles():
    """Reload the tiles Leaflet skips when st_folium grows its iframe late.

    st_folium first renders the map iframe narrow (~300px), then grows it to
    full width a beat later. Leaflet loads tiles for the initial narrow size
    and doesn't refetch, leaving a gray band on the right. st_folium strips
    scripts added to the folium object, so this runs from a sibling component
    that reaches into the map iframe and calls invalidateSize() — debounced on
    an iframe ResizeObserver so it fires once *after* the width settles (calling
    it mid-transition just latches Leaflet to an intermediate width).
    """
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
        (function(){
          function find(){
            var frames = window.parent.document.querySelectorAll('iframe');
            for (var i=0;i<frames.length;i++){
              try{
                var cw = frames[i].contentWindow;
                for (var k in cw){
                  var o = cw[k];
                  if (o && o._container && typeof o.invalidateSize === 'function'){
                    return {frame: frames[i], map: o};
                  }
                }
              }catch(e){}
            }
            return null;
          }
          var t, tries = 0;
          function attach(){
            var f = find();
            if (!f){ if (++tries < 40) setTimeout(attach, 150); return; }
            var fix = function(){ try{ f.map.invalidateSize(true); }catch(e){} };
            try{
              new ResizeObserver(function(){ clearTimeout(t); t = setTimeout(fix, 250); })
                .observe(f.frame);
            }catch(e){}
            setTimeout(fix, 500); setTimeout(fix, 1500); setTimeout(fix, 3000);
          }
          attach();
        })();
        </script>
        """,
        height=0,
    )


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


def _add_area_overlays(m: folium.Map):
    """Add current-area (solid) and saved-area (dashed) footprints, plus
    current result bounds as rectangles, so users see what they already have
    without needing the sidebar. Rebuilt every rerun — no remount needed
    since these aren't user-editable drawings."""
    current_area = st.session_state.get("current_area")
    current_geojson = current_area.params.aoi_geojson if current_area is not None else None
    if current_area is not None:
        try:
            folium.GeoJson(
                current_geojson,
                name="Current area",
                style_function=lambda _: {
                    "color": "#2563eb", "weight": 3, "fillOpacity": 0.05,
                },
                tooltip=current_area.name,
            ).add_to(m)
        except Exception:
            pass

    # Pending selection: the user has drawn a shape but not yet loaded it as a
    # current_area. The live Leaflet.Draw shape doesn't survive the reruns that
    # happen during ingest, so overlay it ourselves (amber, dashed) to keep the
    # box visible while the area loads.
    pending_aoi = st.session_state.get("aoi_geojson")
    if pending_aoi is not None and pending_aoi != current_geojson:
        try:
            folium.GeoJson(
                pending_aoi,
                name="Selected area",
                style_function=lambda _: {
                    "color": "#f59e0b", "weight": 3, "dashArray": "8,4", "fillOpacity": 0.05,
                },
                tooltip="Selected area (pending load)",
            ).add_to(m)
        except Exception:
            pass

    try:
        from pipeline.area_store import list_saved_areas
        current_id = getattr(current_area, "area_id", None)
        for meta in list_saved_areas():
            if meta["area_id"] == current_id:
                continue  # already drawn solid above
            try:
                folium.GeoJson(
                    meta["params"]["aoi_geojson"],
                    name=meta["name"],
                    style_function=lambda _: {
                        "color": "#94a3b8", "weight": 2, "dashArray": "5,5", "fillOpacity": 0,
                    },
                    tooltip=meta["name"],
                ).add_to(m)
            except Exception:
                continue
    except Exception:
        pass

    scored_bounds = st.session_state.get("last_result_bounds") or []
    max_score = max((sb.get("score", 0.0) for sb in scored_bounds), default=0.0)
    high_scale = max_score > 0.45
    for sb in scored_bounds:
        try:
            minx, miny, maxx, maxy = sb["bounds"]
            color = get_score_color(sb.get("score", 0.0), high_scale)
            folium.Rectangle(
                bounds=[(miny, minx), (maxy, maxx)],
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.25,
            ).add_to(m)
        except Exception:
            continue


def render_map_viewer(
    height: int = 650,
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

    # The map (and its pending-AOI overlay) is built below from the AOI as it
    # stands at the *start* of this rerun. If st_folium then reports a changed
    # drawing, the on-screen map is stale — rerun once so the overlay catches
    # up. Captured here, compared after extraction.
    aoi_at_render_start = st.session_state.get("aoi_geojson")

    from app.theme import render_section
    render_section("01", "Area of Interest")
    if st.session_state.get("current_area") is None:
        st.caption("Use the rectangle or polygon tools on the map to draw your search area.")

    # Create base map with folium
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="OpenStreetMap"
    )

    _add_area_overlays(m)

    # Add Draw control with specific options
    draw = Draw(
        draw_options={
            'polyline': False,
            # repeatMode=False so the tool disarms after one shape completes;
            # otherwise Leaflet.Draw re-arms on mouse-release and immediately
            # starts drawing another shape.
            'polygon': {'repeatMode': False},
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'rectangle': {'repeatMode': False},
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
        width=None,  # full container width
        key=f"{key}_{nonce}",
        returned_objects=["all_drawings", "last_active_drawing"],
    )

    _fix_folium_tiles()

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

    # The map was drawn with aoi_at_render_start; if the drawing changed this
    # rerun, rebuild so the pending-AOI overlay reflects it immediately (not one
    # interaction late). Loop-free: after the rerun both values match.
    if aoi != aoi_at_render_start:
        st.rerun()

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
    current_area = st.session_state.get("current_area")
    if aoi is None and current_area is None:
        st.info("👆 Draw a rectangle or polygon on the map to select your search area.")
    elif aoi is None and current_area is not None:
        st.success(
            f"✅ Using loaded area **{current_area.name}** (outlined in blue on the map). "
            "Draw a new shape here only if you want to load a different area."
        )
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
                    '<span class="ee-eyebrow">Bounds</span>&nbsp; '
                    f'<span class="ee-readout">W {min(lons):.4f}° · '
                    f'E {max(lons):.4f}° · S {min(lats):.4f}° · N {max(lats):.4f}°</span>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            debug_log(f"Error displaying bounds: {e}")

    return aoi

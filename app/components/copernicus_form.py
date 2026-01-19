import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config import gee_config

@dataclass
class CopernicusParameters:
    query_geom: Optional[Dict] = None
    search_geom: Optional[Dict] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    sensor: str = "Sentinel-2"
    resolution: float = 10.0
    threshold: float = 0.5
    submitted: bool = False

def render_copernicus_form(current_map_aoi: Optional[Dict]) -> CopernicusParameters:
    """
    Render form for CopernicusFM.
    Allows capturing Map AOI as Query or Search area.
    """
    
    st.markdown("### 🛰️ Copernicus Foundation Model")
    st.markdown("Feature extraction and similarity search using CopernicusFM.")
    
    # Initialize session state for this form if needed
    if 'copernicus_query_geom' not in st.session_state:
        st.session_state.copernicus_query_geom = None
    if 'copernicus_search_geom' not in st.session_state:
        st.session_state.copernicus_search_geom = None
        
    params = CopernicusParameters()
    
    # Area Selection UI
    st.info("Step 1: Draw on the map, then capture as Query or Search area.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Query Area** (Pattern to find)")
        if st.button("📍 Capture Map as Query Area"):
            if current_map_aoi:
                st.session_state.copernicus_query_geom = current_map_aoi
                st.success("Captured!")
            else:
                st.error("Draw on map first!")
        
        if st.session_state.copernicus_query_geom:
            st.success("✅ Query Area Set")
            # Maybe show bounds or area?
        else:
            st.warning("⚠️ Not Set")

    with col2:
        st.markdown("**Search Area** (Where to look)")
        if st.button("🗺️ Capture Map as Search Area"):
            if current_map_aoi:
                st.session_state.copernicus_search_geom = current_map_aoi
                st.success("Captured!")
            else:
                st.error("Draw on map first!")
                
        if st.session_state.copernicus_search_geom:
            st.success("✅ Search Area Set")
        else:
            st.warning("⚠️ Not Set")
            
    st.divider()
    
    # Form for other parameters
    with st.form("copernicus_fm_form"):
        # Sensor Selector
        sensor = st.selectbox(
            "Sensor",
            options=["Sentinel-2", "Sentinel-1"],
            help="Choose between Optical (Sentinel-2) or Radar (Sentinel-1)."
        )
        params.sensor = sensor
        
        # Date inputs
        c1, c2 = st.columns(2)
        with c1:
            default_start = datetime.now() - timedelta(days=gee_config.default_days_back)
            start_date = st.date_input("Start Date", value=default_start)
        with c2:
            end_date = st.date_input("End Date", value=datetime.now())
            
        params.start_date = start_date
        params.end_date = end_date
        

        # Resolution
        resolution = st.slider(
            "Resolution (m/px)",
            min_value=10.0,
            max_value=60.0,
            value=10.0,
            step=10.0,
            help="Resolution for analysis. 10m is standard for S2/S1."
        )
        params.resolution = resolution
        
        # Similarity Threshold (Replicating SearchForm logic)
        threshold_pct = st.slider(
            "Minimum Match Confidence (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            format="%d%%",
            help="Minimum similarity percentage. Tiles below this score will be filtered out."
        )
        params.threshold = threshold_pct / 100.0
        
        submitted = st.form_submit_button("🚀 Run Copernicus Search", type="primary")
        params.submitted = submitted
        
    # Populate params with stored geoms
    params.query_geom = st.session_state.copernicus_query_geom
    params.search_geom = st.session_state.copernicus_search_geom
    
    return params

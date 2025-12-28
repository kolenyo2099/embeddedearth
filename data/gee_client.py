"""
Google Earth Engine Client Module

Handles authentication, initialization, and provides geemap integration
for the Streamlit frontend.
"""

import ee
import geemap
import streamlit as st
from typing import Optional
from functools import lru_cache

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import gee_config


class GEEClient:
    """
    Google Earth Engine client wrapper.
    
    Manages authentication and provides a singleton connection to GEE.
    Supports both interactive and service account authentication.
    """
    
    _initialized: bool = False
    _project_id: Optional[str] = None
    
    @classmethod
    def initialize(cls, project_id: Optional[str] = None) -> bool:
        """
        Initialize Earth Engine.
        
        Args:
            project_id: GEE project ID. If None, uses config or prompts.
            
        Returns:
            True if initialization successful.
        """
        if cls._initialized:
            return True
        
        project = project_id or gee_config.project_id
        
        try:
            # Try to initialize with existing credentials
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            
            cls._initialized = True
            cls._project_id = project
            return True
            
        except ee.EEException:
            # Need to authenticate first
            try:
                ee.Authenticate()
                if project:
                    ee.Initialize(project=project)
                else:
                    ee.Initialize()
                
                cls._initialized = True
                cls._project_id = project
                return True
                
            except Exception as e:
                raise RuntimeError(f"Failed to authenticate with GEE: {e}")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if GEE is initialized."""
        return cls._initialized
    
    @classmethod
    def get_project_id(cls) -> Optional[str]:
        """Get the current project ID."""
        return cls._project_id
    
    @classmethod
    def get_image_data(
        cls,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
        cloud_cover: int = 20
    ) -> Optional[dict]:
        """
        Fetch multi-band image data as numpy array for inference.
        
        Args:
            geometry: Area of Interest (AOI).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            cloud_cover: Maximum cloud cover percentage.
            
        Returns:
            Dictionary containing 'data' (numpy array) and 'transform' (affine).
            Returns None if no valid data found.
        """
        if not cls._initialized:
            cls.initialize()
            
        # Get config
        from config import gee_config, sentinel2_bands
        
        # 1. Filter Collection
        s2 = ee.ImageCollection(gee_config.s2_collection) \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)) \
            .select(sentinel2_bands.band_names)
            
        # 2. Composite (Median) to reduce clouds/noise
        image = s2.median().clip(geometry)
        
        # 3. Check if image has data
        # We'll try to export a small region; if it fails/is empty, return None.
        # Ideally, we handle this more robustly, but for now:
        try:
            # Get region bounds
            region = geometry.bounds().getInfo()['coordinates']
            
            # Request pixel data
            # scale=10 means 10m resolution (Sentinel-2 native)
            # format='NPY' returns a structured array we can convert to numpy
            # Note: For large areas, this might hit limits. We assume tiled requests in pipeline.
            url = image.getDownloadURL({
                'scale': 10,
                'crs': 'EPSG:4326',
                'filePerBand': False,
                'region': region,
                'format': 'NPY'
            })
            
            # Download and parse
            import requests
            import numpy as np
            import io
            
            response = requests.get(url)
            response.raise_for_status()
            
            # NPY format from GEE is a zipped file containing .npy per band or combined
            # Actually, 'NPY' format might not be standard in getDownloadURL for all endpoints.
            # Using 'GEO_TIFF' is safer for preservation of georeference info.
            
            # Let's switch to GeoTIFF download for safer parsing with rasterio
             # (which preserves spatial metadata needed for tiling)
             
            return cls._download_as_geotiff(image, region)
            
        except Exception as e:
            print(f"Error fetching GEE data: {e}")
            return None

    @classmethod
    def _download_as_geotiff(cls, image, region):
        """Helper to download and parse GeoTIFF from GEE."""
        import requests
        import rasterio
        import io
        import numpy as np
        
        url = image.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:4326',
            'filePerBand': False,
            'region': region,
            'format': 'GEO_TIFF'
        })
        
        response = requests.get(url)
        response.raise_for_status()
        
        with rasterio.open(io.BytesIO(response.content)) as src:
            data = src.read()  # (C, H, W)
            transform = src.transform
            nodata = src.nodata
            
            # Handle nodata (replace with 0 or NaN)
            if nodata is not None:
                data = np.where(data == nodata, 0, data)
                
            return {
                'data': data,
                'transform': transform,
                'crs': src.crs,
                'bounds': src.bounds
            }


def initialize_gee_for_streamlit() -> bool:
    """
    Initialize GEE for Streamlit with proper session state handling.
    
    Returns:
        True if initialized successfully.
    """
    if 'gee_initialized' not in st.session_state:
        st.session_state.gee_initialized = False
    
    if not st.session_state.gee_initialized:
        try:
            GEEClient.initialize()
            st.session_state.gee_initialized = True
        except RuntimeError as e:
            st.error(f"Failed to initialize Google Earth Engine: {e}")
            return False
    
    return True


@lru_cache(maxsize=1)
def create_map(center: tuple = None, zoom: int = None) -> geemap.Map:
    """
    Create a geemap Map instance.
    
    Args:
        center: (lat, lon) tuple for map center.
        zoom: Initial zoom level.
        
    Returns:
        Configured geemap.Map instance.
    """
    from config import ui_config
    
    center = center or ui_config.default_center
    zoom = zoom or ui_config.default_zoom
    
    m = geemap.Map(
        center=center,
        zoom=zoom,
        add_google_map=False,  # Use OSM for accessibility
    )
    
    # Add basemap options
    m.add_basemap("OpenStreetMap")
    
    return m


def get_streamlit_map(
    center: tuple = None,
    zoom: int = None,
    height: int = 600
) -> None:
    """
    Display a geemap in Streamlit.
    
    Uses session state to persist map state across reruns.
    
    Args:
        center: (lat, lon) tuple for map center.
        zoom: Initial zoom level.
        height: Map height in pixels.
    """
    # Initialize GEE if needed
    if not initialize_gee_for_streamlit():
        st.warning("Please authenticate with Google Earth Engine to continue.")
        return
    
    # Get or create map
    if 'map' not in st.session_state:
        st.session_state.map = create_map.__wrapped__(center, zoom)
    
    # Render map
    st.session_state.map.to_streamlit(height=height)


# Convenience function for AOI extraction
def get_drawn_geometry(m: geemap.Map) -> Optional[ee.Geometry]:
    """
    Extract drawn geometry from a geemap instance.
    
    Args:
        m: geemap.Map instance with drawing controls.
        
    Returns:
        ee.Geometry if user has drawn something, None otherwise.
    """
    if hasattr(m, 'user_roi') and m.user_roi is not None:
        return m.user_roi
    
    if hasattr(m, 'draw_last_feature') and m.draw_last_feature is not None:
        return ee.Feature(m.draw_last_feature).geometry()
    
    return None

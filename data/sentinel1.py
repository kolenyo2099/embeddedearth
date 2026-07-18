"""
Sentinel-1 Data Retrieval Module

Handles Sentinel-1 (SAR) data acquisition via Google Earth Engine.
"""

import ee
from typing import Tuple, Optional, List
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import gee_config, sentinel1_bands
from data.gee_client import GEEClient


class Sentinel1Retriever:
    """
    Retrieves and processes Sentinel-1 imagery from Google Earth Engine.
    
    Features:
    - Filters by Instrument Mode (IW) and Transmitter Receiver Polarisation (VV, VH)
    - Orbit processing (Descending/Ascending)
    - Temporal compositing (Median) to reduce speckle
    """
    
    def __init__(self):
        """Initialize the retriever, ensuring GEE is ready."""
        GEEClient.initialize()
        
        # Collection references
        self._s1_collection = gee_config.s1_collection
        
        # Band configuration
        self._bands = sentinel1_bands.band_names
    
    def get_collection(
        self,
        aoi: ee.Geometry,
        start_date: str = None,
        end_date: str = None,
        orbit_pass: str = 'DESCENDING' # 'ASCENDING', 'DESCENDING' or None (Both)
    ) -> ee.ImageCollection:
        """
        Get filtered Sentinel-1 collection for an area.
        
        Args:
            aoi: Area of interest as ee.Geometry.
            start_date: Start date (YYYY-MM-DD). Default: 90 days ago.
            end_date: End date (YYYY-MM-DD). Default: today.
            orbit_pass: Orbit pass direction to filter.
            
        Returns:
            Filtered ImageCollection.
        """
        # Default date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start = datetime.now() - timedelta(days=gee_config.default_days_back)
            start_date = start.strftime('%Y-%m-%d')
        
        # Load and filter Sentinel-1 collection
        # COPERNICUS/S1_GRD contains Sigma0 (Backscatter coefficient)
        s1 = (
            ee.ImageCollection(self._s1_collection)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
        )
        
        if orbit_pass:
             s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
        
        return s1
    
    def get_composite(
        self,
        aoi: ee.Geometry,
        start_date: str = None,
        end_date: str = None,
        reducer: str = 'median'
    ) -> ee.Image:
        """
        Get a composite for an area.
        Using median reduction helps remove speckle noise.
        
        Args:
            aoi: Area of interest as ee.Geometry.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            reducer: Reduction method ('median', 'mean').
            
        Returns:
            Composite ee.Image with VV and VH bands.
        """
        collection = self.get_collection(aoi, start_date, end_date)

        # Guard against empty collections: median() of nothing yields a band-less
        # image and .select() then raises an opaque EE error downstream.
        if collection.size().getInfo() == 0:
            raise ValueError(
                f"No Sentinel-1 scenes matched the AOI/date filter "
                f"(start={start_date}, end={end_date}). "
                f"Widen the date range or try a different orbit pass."
            )

        # Apply reducer
        if reducer == 'median':
            composite = collection.median()
        elif reducer == 'mean':
            composite = collection.mean()
        else:
            raise ValueError(f"Unknown reducer for SAR: {reducer}")

        # Select required bands
        composite = composite.select(self._bands)
        
        # Clip to AOI
        composite = composite.clip(aoi)
        
        return composite
    
    def normalize_for_model(self, image: ee.Image) -> ee.Image:
        """
        Normalize Sentinel-1 GRD imagery to [0, 1] for DOFA-CLIP input.

        GEE's `COPERNICUS/S1_GRD` delivers sigma-naught already in decibels
        (negative floats, typically VV in ~[-25, 0] and VH in ~[-30, -5]).
        We clip to [-25, 0] dB and linearly rescale to [0, 1]. No log10
        conversion — the values are already log-scaled.
        """
        min_db = -25.0
        max_db = 0.0
        return image.subtract(min_db).divide(max_db - min_db).clamp(0, 1)

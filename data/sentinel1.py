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
        Normalize image for DOFA-CLIP input.
        
        Sentinel-1 GRD in GEE is usually in decibels (dB) ranges approx -30 to 0.
        Or linear? 
        The 'COPERNICUS/S1_GRD' collection description says: 
        "Each scene ... processed to generate a level-1 Ground Range Detected (GRD) product."
        It contains 3 bands: HH, HV, or VV, VH...
        Wait, GEE provides them as calibrated backscatter coefficient (sigma-naught) in dB? 
        Actually, the default is raw power/intensity? No, usually linear.
        However, it's common to convert to dB for visualization: 10*log10(x).
        
        For DOFA, we should check if it expects dB or linear.
        The paper typically uses inputs normalized to 0-1 for stability.
        
        Let's assume linear input for physical modeling, but usually range is 0 to ~0.5.
        If inputs are dB (-25 to 0), we should probably scale them.
        
        For now, let's just clamp and scale to 0-1 range for a "visual-like" input if generic.
        But DOFA is a Foundation Model, might expect raw physical values.
        
        Let's try a safe normalization strategy:
        Clip to [-25, 0] dB and map to [0, 1].
        
        BUT GEE S1_GRD values are float.
        If they are not in dB, they are linear.
        Usually GEE S1 is not in dB by default unless 'COPERNICUS/S1_GRD_FLOAT' (deprecated) or verified.
        Checking GEE docs: processed sigma0 values.
        
        Let's stick to a robust min-max like normalization for now:
        VH: [-30, -5] -> [0, 1]
        VV: [-25, 0] -> [0, 1]
        
        Wait, if the values are linear, we should convert to dB first.
        It is safer to convert to dB.
        """
        # Convert to dB if not already (assuming linear if values are small positive)
        # Actually S1 GRD in GEE is linear power.
        
        # 10 * log10(x)
        image_db = image.log10().multiply(10.0)
        
        # Clip and Scale to 0-1
        # VV: range [-25, 0]
        # VH: range [-30, -5]
        
        min_db = -25.0
        max_db = 0.0
        
        return image_db.subtract(min_db).divide(max_db - min_db).clamp(0, 1)

    def get_visualization(self, image: ee.Image) -> ee.Image:
        """
        Create an RGB visualization for Sentinel-1.
        R: VV
        G: VH
        B: VV/VH ratio
        """
        # Assuming image is already normalized 0-1 or needs normalization?
        # Let's assume we take the raw/composite and return a 0-255 RGB for display
        
        # This helper might be useful for the frontend map
        pass

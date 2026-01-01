"""
Sentinel-2 Data Retrieval Module

Handles Sentinel-2 L2A data acquisition with Cloud Score+ masking
and temporal compositing via Google Earth Engine.
"""

import ee
from typing import Tuple, Optional, List
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import gee_config, sentinel2_bands
from data.gee_client import GEEClient


class Sentinel2Retriever:
    """
    Retrieves and processes Sentinel-2 imagery from Google Earth Engine.
    
    Features:
    - Cloud Score+ masking for superior cloud removal
    - Temporal median compositing
    - Multi-band export for DOFA-CLIP
    """
    
    def __init__(self):
        """Initialize the retriever, ensuring GEE is ready."""
        GEEClient.initialize()
        
        # Collection references
        self._s2_collection = gee_config.s2_collection
        self._cloud_collection = gee_config.cloud_score_collection
        
        # Band configuration
        self._bands = sentinel2_bands.band_names
        self._scale_factor = sentinel2_bands.scale_factor
    
    def _apply_cloud_mask(
        self,
        image: ee.Image,
        threshold: float = None
    ) -> ee.Image:
        """
        Apply Cloud Score+ mask to a Sentinel-2 image.
        
        Args:
            image: ee.Image to mask.
            threshold: Cloud probability threshold (0-1).
            
        Returns:
            Masked ee.Image.
        """
        threshold = threshold or gee_config.cloud_threshold
        
        # Get the cloud score band (cs = clear sky probability)
        # Higher cs means clearer sky
        cs = image.select('cs')
        
        # Create mask where clear sky probability is above threshold
        mask = cs.gte(threshold)
        
        return image.updateMask(mask)
    
    def _join_cloud_scores(
        self,
        s2_collection: ee.ImageCollection,
        aoi: ee.Geometry,
        start_date: str,
        end_date: str
    ) -> ee.ImageCollection:
        """
        Join Sentinel-2 collection with Cloud Score+ data.
        
        Args:
            s2_collection: Sentinel-2 ImageCollection.
            aoi: Area of interest geometry.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            
        Returns:
            ImageCollection with cloud scores joined.
        """
        # Load Cloud Score+ collection
        cs_collection = (
            ee.ImageCollection(self._cloud_collection)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
        )
        
        # Define join condition
        join_filter = ee.Filter.equals(
            leftField='system:index',
            rightField='system:index'
        )
        
        # Perform inner join
        joined = ee.Join.inner().apply(
            primary=s2_collection,
            secondary=cs_collection,
            condition=join_filter
        )
        
        # Merge bands from both collections
        def merge_bands(feature):
            s2_image = ee.Image(feature.get('primary'))
            cs_image = ee.Image(feature.get('secondary'))
            return s2_image.addBands(cs_image)
        
        return ee.ImageCollection(joined.map(merge_bands))
    
    def get_collection(
        self,
        aoi: ee.Geometry,
        start_date: str = None,
        end_date: str = None,
        max_cloud_cover: int = None
    ) -> ee.ImageCollection:
        """
        Get filtered Sentinel-2 collection for an area.
        
        Args:
            aoi: Area of interest as ee.Geometry.
            start_date: Start date (YYYY-MM-DD). Default: 90 days ago.
            end_date: End date (YYYY-MM-DD). Default: today.
            max_cloud_cover: Maximum cloud cover percentage.
            
        Returns:
            Filtered and cloud-masked ImageCollection.
        """
        # Default date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start = datetime.now() - timedelta(days=gee_config.default_days_back)
            start_date = start.strftime('%Y-%m-%d')
        
        max_cloud = max_cloud_cover or gee_config.max_cloud_cover
        
        # Load and filter Sentinel-2 collection
        s2 = (
            ee.ImageCollection(self._s2_collection)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
        )
        
        # Join with Cloud Score+ and apply masking
        s2_with_cs = self._join_cloud_scores(s2, aoi, start_date, end_date)
        s2_masked = s2_with_cs.map(self._apply_cloud_mask)
        
        # Check size (blocking call, but safe for low latency apps or debug)
        # count = s2_masked.size().getInfo()
        # print(f"Found {count} images for {start_date} to {end_date}")
        
        return s2_masked
    
    def get_composite(
        self,
        aoi: ee.Geometry,
        start_date: str = None,
        end_date: str = None,
        reducer: str = 'median'
    ) -> ee.Image:
        """
        Get a cloud-free composite for an area.
        
        Args:
            aoi: Area of interest as ee.Geometry.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            reducer: Reduction method ('median', 'mean', 'min', 'max').
            
        Returns:
            Cloud-free composite ee.Image.
        """
        collection = self.get_collection(aoi, start_date, end_date)
        
        # Apply reducer
        reducers = {
            'median': collection.median,
            'mean': collection.mean,
            'min': collection.min,
            'max': collection.max,
        }
        
        if reducer not in reducers:
            raise ValueError(f"Unknown reducer: {reducer}")
        
        composite = reducers[reducer]()
        
        # Fallback: if composite has no bands (empty collection), try to select from raw if existing
        # But we can't easily check for empty bands without getInfo()
        # Instead, verify we set the default bands correctly.
        
        # Note: If collection is empty, median() returns an image with no bands.
        # We should handle this upstream or set default bands.
        
        # Select only the bands we need for DOFA-CLIP
        # Use regexp to avoid error if band missing? No, strict selection is better.
        # Add a check for collection size
        
        composite = composite.select(self._bands)
        
        # Clip to AOI
        composite = composite.clip(aoi)
        
        return composite
    
    def normalize_for_model(self, image: ee.Image) -> ee.Image:
        """
        Normalize image for DOFA-CLIP input.
        
        Converts from reflectance (0-10000) to (0-1) range.
        
        Args:
            image: ee.Image with reflectance values.
            
        Returns:
            Normalized ee.Image.
        """
        return image.divide(self._scale_factor)
    
    def get_download_url(
        self,
        image: ee.Image,
        aoi: ee.Geometry,
        scale: int = 10,
        format: str = 'GEO_TIFF'
    ) -> str:
        """
        Get download URL for an image.
        
        Note: Limited to ~32MB. For larger areas, use export_to_drive().
        
        Args:
            image: ee.Image to download.
            aoi: Geometry to clip to.
            scale: Resolution in meters.
            format: Export format.
            
        Returns:
            Download URL string.
        """
        return image.getDownloadUrl({
            'region': aoi,
            'scale': scale,
            'format': format,
            'bands': self._bands,
        })
    
    def export_to_drive(
        self,
        image: ee.Image,
        aoi: ee.Geometry,
        filename: str,
        folder: str = 'EmbeddedEarth',
        scale: int = 10
    ) -> ee.batch.Task:
        """
        Export image to Google Drive for large areas.
        
        Args:
            image: ee.Image to export.
            aoi: Geometry for export region.
            filename: Output filename (without extension).
            folder: Drive folder name.
            scale: Resolution in meters.
            
        Returns:
            Export task (already started).
        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename,
            folder=folder,
            fileNamePrefix=filename,
            region=aoi,
            scale=scale,
            maxPixels=1e10,
        )
        
        task.start()
        return task


# Convenience function for quick access
def get_cloud_free_composite(
    aoi: ee.Geometry,
    start_date: str = None,
    end_date: str = None
) -> ee.Image:
    """
    Quick function to get a cloud-free Sentinel-2 composite.
    
    Args:
        aoi: Area of interest.
        start_date: Optional start date.
        end_date: Optional end date.
        
    Returns:
        Normalized, cloud-free composite image.
    """
    retriever = Sentinel2Retriever()
    composite = retriever.get_composite(aoi, start_date, end_date)
    return retriever.normalize_for_model(composite)

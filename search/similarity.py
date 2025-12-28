"""
Similarity Search Module

Provides high-level search functionality combining
text/image encoding with vector similarity.
"""

import numpy as np
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import search_config
from search.faiss_index import FAISSIndex, SearchResult, get_index
from models.encoders import TextEncoder, ImageEncoder, create_encoders


@dataclass
class SemanticSearchResult:
    """Result from a semantic search including tile data."""
    
    tile_index: int
    similarity_score: float
    tile_bounds: Optional[Tuple[float, float, float, float]] = None
    tile_data: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


class SemanticSearchEngine:
    """
    Semantic search engine for satellite imagery.
    
    Supports both text-to-image and image-to-image search.
    """
    
    def __init__(
        self,
        index: FAISSIndex = None,
        text_encoder: TextEncoder = None,
        image_encoder: ImageEncoder = None
    ):
        """
        Initialize the search engine.
        
        Args:
            index: Vector index (creates new if None).
            text_encoder: Text encoder instance.
            image_encoder: Image encoder instance.
        """
        self._index = index
        self._text_encoder = text_encoder
        self._image_encoder = image_encoder
        
        # Tile data storage
        self._tile_data: dict = {}
    
    @property
    def index(self) -> FAISSIndex:
        """Get or create the vector index."""
        if self._index is None:
            self._index = get_index()
        return self._index
    
    @property
    def text_encoder(self) -> TextEncoder:
        """Get or create text encoder."""
        if self._text_encoder is None:
            self._text_encoder, self._image_encoder = create_encoders()
        return self._text_encoder
    
    @property
    def image_encoder(self) -> ImageEncoder:
        """Get or create image encoder."""
        if self._image_encoder is None:
            self._text_encoder, self._image_encoder = create_encoders()
        return self._image_encoder
    
    def index_tiles(
        self,
        tiles: List[np.ndarray],
        bounds: List[tuple] = None,
        metadata: List[dict] = None
    ) -> List[int]:
        """
        Index a list of tiles for search.
        
        Args:
            tiles: List of tile arrays (C, H, W).
            bounds: List of geospatial bounds per tile.
            metadata: Additional metadata per tile.
            
        Returns:
            List of assigned indices.
        """
        # Encode all tiles
        embeddings = self.image_encoder.encode_batch(tiles)
        
        # Prepare metadata
        metas = []
        for i, tile in enumerate(tiles):
            meta = metadata[i] if metadata else {}
            meta['bounds'] = bounds[i] if bounds else None
            metas.append(meta)
        
        # Add to index
        indices = self.index.add(embeddings, metas)
        
        # Store tile data
        for idx, tile in zip(indices, tiles):
            self._tile_data[idx] = tile
        
        return indices
    
    def search_by_text(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> List[SemanticSearchResult]:
        """
        Search indexed tiles by text query.
        
        Args:
            query: Natural language query.
            top_k: Number of results.
            threshold: Minimum similarity.
            
        Returns:
            List of search results.
        """
        top_k = top_k or search_config.top_k
        threshold = threshold or search_config.similarity_threshold
        
        # Encode query
        query_embedding = self.text_encoder.encode(query)
        
        # Search
        results = self.index.search(query_embedding[0], k=top_k, threshold=threshold)
        
        # Convert to semantic results
        return self._to_semantic_results(results)
    
    def search_by_image(
        self,
        reference_image: np.ndarray,
        top_k: int = None,
        threshold: float = None
    ) -> List[SemanticSearchResult]:
        """
        Search indexed tiles by reference image.
        
        Args:
            reference_image: Reference image (C, H, W).
            top_k: Number of results.
            threshold: Minimum similarity.
            
        Returns:
            List of search results.
        """
        top_k = top_k or search_config.top_k
        threshold = threshold or search_config.similarity_threshold
        
        # Encode reference
        ref_embedding = self.image_encoder.encode(reference_image)
        
        # Search
        results = self.index.search(ref_embedding[0], k=top_k, threshold=threshold)
        
        # Convert to semantic results
        return self._to_semantic_results(results)
    
    def _to_semantic_results(
        self,
        results: List[SearchResult]
    ) -> List[SemanticSearchResult]:
        """Convert index results to semantic results."""
        semantic_results = []
        
        for result in results:
            meta = result.metadata or {}
            
            semantic_results.append(SemanticSearchResult(
                tile_index=result.index,
                similarity_score=result.score,
                tile_bounds=meta.get('bounds'),
                tile_data=self._tile_data.get(result.index),
                metadata=meta
            ))
        
        return semantic_results
    
    def clear(self):
        """Clear all indexed data."""
        self.index.clear()
        self._tile_data.clear()


def quick_text_search(
    tiles: List[np.ndarray],
    query: str,
    top_k: int = 10
) -> List[SemanticSearchResult]:
    """
    Quick one-shot text search on tiles.
    
    Args:
        tiles: List of tile arrays.
        query: Text query.
        top_k: Number of results.
        
    Returns:
        Search results.
    """
    engine = SemanticSearchEngine()
    engine.index_tiles(tiles)
    return engine.search_by_text(query, top_k=top_k)


def quick_image_search(
    tiles: List[np.ndarray],
    reference: np.ndarray,
    top_k: int = 10
) -> List[SemanticSearchResult]:
    """
    Quick one-shot image search on tiles.
    
    Args:
        tiles: List of tile arrays.
        reference: Reference image.
        top_k: Number of results.
        
    Returns:
        Search results.
    """
    engine = SemanticSearchEngine()
    engine.index_tiles(tiles)
    return engine.search_by_image(reference, top_k=top_k)

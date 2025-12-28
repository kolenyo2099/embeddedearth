"""
Batched Inference Module

Handles efficient batch processing of tiles through the model
with memory management and progress tracking.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import model_config
from pipeline.tiling import Tile
from models.encoders import ImageEncoder


@dataclass
class EmbeddedTile:
    """A tile with its computed embedding."""
    
    tile: Tile
    embedding: np.ndarray
    score: Optional[float] = None


class BatchInference:
    """
    Performs batched inference on tiles for efficient processing.
    
    Handles:
    - GPU/CPU memory management
    - Progress tracking for UI feedback
    - OOM recovery
    """
    
    def __init__(
        self,
        encoder: ImageEncoder = None,
        batch_size: int = None,
        device: str = None
    ):
        """
        Initialize batch inference.
        
        Args:
            encoder: Image encoder instance.
            batch_size: Number of tiles per batch.
            device: Target device.
        """
        self._encoder = encoder
        self.batch_size = batch_size or model_config.batch_size
        self.device = device or model_config.device
    
    @property
    def encoder(self) -> ImageEncoder:
        """Get or create the encoder."""
        if self._encoder is None:
            self._encoder = ImageEncoder()
        return self._encoder
    
    def encode_tiles(
        self,
        tiles: List[Tile],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: bool = True
    ) -> List[EmbeddedTile]:
        """
        Encode all tiles to embeddings.
        
        Args:
            tiles: List of tiles to process.
            progress_callback: Optional callback(current, total).
            show_progress: Show tqdm progress bar.
            
        Returns:
            List of EmbeddedTile objects.
        """
        results = []
        total = len(tiles)
        
        # Create iterator with optional progress bar
        if show_progress:
            iterator = tqdm(
                range(0, total, self.batch_size),
                desc="Encoding tiles",
                unit="batch"
            )
        else:
            iterator = range(0, total, self.batch_size)
        
        for i in iterator:
            batch_tiles = tiles[i:i + self.batch_size]
            
            # Stack tile data
            batch_data = np.stack([t.data for t in batch_tiles], axis=0)
            
            # Encode batch
            try:
                embeddings = self.encoder.encode(batch_data)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size and retry
                    torch.cuda.empty_cache()
                    embeddings = self._encode_with_smaller_batch(batch_tiles)
                else:
                    raise
            
            # Create embedded tiles
            for tile, emb in zip(batch_tiles, embeddings):
                results.append(EmbeddedTile(tile=tile, embedding=emb))
            
            # Report progress
            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)
        
        return results
    
    def _encode_with_smaller_batch(
        self,
        tiles: List[Tile]
    ) -> np.ndarray:
        """
        Fallback encoding with smaller batches for OOM recovery.
        
        Args:
            tiles: Tiles to encode.
            
        Returns:
            Stacked embeddings.
        """
        embeddings = []
        small_batch_size = max(1, self.batch_size // 4)
        
        for i in range(0, len(tiles), small_batch_size):
            batch = tiles[i:i + small_batch_size]
            batch_data = np.stack([t.data for t in batch], axis=0)
            
            emb = self.encoder.encode(batch_data)
            embeddings.append(emb)
            
            torch.cuda.empty_cache()
        
        return np.concatenate(embeddings, axis=0)
    
    def rank_tiles(
        self,
        embedded_tiles: List[EmbeddedTile],
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[EmbeddedTile]:
        """
        Rank tiles by similarity to query.
        
        Args:
            embedded_tiles: List of tiles with embeddings.
            query_embedding: Query embedding (1, D) or (D,).
            top_k: Number of top results to return.
            
        Returns:
            Top-k tiles sorted by score (descending).
        """
        # Ensure query is 1D
        query = query_embedding.flatten()
        
        # Compute similarities
        for tile in embedded_tiles:
            score = np.dot(tile.embedding, query)
            tile.score = float(score)
        
        # Sort by score descending
        sorted_tiles = sorted(
            embedded_tiles,
            key=lambda t: t.score,
            reverse=True
        )
        
        return sorted_tiles[:top_k]


def process_image_with_query(
    image: np.ndarray,
    query_embedding: np.ndarray,
    tile_size: int = None,
    top_k: int = 10,
    bounds: tuple = None
) -> List[EmbeddedTile]:
    """
    Full pipeline: tile image, encode, and rank by query.
    
    Args:
        image: Source image (C, H, W).
        query_embedding: Text/image query embedding.
        tile_size: Tile size in pixels.
        top_k: Number of results.
        bounds: Geospatial bounds.
        
    Returns:
        Top-k matched tiles with scores.
    """
    from pipeline.tiling import tile_image
    
    # Generate tiles
    tiles = tile_image(image, tile_size=tile_size, bounds=bounds)
    
    # Encode tiles
    inference = BatchInference()
    embedded = inference.encode_tiles(tiles)
    
    # Rank by query
    results = inference.rank_tiles(embedded, query_embedding, top_k)
    
    return results

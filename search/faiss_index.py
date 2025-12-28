"""
FAISS Vector Index Module

Provides in-memory vector storage and similarity search
using Facebook AI Similarity Search (FAISS).
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config import search_config


@dataclass
class SearchResult:
    """Result from a similarity search."""
    
    index: int
    score: float
    metadata: Optional[dict] = None


class FAISSIndex:
    """
    FAISS-based vector index for semantic search.
    
    Uses IndexFlatIP (Inner Product) for normalized vectors,
    which is equivalent to cosine similarity.
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = None
    ):
        """
        Initialize the FAISS index.
        
        Args:
        Args:
            dimension: Vector dimension (768 for DOFA-CLIP/ViT-B).
            index_type: FAISS index type.
        """
        self.dimension = dimension
        self.index_type = index_type or search_config.index_type
        
        # Create index
        self._index = self._create_index()
        
        # Metadata storage (maps index -> metadata)
        self._metadata: dict = {}
        
        # Track number of vectors
        self._count = 0
    
    def _create_index(self) -> faiss.Index:
        """Create the FAISS index based on type."""
        if self.index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # For larger datasets, use IVF
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            return faiss.IndexFlatIP(self.dimension)
    
    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        return self._count
    
    def add(
        self,
        vectors: np.ndarray,
        metadata: List[dict] = None
    ) -> List[int]:
        """
        Add vectors to the index.
        
        Args:
            vectors: Array of shape (N, D).
            metadata: Optional list of metadata dicts.
            
        Returns:
            List of assigned indices.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        vectors = vectors.astype(np.float32)
        
        # FAISS requires contiguous arrays
        if not vectors.flags['C_CONTIGUOUS']:
            vectors = np.ascontiguousarray(vectors)
        
        # Get starting index
        start_idx = self._count
        
        # Add to index
        self._index.add(vectors)
        
        # Update count
        n_added = vectors.shape[0]
        self._count += n_added
        
        # Store metadata
        indices = list(range(start_idx, start_idx + n_added))
        if metadata:
            for idx, meta in zip(indices, metadata):
                self._metadata[idx] = meta
        
        return indices
    
    def search(
        self,
        query: np.ndarray,
        k: int = None,
        threshold: float = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector (D,) or (1, D).
            k: Number of results to return.
            threshold: Minimum similarity threshold.
            
        Returns:
            List of SearchResult objects.
        """
        k = k or search_config.top_k
        threshold = threshold or search_config.similarity_threshold
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        if not query.flags['C_CONTIGUOUS']:
            query = np.ascontiguousarray(query)
        
        # Limit k to available vectors
        k = min(k, self._count)
        if k == 0:
            return []
        
        # Perform search
        scores, indices = self._index.search(query, k)
        
        # Convert to results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue
            if score < threshold:
                continue
            
            results.append(SearchResult(
                index=int(idx),
                score=float(score),
                metadata=self._metadata.get(int(idx))
            ))
        
        return results
    
    def get_metadata(self, index: int) -> Optional[dict]:
        """Get metadata for a specific index."""
        return self._metadata.get(index)
    
    def clear(self):
        """Clear all vectors and metadata."""
        self._index = self._create_index()
        self._metadata.clear()
        self._count = 0
    
    def save(self, path: str):
        """Save index to file."""
        faiss.write_index(self._index, path)
    
    def load(self, path: str):
        """Load index from file."""
        self._index = faiss.read_index(path)
        self._count = self._index.ntotal


# Global index instance
_global_index: Optional[FAISSIndex] = None


def get_index(dimension: int = 768) -> FAISSIndex:
    """
    Get or create the global FAISS index.
    
    Args:
        dimension: Vector dimension.
        
    Returns:
        FAISSIndex instance.
    """
    global _global_index
    if _global_index is None:
        _global_index = FAISSIndex(dimension)
    return _global_index


def reset_index():
    """Reset the global index."""
    global _global_index
    if _global_index:
        _global_index.clear()
    _global_index = None

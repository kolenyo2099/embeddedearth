"""
Postprocessing Module

Handles deduplication and Non-Maximum Suppression (NMS)
for overlapping tile results.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from pipeline.inference import EmbeddedTile


def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: (x1, y1, x2, y2) for first box.
        box2: (x1, y1, x2, y2) for second box.
        
    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def tile_to_box(tile: EmbeddedTile) -> Tuple[int, int, int, int]:
    """
    Convert tile to bounding box coordinates.
    
    Args:
        tile: EmbeddedTile instance.
        
    Returns:
        (x1, y1, x2, y2) tuple.
    """
    t = tile.tile
    return (t.x, t.y, t.x + t.width, t.y + t.height)


def non_maximum_suppression(
    tiles: List[EmbeddedTile],
    iou_threshold: float = 0.5
) -> List[EmbeddedTile]:
    """
    Apply Non-Maximum Suppression to remove redundant tiles.
    
    Keeps the highest-scoring tile among overlapping detections.
    
    Args:
        tiles: List of tiles sorted by score (descending).
        iou_threshold: IoU threshold for suppression.
        
    Returns:
        Filtered list of tiles.
    """
    if len(tiles) <= 1:
        return tiles
    
    # Sort by score descending (should already be sorted)
    sorted_tiles = sorted(tiles, key=lambda t: t.score or 0, reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, tile_i in enumerate(sorted_tiles):
        if i in suppressed:
            continue
        
        keep.append(tile_i)
        box_i = tile_to_box(tile_i)
        
        for j in range(i + 1, len(sorted_tiles)):
            if j in suppressed:
                continue
            
            box_j = tile_to_box(sorted_tiles[j])
            iou = compute_iou(box_i, box_j)
            
            if iou > iou_threshold:
                suppressed.add(j)
    
    return keep


def spatial_deduplication(
    tiles: List[EmbeddedTile],
    distance_threshold: int = None
) -> List[EmbeddedTile]:
    """
    Remove tiles whose centers are too close together.
    
    Simpler alternative to NMS based on center distance.
    
    Args:
        tiles: List of tiles sorted by score.
        distance_threshold: Minimum distance between centers.
        
    Returns:
        Deduplicated list.
    """
    from config import tiling_config
    
    if distance_threshold is None:
        # Default: stride (half tile size with 50% overlap)
        distance_threshold = tiling_config.stride
    
    if len(tiles) <= 1:
        return tiles
    
    sorted_tiles = sorted(tiles, key=lambda t: t.score or 0, reverse=True)
    
    keep = []
    kept_centers = []
    
    for tile in sorted_tiles:
        center = tile.tile.center
        
        # Check distance to all kept centers
        is_unique = True
        for kept_center in kept_centers:
            dist = np.sqrt(
                (center[0] - kept_center[0])**2 +
                (center[1] - kept_center[1])**2
            )
            if dist < distance_threshold:
                is_unique = False
                break
        
        if is_unique:
            keep.append(tile)
            kept_centers.append(center)
    
    return keep


def filter_by_threshold(
    tiles: List[EmbeddedTile],
    threshold: float = 0.3
) -> List[EmbeddedTile]:
    """
    Filter tiles by minimum similarity score.
    
    Args:
        tiles: List of tiles with scores.
        threshold: Minimum score to keep.
        
    Returns:
        Filtered list.
    """
    return [t for t in tiles if t.score is not None and t.score >= threshold]


def postprocess_results(
    tiles: List[EmbeddedTile],
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    use_nms: bool = True
) -> List[EmbeddedTile]:
    """
    Full postprocessing pipeline.
    
    Args:
        tiles: Raw tile results.
        score_threshold: Minimum similarity score.
        iou_threshold: IoU threshold for NMS.
        use_nms: Whether to apply NMS (vs simpler dedup).
        
    Returns:
        Processed and filtered tiles.
    """
    # Filter by score
    filtered = filter_by_threshold(tiles, score_threshold)
    
    # Apply deduplication
    if use_nms:
        deduplicated = non_maximum_suppression(filtered, iou_threshold)
    else:
        deduplicated = spatial_deduplication(filtered)
    
    return deduplicated

"""
Postprocessing Module

Non-Maximum Suppression (NMS) for overlapping tile results.

The search grids use 50% tile overlap, so a strong match typically appears
in up to four adjacent tiles. NMS keeps the highest-scoring tile among
overlapping detections so the result list isn't dominated by near-duplicates
of the same location.
"""

from typing import List, Tuple


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection over Union between two boxes.

    Args:
        box1: (minx, miny, maxx, maxy) for first box.
        box2: (minx, miny, maxx, maxy) for second box.

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def nms_results(
    results: List[dict],
    iou_threshold: float = 0.3
) -> List[dict]:
    """
    Apply NMS to a list of result dicts with geographic bounds.

    Args:
        results: Result dicts each containing 'score' (float) and
                 'bounds' (minx, miny, maxx, maxy). Results without bounds
                 are always kept.
        iou_threshold: Overlap above which the lower-scoring result is dropped.
                       Grid geometry with 50% tile overlap: side neighbors have
                       IoU = 1/3, diagonal neighbors 1/7. The default 0.3
                       suppresses side-neighbor duplicates while keeping
                       diagonal neighbors; use ~0.1 to collapse a hotspot's
                       whole neighborhood to a single result.

    Returns:
        Filtered list, sorted by score descending.
    """
    ordered = sorted(results, key=lambda r: r.get('score', 0.0), reverse=True)

    keep: List[dict] = []
    for candidate in ordered:
        c_bounds = candidate.get('bounds')
        if c_bounds is None:
            keep.append(candidate)
            continue

        overlaps = any(
            k.get('bounds') is not None
            and compute_iou(c_bounds, k['bounds']) > iou_threshold
            for k in keep
        )
        if not overlaps:
            keep.append(candidate)

    return keep

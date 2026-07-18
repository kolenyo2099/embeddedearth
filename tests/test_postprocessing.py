"""
Tests for pipeline/postprocessing.py

Verifies IoU math and geographic NMS on result dicts.
No model loading, no GEE.
"""

import pytest
from pipeline.postprocessing import compute_iou, nms_results


class TestComputeIoU:
    def test_identical_boxes(self):
        box = (0.0, 0.0, 1.0, 1.0)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert compute_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0

    def test_half_overlap(self):
        # Two unit boxes sharing half their area: I=0.5, U=1.5 -> IoU=1/3
        iou = compute_iou((0, 0, 1, 1), (0.5, 0, 1.5, 1))
        assert iou == pytest.approx(1 / 3)

    def test_zero_area(self):
        assert compute_iou((0, 0, 0, 0), (0, 0, 1, 1)) == 0.0


class TestNMSResults:
    def _result(self, score, bounds):
        return {'score': score, 'bounds': bounds}

    def test_empty(self):
        assert nms_results([]) == []

    def test_keeps_highest_of_overlapping_pair(self):
        # 50%-overlap neighbors (IoU = 1/3 > default threshold 0.3)
        a = self._result(0.9, (0.0, 0.0, 1.0, 1.0))
        b = self._result(0.5, (0.5, 0.0, 1.5, 1.0))
        kept = nms_results([b, a])
        assert kept == [a]

    def test_keeps_distant_results(self):
        a = self._result(0.9, (0.0, 0.0, 1.0, 1.0))
        b = self._result(0.5, (5.0, 5.0, 6.0, 6.0))
        kept = nms_results([b, a])
        assert kept == [a, b]  # sorted by score desc

    def test_diagonal_neighbors_survive_default(self):
        # Diagonal 50%-overlap neighbors: IoU = 1/7 < 0.3
        a = self._result(0.9, (0.0, 0.0, 1.0, 1.0))
        b = self._result(0.5, (0.5, 0.5, 1.5, 1.5))
        kept = nms_results([a, b])
        assert len(kept) == 2

    def test_aggressive_threshold_collapses_neighborhood(self):
        a = self._result(0.9, (0.0, 0.0, 1.0, 1.0))
        b = self._result(0.5, (0.5, 0.5, 1.5, 1.5))
        kept = nms_results([a, b], iou_threshold=0.1)
        assert kept == [a]

    def test_missing_bounds_always_kept(self):
        a = self._result(0.9, (0.0, 0.0, 1.0, 1.0))
        b = {'score': 0.5, 'bounds': None}
        kept = nms_results([a, b])
        assert len(kept) == 2

    def test_sorts_by_score_descending(self):
        results = [
            self._result(0.2, (10, 10, 11, 11)),
            self._result(0.8, (20, 20, 21, 21)),
            self._result(0.5, (30, 30, 31, 31)),
        ]
        kept = nms_results(results)
        assert [r['score'] for r in kept] == [0.8, 0.5, 0.2]

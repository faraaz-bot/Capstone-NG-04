import unittest

import cv2
import numpy as np

from pen_tracking.tracking import PageCalibration, PointFilter, StrokeRecorder, find_marker


class TrackingTests(unittest.TestCase):
    def test_perspective_maps_page_corners(self):
        page = PageCalibration([[40, 20], [440, 40], [400, 620], [60, 600]], (640, 720))
        for corner, expected in zip(page.corners, [(0, 0), (210, 0), (210, 297), (0, 297)]):
            result = cv2.perspectiveTransform(np.float32([[corner]]), page.matrix)[0, 0]
            np.testing.assert_allclose(result, expected, atol=0.001)
            np.testing.assert_allclose(page.to_mm(corner), expected, atol=0.001)
        self.assertIsNone(page.to_mm((0, 0)))

    def test_rejects_crossed_corners(self):
        with self.assertRaises(ValueError):
            PageCalibration([[0, 0], [100, 100], [100, 0], [0, 100]], (200, 200))

    def test_marker_is_found_and_roi_excludes_distractor(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.circle(frame, (50, 70), 10, (0, 255, 0), -1)
        cv2.circle(frame, (230, 70), 25, (0, 255, 0), -1)
        detection, _ = find_marker(frame, [35, 80, 60], [85, 255, 255], roi=(0, 0, 100, 150))
        np.testing.assert_allclose(detection[0], [50, 70], atol=0.1)
        missing, _ = find_marker(frame, [35, 80, 60], [85, 255, 255], roi=(0, 120, 100, 200))
        self.assertIsNone(missing)

    def test_missing_and_uncertain_points_split_strokes(self):
        recorder = StrokeRecorder()
        self.assertIsNone(recorder.observe((1, 1), 'unknown'))
        self.assertEqual(recorder.observe((1, 1), 'down'), 1)
        self.assertEqual(recorder.observe((2, 2), 'down'), 1)
        self.assertIsNone(recorder.observe(None, 'down'))
        self.assertEqual(recorder.observe((3, 3), 'down'), 2)
        self.assertIsNone(recorder.observe((4, 4), 'up'))
        self.assertEqual(recorder.observe((5, 5), 'down'), 3)
        self.assertIsNone(recorder.observe((6, 6), 'unknown'))
        self.assertEqual(recorder.observe((7, 7), 'down'), 4)

    def test_filter_reduces_stationary_jitter(self):
        smoother = PointFilter()
        raw = [np.array([30 + (-1) ** i, 40.0]) for i in range(100)]
        filtered = [smoother.update(point, i / 30) for i, point in enumerate(raw)]
        self.assertLess(np.std(np.array(filtered)[20:, 0]), np.std(np.array(raw)[20:, 0]))


if __name__ == '__main__':
    unittest.main()

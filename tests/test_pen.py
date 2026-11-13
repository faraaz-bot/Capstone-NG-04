import unittest

import cv2
import numpy as np

from pen_tracking.pen import PenTipTracker


class PenTrackerTests(unittest.TestCase):
    @staticmethod
    def scene():
        image = np.full((220, 300, 3), 240, np.uint8)
        # Nearby ink and a large blue object must not become the selected pen.
        cv2.putText(image, 'ABC', (15, 180), 0, 1, (20, 20, 20), 2)
        cv2.rectangle(image, (210, 15), (290, 80), (220, 80, 30), -1)
        cv2.line(image, (70, 100), (160, 120), (35, 35, 35), 12)
        cv2.line(image, (70, 100), (57, 97), (90, 90, 90), 3)
        cv2.line(image, (97, 100), (93, 112), (200, 200, 200), 4)
        return image

    def test_tracks_selected_pen_translation_among_distractors(self):
        first = self.scene()
        tracker = PenTipTracker(radius=16)
        tracker.initialize(first, (57, 97), (95, 106))
        transform = np.float32([[1, 0, 8], [0, 1, 5]])
        second = cv2.warpAffine(first, transform, first.shape[1::-1], borderValue=(240, 240, 240))
        point = tracker.update(second, 1 / 30)
        self.assertIsNotNone(point)
        np.testing.assert_allclose(point, (65, 102), atol=2)

    def test_rotation_moves_tip_offset_with_body(self):
        first = self.scene()
        tracker = PenTipTracker(radius=16)
        tracker.initialize(first, (57, 97), (95, 106))
        transform = cv2.getRotationMatrix2D((95, 106), 9, 1)
        second = cv2.warpAffine(first, transform, first.shape[1::-1], borderValue=(240, 240, 240))
        expected = transform @ np.array([57, 97, 1])
        point = tracker.update(second, 1 / 30)
        self.assertIsNotNone(point)
        np.testing.assert_allclose(point, expected, atol=3)

    def test_occlusion_emits_missing_and_long_gap_requires_reselection(self):
        frame = self.scene()
        tracker = PenTipTracker(radius=16)
        tracker.initialize(frame, (57, 97), (95, 106))
        blank = np.full_like(frame, 240)
        self.assertIsNone(tracker.update(blank, 0.1))
        self.assertEqual(tracker.state, 'uncertain')
        self.assertIsNone(tracker.update(frame, 0.6))
        self.assertEqual(tracker.state, 'lost')
        self.assertIsNone(tracker.update(frame, 0.7))
        tracker.initialize(frame, (57, 97), (95, 106), timestamp=0.8)
        self.assertIsNotNone(tracker.update(frame, 0.9))

    def test_short_gap_can_recover_without_emitting_predicted_ink(self):
        frame = self.scene()
        tracker = PenTipTracker(radius=16)
        tracker.initialize(frame, (57, 97), (95, 106))
        self.assertIsNone(tracker.update(np.full_like(frame, 240), 0.1))
        self.assertIsNotNone(tracker.update(frame, 0.2))

    def test_rejects_flat_patch_and_resolution_change(self):
        tracker = PenTipTracker(radius=16)
        with self.assertRaises(ValueError):
            tracker.initialize(np.full((220, 300, 3), 240, np.uint8), (57, 97), (95, 106))
        tracker.initialize(self.scene(), (57, 97), (95, 106))
        with self.assertRaises(ValueError):
            tracker.initialize(np.full((220, 300, 3), 240, np.uint8), (57, 97), (95, 106))
        self.assertIsNotNone(tracker.update(self.scene(), 0.05))
        self.assertIsNone(tracker.update(np.zeros((100, 100, 3), np.uint8), 0.1))


if __name__ == '__main__':
    unittest.main()

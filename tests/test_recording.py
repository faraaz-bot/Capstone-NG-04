import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from pen_tracking.app import main
from pen_tracking.tracking import PageCalibration


class RecordingTests(unittest.TestCase):
    def test_ordinary_pen_session_needs_no_green_marker_and_breaks_on_occlusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'pen.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 240))
            self.assertTrue(writer.isOpened())
            for index in range(6):
                frame = np.full((240, 320, 3), 240, np.uint8)
                if index != 3:
                    dx = index * 2
                    cv2.line(frame, (100 + dx, 120), (200 + dx, 160), (20, 20, 20), 12)
                    cv2.line(frame, (140 + dx, 130), (135 + dx, 143), (200, 200, 200), 4)
                writer.write(frame)
            writer.release()
            calibration = root / 'page.json'
            page = PageCalibration([[0, 0], [319, 0], [319, 239], [0, 239]], (320, 240))
            calibration.write_text(json.dumps(page.as_dict()))
            argv = ['pen-track', '--video', str(video), '--calibration', str(calibration),
                    '--output', str(root / 'sessions')]

            def select(frame, tracker):
                return tracker.initialize(frame, (100, 120), (140, 136))

            with patch.object(sys, 'argv', argv), patch('cv2.imshow'), \
                    patch('pen_tracking.app.select_tip', side_effect=select), \
                    patch('cv2.destroyAllWindows'), \
                    patch('cv2.waitKey', side_effect=[ord('d'), -1, -1, -1, ord('d'), ord('q')]):
                main()
            session = next((root / 'sessions').iterdir())
            rows = [json.loads(line) for line in (session / 'points.jsonl').read_text().splitlines()]
            self.assertEqual([r['stroke_id'] for r in rows], [None, 1, 1, None, None, 2])
            self.assertEqual(rows[3]['tracking_state'], 'uncertain')
            self.assertIsNone(rows[3]['tip_px'])
            self.assertIsNone(rows[3]['raw_mm'])
            self.assertEqual(rows[4]['contact'], 'unknown')
            self.assertTrue(all(r['marker_px'] is None for r in rows))
            self.assertEqual(json.loads((session / 'metadata.json').read_text())['tracker'], 'pen')

    def test_replay_ignores_arbitrary_hand_and_breaks_ink_after_marker_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'trial.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 240))
            self.assertTrue(writer.isOpened())
            for index in range(6):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                if index != 3:
                    cv2.circle(frame, (60 + index * 20, 120), 10, (0, 255, 0), -1)
                writer.write(frame)
            writer.release()
            calibration = root / 'page.json'
            page = PageCalibration([[0, 0], [319, 0], [319, 239], [0, 239]], (320, 240))
            calibration.write_text(json.dumps(page.as_dict()))
            argv = ['pen-track', '--tracker', 'marker', '--video', str(video), '--calibration', str(calibration),
                    '--hand-model', 'unused.task', '--output', str(root / 'sessions')]
            # Simulate a detector that sees only the non-writing hand, far away
            # from the marker. Its hint must not veto detections by default.
            wrong_hand = SimpleNamespace(find=lambda frame, timestamp: (0, 0, 10, 10), close=lambda: None)
            hands_module = SimpleNamespace(HandRegion=lambda *args: wrong_hand)
            # Frame 0 unknown, 1/2 down, 3 missing, 4 unknown, 5 manually down again.
            with patch.object(sys, 'argv', argv), patch('cv2.imshow'), \
                    patch.dict(sys.modules, {'pen_tracking.hands': hands_module}), \
                    patch('cv2.destroyAllWindows'), \
                    patch('cv2.waitKey', side_effect=[ord('d'), -1, -1, -1, ord('d'), ord('q')]):
                main()
            session = next((root / 'sessions').iterdir())
            rows = [json.loads(line) for line in (session / 'points.jsonl').read_text().splitlines()]
            self.assertEqual([row['stroke_id'] for row in rows], [None, 1, 1, None, None, 2])
            self.assertEqual(rows[3]['contact'], 'unknown')
            self.assertIsNone(rows[3]['filtered_mm'])
            self.assertEqual(rows[4]['contact'], 'unknown')
            self.assertEqual(rows[-1]['timestamp_s'], 5 / 30)
            ink = cv2.imread(str(session / 'ink.png'))
            self.assertIsNotNone(ink)
            self.assertLess(int(ink.min()), 255)
            metadata = json.loads((session / 'metadata.json').read_text())
            self.assertEqual(metadata['contact_method'], 'manual keyboard')


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from pen_tracking.evaluate_spots import evaluate


class EvaluationTests(unittest.TestCase):
    def test_missing_estimates_are_preserved_and_pixel_scale_is_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            (report / 'summary.json').write_text(json.dumps({'original_size': [2560, 1440]}))
            rows = [{'frame': 1, 'centroid_original_px': [26, 48]},
                    {'frame': 2, 'centroid_original_px': None}]
            (report / 'detections.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            labels = {'coordinate_width': 1280, 'clips': [{'id': 'sample', 'labels': [
                {'frame': 1, 'tip_px': [10, 20]}, {'frame': 2, 'tip_px': [20, 30]}]}]}
            result = evaluate(report, labels, 'sample')
            self.assertEqual(result['spot_labels_total'], 2)
            self.assertEqual(result['spot_labels_with_estimate'], 1)
            self.assertEqual(result['spot_checks'][0]['error_px'], 5)
            self.assertIsNone(result['spot_checks'][1]['error_px'])


if __name__ == '__main__':
    unittest.main()

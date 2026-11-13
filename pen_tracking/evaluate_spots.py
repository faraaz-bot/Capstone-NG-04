"""Compare a replay's estimates to a small set of manually marked frames."""
import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(report, annotations, clip_id):
    summary = json.loads((report / 'summary.json').read_text())
    rows = {row['frame']: row for row in map(json.loads, (report / 'detections.jsonl').read_text().splitlines())}
    clip = next(clip for clip in annotations['clips'] if clip['id'] == clip_id)
    scale = annotations['coordinate_width'] / summary['original_size'][0]
    spots = []
    for label in clip['labels']:
        prediction = rows.get(label['frame'], {}).get('centroid_original_px')
        estimate = np.asarray(prediction) * scale if prediction is not None else None
        error = float(np.linalg.norm(estimate - label['tip_px'])) if estimate is not None else None
        spots.append(dict(**label, estimate_px=estimate.tolist() if estimate is not None else None,
                          error_px=error))
    errors = [row['error_px'] for row in spots if row['error_px'] is not None]
    summary['spot_checks'] = spots
    summary['spot_median_error_at_width1280_px'] = float(np.median(errors)) if errors else None
    summary['spot_max_error_at_width1280_px'] = max(errors) if errors else None
    summary['spot_labels_with_estimate'] = len(errors)
    summary['spot_labels_total'] = len(spots)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--labels', type=Path, default=Path('docs/pen-tip-spot-labels.json'))
    parser.add_argument('--clip-id', required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.report, json.loads(args.labels.read_text()), args.clip_id), indent=2))


if __name__ == '__main__':
    main()

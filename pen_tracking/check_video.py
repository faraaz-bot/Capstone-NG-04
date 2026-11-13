"""Inspect colour detections without a camera, calibration, or GUI."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .tracking import find_marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config/marker.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hand-model', type=Path)
    parser.add_argument('--stride', type=int, default=5, help='Process every Nth frame')
    parser.add_argument('--max-width', type=int, default=1280)
    args = parser.parse_args()
    if args.stride < 1 or args.max_width < 1:
        parser.error('stride and max-width must be positive')
    config = json.loads(args.config.read_text())
    cap = cv2.VideoCapture(str(args.video))
    hands = None
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open {args.video}')
        if args.hand_model:
            from .hands import HandRegion
            hands = HandRegion(args.hand_model)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError('Video has no usable FPS')
        args.output.mkdir(parents=True, exist_ok=True)
        samples, previews = [], []
        index, next_preview = 0, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.stride:
                index += 1
                continue
            original_size = (frame.shape[1], frame.shape[0])
            scale = min(1.0, args.max_width / frame.shape[1])
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            timestamp = index / fps
            start = time.perf_counter()
            roi = hands.find(frame.copy(), timestamp) if hands else None
            detection, _ = find_marker(frame, config['hsv_lower'], config['hsv_upper'],
                                       config['min_area_px'] * scale * scale, roi)
            if hands and roi is None:
                detection = None
            elapsed = (time.perf_counter() - start) * 1000
            samples.append(dict(frame=index, timestamp_s=timestamp,
                                centroid_original_px=[p / scale for p in detection[0]] if detection else None,
                                area_original_px=detection[1] / scale ** 2 if detection else None,
                                hand_detected=roi is not None if hands else None,
                                processing_ms=elapsed))
            if timestamp >= next_preview and len(previews) < 6:
                view = frame.copy()
                if detection:
                    cv2.circle(view, tuple(map(int, detection[0])), 12, (0, 0, 255), 3)
                if roi:
                    cv2.rectangle(view, roi[:2], roi[2:], (255, 100, 0), 2)
                label = f'{timestamp:.1f}s | candidate' if detection else f'{timestamp:.1f}s | no candidate'
                cv2.putText(view, label, (12, 30), 0, 0.7, (0, 0, 255), 2)
                previews.append(cv2.resize(view, (640, round(view.shape[0] * 640 / view.shape[1]))))
                next_preview += 2.0
            index += 1
        if not samples:
            raise ValueError('Video has no readable frames')
        summary = dict(video=str(args.video), config=config, fps=fps,
                       original_size=original_size, processing_scale=scale,
                       stride=args.stride, frames_decoded=index, samples=len(samples),
                       detections=sum(s['centroid_original_px'] is not None for s in samples),
                       hand_detections=sum(s['hand_detected'] is True for s in samples) if hands else None,
                       median_processing_ms=float(np.median([s['processing_ms'] for s in samples])),
                       note='Candidate counts are not tracking accuracy. No contact or metric calibration is inferred.')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (args.output / 'detections.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
        cv2.imwrite(str(args.output / 'detections.jpg'), np.vstack(previews))
        print(json.dumps(summary, indent=2))
    finally:
        cap.release()
        if hands:
            hands.close()


if __name__ == '__main__':
    main()

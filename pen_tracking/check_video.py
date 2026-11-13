"""Inspect pen or marker tracking without a camera, calibration, or GUI."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .tracking import find_marker
from .pen import PenTipTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config/marker.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hand-model', type=Path)
    parser.add_argument('--writing-hand', choices=['left', 'right'])
    parser.add_argument('--tracker', choices=['marker', 'pen'], default='marker')
    parser.add_argument('--tip', type=float, nargs=2, help='Initial physical tip in original video pixels')
    parser.add_argument('--anchor', type=float, nargs=2, help='Visible pen-body point near tip in original pixels')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--preview-video', action='store_true', help='Save an annotated MP4')
    parser.add_argument('--stride', type=int, default=5, help='Process every Nth frame')
    parser.add_argument('--max-width', type=int, default=1280)
    args = parser.parse_args()
    if args.stride < 1 or args.max_width < 1:
        parser.error('stride and max-width must be positive')
    if args.start_frame < 0:
        parser.error('start-frame must be nonnegative')
    if args.tracker == 'pen' and (args.tip is None or args.anchor is None):
        parser.error('pen tracking requires --tip X Y and --anchor X Y')
    if args.writing_hand and (args.tracker != 'marker' or not args.hand_model):
        parser.error('--writing-hand requires --tracker marker and --hand-model')
    config = json.loads(args.config.read_text())
    cap = cv2.VideoCapture(str(args.video))
    hands = None
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open {args.video}')
        if args.hand_model and args.tracker == 'marker':
            from .hands import HandRegion
            hands = HandRegion(args.hand_model, args.writing_hand)
        pen = PenTipTracker() if args.tracker == 'pen' else None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError('Video has no usable FPS')
        args.output.mkdir(parents=True, exist_ok=True)
        samples, previews = [], []
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        index, next_preview = args.start_frame, args.start_frame / fps
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.stride and not pen:
                index += 1
                continue
            original_size = (frame.shape[1], frame.shape[0])
            scale = min(1.0, args.max_width / frame.shape[1])
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            timestamp = index / fps
            start = time.perf_counter()
            roi = hands.find(frame.copy(), timestamp) if hands else None
            if pen:
                if index == args.start_frame:
                    point = pen.initialize(frame, [v * scale for v in args.tip],
                                           [v * scale for v in args.anchor], timestamp)
                else:
                    point = pen.update(frame, timestamp)
                detection = (point, None) if point is not None else None
            else:
                detection, _ = find_marker(frame, config['hsv_lower'], config['hsv_upper'],
                                           config['min_area_px'] * scale * scale,
                                           roi if args.writing_hand else None)
                if hands and args.writing_hand and roi is None:
                    detection = None
            elapsed = (time.perf_counter() - start) * 1000
            samples.append(dict(frame=index, timestamp_s=timestamp,
                                centroid_original_px=[p / scale for p in detection[0]] if detection else None,
                                area_original_px=detection[1] / scale ** 2 if detection and detection[1] is not None else None,
                                appearance_score=pen.score if pen else None,
                                tip_appearance_score=pen.tip_score if pen else None,
                                tip_method=pen.tip_method if pen else None,
                                tracking_state=pen.state if pen else ('candidate' if detection else 'missing'),
                                hand_detected=roi is not None if hands else None,
                                processing_ms=elapsed))
            if args.preview_video or (timestamp >= next_preview and len(previews) < 6):
                view = frame.copy()
                if detection:
                    cv2.circle(view, tuple(map(int, detection[0])), 12, (0, 0, 255), 3)
                if roi:
                    cv2.rectangle(view, roi[:2], roi[2:], (255, 100, 0), 2)
                state = pen.state if pen else ('candidate' if detection else 'missing')
                label = f'{timestamp:.1f}s | {state}'
                cv2.putText(view, label, (12, 30), 0, 0.7, (0, 0, 255), 2)
                if args.preview_video:
                    if writer is None:
                        output_fps = fps if pen else fps / args.stride
                        writer = cv2.VideoWriter(str(args.output / 'tracking.mp4'),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), output_fps,
                                                 (view.shape[1], view.shape[0]))
                        if not writer.isOpened():
                            raise ValueError('Could not create preview video')
                    writer.write(view)
                if timestamp >= next_preview and len(previews) < 6:
                    previews.append(cv2.resize(view, (640, round(view.shape[0] * 640 / view.shape[1]))))
                    next_preview += 2.0
            index += 1
        if not samples:
            raise ValueError('Video has no readable frames')
        summary = dict(video=str(args.video), config=config, fps=fps,
                       original_size=original_size, processing_scale=scale,
                       stride=1 if pen else args.stride, start_frame=args.start_frame,
                       tracker=args.tracker, seed_original_px=args.tip,
                       anchor_original_px=args.anchor,
                       frames_decoded=index - args.start_frame, samples=len(samples),
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
        if writer is not None:
            writer.release()
        if hands:
            hands.close()


if __name__ == '__main__':
    main()

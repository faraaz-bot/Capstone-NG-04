import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .tracking import PageCalibration, PointFilter, StrokeRecorder, find_marker
from .pen import PenTipTracker, select_tip


def arguments():
    parser = argparse.ArgumentParser(description='NG04 camera pen tracker')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video', type=Path, help='Replay a video instead of a webcam')
    parser.add_argument('--calibration', type=Path, default=Path('calibration.json'))
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--config', type=Path, default=Path('config/marker.json'))
    parser.add_argument('--hand-model', type=Path, help='Enable MediaPipe hand ROI')
    parser.add_argument('--writing-hand', choices=['left', 'right'])
    parser.add_argument('--tracker', choices=['pen', 'marker'], default='pen')
    parser.add_argument('--output', type=Path, default=Path('sessions'))
    args = parser.parse_args()
    if args.writing_hand and (args.tracker != 'marker' or not args.hand_model):
        parser.error('--writing-hand requires --tracker marker and --hand-model')
    return args


def calibrate(frame, path):
    points = []
    window = 'Calibration: TL, TR, BR, BL; Enter saves, R resets, Esc cancels'

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            view = frame.copy()
            for i, point in enumerate(points):
                cv2.circle(view, point, 6, (0, 0, 255), -1)
                cv2.putText(view, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(20) & 0xff
            if key == 27:
                return None
            if key == ord('r'):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                try:
                    page = PageCalibration(points, (frame.shape[1], frame.shape[0]))
                except ValueError as error:
                    print(error)
                    points.clear()
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(page.as_dict(), indent=2) + '\n')
                return page
    finally:
        cv2.destroyWindow(window)


def main():
    args = arguments()
    config = json.loads(args.config.read_text())
    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    hands = None
    try:
        if not capture.isOpened():
            raise ValueError('Could not open camera/video. Check the device and macOS camera permission.')
        if not args.video:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            capture.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = capture.read()
        if not ok:
            raise ValueError('No frame received')
        size = (frame.shape[1], frame.shape[0])
        if args.calibrate or not args.calibration.exists():
            page = calibrate(frame, args.calibration)
            if page is None:
                return
        else:
            page = PageCalibration(**json.loads(args.calibration.read_text()))
        if page.frame_size != size:
            raise ValueError('Frame size changed. Run again with --calibrate.')
        if args.hand_model and args.tracker == 'marker':
            from .hands import HandRegion
            hands = HandRegion(args.hand_model, args.writing_hand)
        pen = PenTipTracker() if args.tracker == 'pen' else None
        initial_tip = select_tip(frame, pen) if pen else None
        if pen and initial_tip is None:
            return
        args.output.mkdir(parents=True, exist_ok=True)
        session = args.output / (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8])
        session.mkdir()
        metadata = dict(schema_version=1, source='camera', units='mm',
                        started_at=datetime.now(timezone.utc).isoformat(),
                        calibration=page.as_dict(), config=config,
                        input=str(args.video) if args.video else args.camera,
                        requested_fps=30, reported_fps=capture.get(cv2.CAP_PROP_FPS),
                        hand_model=str(args.hand_model) if hands else None,
                        contact_method='manual keyboard', opencv_version=cv2.__version__)
        metadata['tracker'] = args.tracker
        if pen:
            metadata['pen_tracking'] = dict(max_width=pen.max_width, radius=pen.radius,
                                           search_radius=pen.search_radius,
                                           min_score=pen.min_score, max_gap_s=pen.max_gap,
                                           initial_selection=pen.selection,
                                           tip_estimation='2D rotated offset from selected body patch')
        (session / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('Session:', session)
        print('D: down, U: up, X: unknown, R: reselect tip, Q: quit. Contact resets after tracking loss.')
        ink = np.full((891, 630, 3), 255, np.uint8)
        recorder, smoother = StrokeRecorder(), PointFilter()
        contact, previous, previous_id = 'unknown', None, None
        start, frame_id = time.perf_counter(), 0
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        with (session / 'points.jsonl').open('w') as output:
            try:
                while ok:
                    tick = time.perf_counter()
                    timestamp = frame_id / fps if args.video else tick - start
                    # Detect on the original image; draw overlays only afterwards.
                    roi = hands.find(frame.copy(), timestamp) if hands and not pen else None
                    if pen:
                        marker = initial_tip if frame_id == 0 else pen.update(frame, timestamp)
                        detection = None
                        mask = None
                    else:
                        # An arbitrary detected hand is only a visual hint.
                        # Gate colour detections only when a writing hand was selected.
                        search_roi = roi if args.writing_hand else None
                        detection, mask = find_marker(frame, config['hsv_lower'],
                                                      config['hsv_upper'], config['min_area_px'], search_roi)
                        if hands and args.writing_hand and roi is None:
                            detection = None
                        marker = detection[0] if detection else None
                    raw = None
                    if marker:
                        dx, dy = (0, 0) if pen else config['tip_offset_px']
                        raw = page.to_mm((marker[0] + dx, marker[1] + dy))
                    if raw is None:
                        contact = 'unknown'
                    stroke_id = recorder.observe(raw, contact)
                    if stroke_id is None or stroke_id != previous_id:
                        smoother = PointFilter()
                    filtered = smoother.update(raw, timestamp) if raw is not None else None
                    processing_ms = (time.perf_counter() - tick) * 1000
                    row = dict(schema_version=1, source='camera', frame_id=frame_id,
                               timestamp_s=timestamp, units='mm', marker_px=marker if not pen else None,
                               raw_mm=raw, filtered_mm=filtered, contact=contact,
                               stroke_id=stroke_id, valid=raw is not None,
                               marker_area_px=detection[1] if detection else None,
                               processing_ms=processing_ms)
                    row.update(tracker=args.tracker, tip_px=marker if pen else None,
                               appearance_score=pen.score if pen else None,
                               tip_appearance_score=pen.tip_score if pen else None,
                               tip_method=pen.tip_method if pen else None,
                               target_selection=pen.selection if pen else None,
                               tracking_state=pen.state if pen else ('tracking' if marker else 'missing'))
                    output.write(json.dumps(row, allow_nan=False) + '\n')
                    if stroke_id is not None:
                        point = tuple(int(round(v * 3)) for v in filtered)
                        if previous is not None and previous_id == stroke_id:
                            cv2.line(ink, previous, point, (20, 20, 20), 2)
                        else:
                            cv2.circle(ink, point, 1, (20, 20, 20), -1)
                        previous = point
                    else:
                        previous = None
                    previous_id = stroke_id
                    if marker:
                        cv2.circle(frame, tuple(map(int, marker)), 7, (0, 0, 255), 2)
                    if roi:
                        cv2.rectangle(frame, roi[:2], roi[2:], (255, 100, 0), 2)
                    state = pen.state if pen else ('tracking' if marker else 'missing')
                    cv2.putText(frame, f'{contact} | {state} | {processing_ms:.1f} ms | D/U/X, R reselect, Q quits',
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Camera', frame)
                    if mask is not None:
                        cv2.imshow('Marker mask', mask)
                    cv2.imshow('Ink', ink)
                    key = cv2.waitKey(1) & 0xff
                    if key in (ord('q'), 27):
                        break
                    if key == ord('r') and pen:
                        # Reselection changes the target; break ink even if the
                        # user cancels and the old tracker still has a lock.
                        select_tip(frame, pen, timestamp)
                        contact, previous, previous_id = 'unknown', None, None
                        recorder.observe(None, 'unknown')
                    contact = {ord('d'): 'down', ord('u'): 'up', ord('x'): 'unknown'}.get(key, contact)
                    ok, frame = capture.read()
                    frame_id += 1
            finally:
                cv2.imwrite(str(session / 'ink.png'), ink)
    finally:
        capture.release()
        if hands:
            hands.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

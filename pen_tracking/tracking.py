import cv2
import numpy as np


class PageCalibration:
    def __init__(self, corners, frame_size, page_mm=(210.0, 297.0)):
        points = np.asarray(corners, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Calibration needs four finite page corners")
        if not cv2.isContourConvex(points.astype(np.int32)):
            raise ValueError("Click corners in order around a convex page")
        if abs(cv2.contourArea(points)) < 100:
            raise ValueError("Page corners are too close together")
        self.corners = points
        self.frame_size = tuple(frame_size)
        self.page_mm = tuple(page_mm)
        width, height = self.page_mm
        self.matrix = cv2.getPerspectiveTransform(
            points, np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        )

    def to_mm(self, point):
        result = cv2.perspectiveTransform(np.float32([[point]]), self.matrix)[0, 0]
        x, y = map(float, result)
        w, h = self.page_mm
        # Homography arithmetic can put an exact page edge just below zero.
        tolerance = 1e-4
        if -tolerance <= x <= w + tolerance and -tolerance <= y <= h + tolerance:
            return min(w, max(0, x)), min(h, max(0, y))
        return None

    def as_dict(self):
        return dict(corners=self.corners.tolist(), frame_size=self.frame_size,
                    page_mm=self.page_mm)


def find_marker(frame, lower, upper, min_area=20, roi=None):
    """Return marker centroid and contour area; reject tiny blobs."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    if roi is not None:
        x0, y0, x1, y1 = roi
        allowed = np.zeros_like(mask)
        allowed[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, allowed)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        return None, mask
    contour = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(contour)
    point = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
    return (point, float(cv2.contourArea(contour))), mask


class PointFilter:
    """Constant-velocity Kalman filter. Missing observations are never ink."""
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.float32([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.25
        self.last_time = None

    def update(self, point, timestamp):
        measurement = np.float32(point).reshape(2, 1)
        if self.last_time is None:
            self.kalman.statePost = np.float32([[point[0]], [point[1]], [0], [0]])
            self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
            self.last_time = timestamp
            return tuple(map(float, point))
        dt = max(0.001, min(timestamp - self.last_time, 0.2))
        self.last_time = timestamp
        self.kalman.transitionMatrix = np.float32(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * dt
        self.kalman.predict()
        result = self.kalman.correct(measurement)
        return float(result[0, 0]), float(result[1, 0])


class StrokeRecorder:
    def __init__(self):
        self.stroke_id = 0
        self.active = False

    def observe(self, point, contact):
        if contact not in ('down', 'up', 'unknown'):
            raise ValueError('Invalid contact state')
        if point is None or contact != 'down':
            self.active = False
            return None
        if not self.active:
            self.stroke_id += 1
        self.active = True
        return self.stroke_id

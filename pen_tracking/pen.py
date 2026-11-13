"""User-initialized appearance tracking for pens without a colour marker."""
import cv2
import numpy as np


class PenTipTracker:
    """Track an exposed shaft patch and its selected tip offset in image space.

    A fixed reference avoids learning paper or fingers after drift. Short gaps
    allow local recovery; a longer loss requires reselection. Missing points are
    never predicted. The tip is an estimate from a rigid 2D shaft-to-tip offset.
    All internal coordinates use a fixed working image scale.
    """
    def __init__(self, max_width=1280, radius=24, search_radius=40, min_score=0.65,
                 max_gap=0.5):
        self.max_width = max_width
        self.radius = radius
        self.search_radius = search_radius
        self.min_score = min_score
        self.max_gap = max_gap
        self.templates = []
        self.point = None
        self.score = None
        self.state = 'uninitialized'
        self.frame_size = None
        self.last_time = None
        self.angle = 0.0
        self.tip_score = None
        self.tip_method = None

    def _image(self, frame):
        scale = min(1.0, self.max_width / frame.shape[1])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale

    def initialize(self, frame, tip, anchor, timestamp=0.0):
        gray, scale = self._image(frame)
        point = np.asarray(anchor, dtype=np.float32) * scale
        selected_tip = np.asarray(tip, dtype=np.float32) * scale
        r = self.radius
        x, y = point
        if not np.isfinite(point).all() or not (r <= x < gray.shape[1] - r and
                                                r <= y < gray.shape[0] - r):
            raise ValueError('Select a pen-body point at least one patch radius inside the image')
        if (not np.isfinite(selected_tip).all() or
                not (0 <= selected_tip[0] < gray.shape[1] and 0 <= selected_tip[1] < gray.shape[0])):
            raise ValueError('Tip must be inside the frame')
        offset = selected_tip - point
        if not 10 <= float(np.linalg.norm(offset)) <= 200:
            raise ValueError('Select a visible body point 10–200 working pixels from the tip')
        reference = cv2.getRectSubPix(gray, (2 * r + 1, 2 * r + 1), tuple(point))
        if float(reference.std()) < 8:
            raise ValueError('Selected patch lacks contrast; select a visible pen tip')
        templates = {}
        tip_templates = {}
        tip_radius = 10
        if not (tip_radius <= selected_tip[0] < gray.shape[1] - tip_radius and
                tip_radius <= selected_tip[1] < gray.shape[0] - tip_radius):
            raise ValueError('Select a tip at least 10 working pixels inside the image')
        tip_reference = cv2.getRectSubPix(gray, (21, 21), tuple(selected_tip))
        if float(tip_reference.std()) < 5:
            raise ValueError('Selected tip has insufficient contrast')
        for angle in range(-180, 180, 3):
            rotation = cv2.getRotationMatrix2D((r, r), angle, 1)
            templates[angle] = cv2.warpAffine(reference, rotation, reference.shape[::-1],
                                                  borderMode=cv2.BORDER_REFLECT_101)
            tip_rotation = cv2.getRotationMatrix2D((10, 10), angle, 1)
            tip_templates[angle] = cv2.warpAffine(tip_reference, tip_rotation, (21, 21),
                                                       borderMode=cv2.BORDER_REFLECT_101)
        # Commit the selection only after validation so a failed/cancelled
        # reselection leaves the previous target intact.
        self.scale, self.offset = scale, offset
        self.templates, self.tip_templates = templates, tip_templates
        self.selection = dict(tip_px=list(map(float, tip)), anchor_px=list(map(float, anchor)),
                              timestamp_s=timestamp)
        self.point = point
        self.score = 1.0
        self.tip_score = 1.0
        self.tip_method = 'selected'
        self.frame_size = frame.shape[:2]
        self.state = 'tracking'
        self.angle = 0.0
        self.last_time = timestamp
        return tuple(map(float, np.asarray(tip)))

    def update(self, frame, timestamp):
        self.tip_score = None
        self.tip_method = None
        if self.state not in ('tracking', 'uncertain'):
            return None
        if timestamp <= self.last_time or timestamp - self.last_time > self.max_gap:
            self.state = 'lost'
            self.score = None
            return None
        if frame.shape[:2] != self.frame_size:
            self.state = 'lost'
            self.score = None
            return None
        gray, scale = self._image(frame)
        r, distance = self.radius, self.search_radius
        x, y = map(int, self.point)
        x0, y0 = max(0, x - distance - r), max(0, y - distance - r)
        x1 = min(gray.shape[1], x + distance + r + 1)
        y1 = min(gray.shape[0], y + distance + r + 1)
        search = gray[y0:y1, x0:x1]
        best, location = -1.0, None
        if min(search.shape) >= 2 * r + 1:
            for angle, template in self.templates.items():
                delta = (angle - self.angle + 180) % 360 - 180
                if abs(delta) > 15:
                    continue
                response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                # A circular gate avoids jumps to diagonal corners of the search box.
                yy, xx = np.indices(response.shape)
                valid = (xx + x0 + r - self.point[0]) ** 2 + (
                    yy + y0 + r - self.point[1]) ** 2 <= distance ** 2
                response[~valid] = -1
                _, score, _, pos = cv2.minMaxLoc(response)
                if score > best:
                    best, location = score, np.float32([x0 + pos[0] + r, y0 + pos[1] + r])
                    best_angle = angle
        self.score = float(best)
        if best < self.min_score or location is None:
            self.state = 'uncertain'
            return None
        self.point = location
        self.angle = best_angle
        self.state = 'tracking'
        rotation = cv2.getRotationMatrix2D((0, 0), best_angle, 1)[:, :2]
        tip = location + rotation @ self.offset
        if not (0 <= tip[0] < gray.shape[1] and 0 <= tip[1] < gray.shape[0]):
            self.state = 'uncertain'
            return None
        # Refine the rigid-offset estimate when the original nib appearance
        # remains visible. Lighting/background changes may invalidate the nib
        # patch while the observed body still has a usable lock.
        tx, ty = map(int, tip)
        pad, tr = 24, 10
        sx0, sy0 = max(0, tx - pad - tr), max(0, ty - pad - tr)
        sx1 = min(gray.shape[1], tx + pad + tr + 1)
        sy1 = min(gray.shape[0], ty + pad + tr + 1)
        search = gray[sy0:sy1, sx0:sx1]
        best_tip, found_tip = -1, None
        if min(search.shape) >= 21:
            for angle, template in self.tip_templates.items():
                if abs((angle - self.angle + 180) % 360 - 180) > 12:
                    continue
                response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, pos = cv2.minMaxLoc(response)
                if score > best_tip:
                    best_tip = score
                    found_tip = np.float32([sx0 + pos[0] + tr, sy0 + pos[1] + tr])
        self.tip_score = float(best_tip)
        self.last_time = timestamp
        if best_tip >= 0.8 and found_tip is not None:
            self.tip_method = 'local appearance'
            return tuple(map(float, found_tip / scale))
        self.tip_method = 'body offset'
        return tuple(map(float, tip / scale))


def select_tip(frame, tracker, timestamp=0.0):
    """Select on a paused, scaled image. Escape cancels without an observation."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width)
    view = cv2.resize(frame, (round(width * scale), round(height * scale)))
    window = 'Click tip, then visible pen body near tip; Esc cancels'
    selected = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 2:
            selected.append((x / scale, y / scale))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            display = view.copy()
            for x, y in selected:
                cv2.circle(display, (round(x * scale), round(y * scale)), 6, (0, 0, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xff
            if key == 27:
                return None
            if len(selected) == 2:
                try:
                    return tracker.initialize(frame, selected[0], selected[1], timestamp)
                except ValueError as error:
                    print(error)
                    selected.clear()
    finally:
        cv2.destroyWindow(window)

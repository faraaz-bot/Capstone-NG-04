import cv2
import mediapipe as mp


class HandRegion:
    def __init__(self, model_path, writing_hand=None):
        self.writing_hand = writing_hand
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_ms = -1

    def find(self, frame, timestamp):
        self.last_ms = max(self.last_ms + 1, int(timestamp * 1000))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.last_ms)
        if not result.hand_landmarks:
            return None
        height, width = frame.shape[:2]
        points = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            if self.writing_hand and handedness[0].category_name.lower() != self.writing_hand:
                continue
            points.extend((int(p.x * width), int(p.y * height)) for p in landmarks)
        if not points:
            return None
        # Allow space beyond the fingers for the pen and its marker.
        margin = 100
        return (max(0, min(x for x, _ in points) - margin),
                max(0, min(y for _, y in points) - margin),
                min(width, max(x for x, _ in points) + margin),
                min(height, max(y for _, y in points) + margin))

    def close(self):
        self.detector.close()

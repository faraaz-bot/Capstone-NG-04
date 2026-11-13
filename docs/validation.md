# Prototype checks — November 12, 2026

Environment: macOS arm64, Python 3.9.6, NumPy 1.26.4, OpenCV 4.11.0,
MediaPipe 0.10.21. Installed into this project's `.venv`.

`python -m unittest discover -s tests -v`: six tests passed. These cover
synthetic green-marker detection with an ROI and distractor, page calibration,
invalid corner order, reduced stationary jitter, stroke interruption, and a
generated six-frame video through the recording application. The replay checks
session JSON and PNG output and confirms that missing tracking resets contact.
Display functions are mocked in that replay test.

MediaPipe loaded the downloaded Hand Landmarker model and returned no hand on a
blank 640×480 image. On this Mac it required macOS graphics access outside the
execution sandbox. This checks initialization and one inference call, not real
hand detection quality.

`pip check` reports MediaPipe 0.10.21 as unsupported on this platform: the
downloaded universal2 wheel contains an internal `WHEEL` tag for x86_64. Its native
module nevertheless imports and the inference check above succeeds on this arm64
Mac. This packaging warning remains unresolved; third-party metadata was not edited.

The command-line help runs successfully. No physical webcam session, pen-contact
accuracy test, page measurement or end-to-end latency trial has been performed.

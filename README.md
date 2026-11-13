# NG04 pen tracking

Computer vision prototype for the vision-based touchless control project. It tracks
a coloured marker on a pen, maps its position onto an A4 page, and records strokes.

## Run

Python 3.9 or newer. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[hands]'
python -m pen_tracking.app --calibrate
```

Use an overhead webcam with a fixed mount, an A4 page in portrait orientation, and
a green marker near the pen tip. Keep the whole page visible. Click the page's
top-left, top-right, bottom-right and bottom-left corners, then press Enter.
Press R to redo the corners or Escape to cancel. Calibration is saved locally;
repeat it whenever the camera, page or resolution changes.

The camera requests 1920×1080 at 30 fps; the actual device may provide less.
On macOS, allow camera access for the terminal/application running Python when
prompted. Use `--camera 1` to try another camera.

In the camera window:

- D marks contact as down; U marks it up; X marks it unknown.
- Q or Escape stops recording.
- Losing the marker, hand ROI or page position resets contact to unknown.
  Press D again only when contact is confirmed.

The keyboard control is a development aid, not an automatic pen-contact detector.
Without reliable contact evidence the prototype cannot determine physical contact
from marker position alone.

## Marker and pen tip

Edit `config/marker.json` for your marker colour and lighting. The defaults select
green. OpenCV hue ranges from 0–179; saturation and value range from 0–255. Watch
the mask window and adjust the bounds until it isolates the marker. The largest
remaining contour above `min_area_px` is selected, so remove similarly coloured
objects from the view.

`tip_offset_px` is `[dx, dy]` from marker centre to the projected physical pen tip.
Zero offset tracks the marker centre. Measure the offset before assessing tip
accuracy. A fixed image offset is only an approximation for a fixed pen pose;
rotating or tilting the pen changes it. Homography also assumes a point on the
writing plane, so an elevated marker introduces parallax.

## Optional hand tracking

Download Google's model into the ignored `models` directory:

```sh
mkdir -p models
curl -fL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o models/hand_landmarker.task
python -m pen_tracking.app --hand-model models/hand_landmarker.task
```

MediaPipe restricts marker detection to a padded box around one detected hand.
No detected hand means no accepted point. Leave `--hand-model` out to debug marker
tracking alone. The dependency versions support the installed Python 3.9 and avoid
installing two competing OpenCV packages.

## Saved data

Each run creates an ignored `sessions/<timestamp>-<id>/` folder:

- `metadata.json`: calibration, marker settings, input, time and reported frame rate.
- `points.jsonl`: one record per frame, including raw marker pixels, raw and filtered
  page coordinates in millimetres, contact state, stroke ID and processing time.
- `ink.png`: the accepted strokes at 3 pixels/mm.

Coordinates originate at the page's top-left; x points right and y points down.
Only observed in-page points with manual down contact produce ink. Null stroke IDs
must never be connected. The filter resets across stroke boundaries and missing
observations; predictions alone are never emitted as ink. `valid` means a usable
position, independently of contact. Marker area is a diagnostic, not a calibrated
confidence probability.

Webcam timestamps use elapsed monotonic time at processing. Video replay uses
frame index divided by reported FPS. `processing_ms` measures tracking computation,
including hand inference if enabled; it excludes capture, display, and total
end-to-end latency. Raw camera video is not saved. For repeatable trials, record
video separately and run `--video path/to/trial.mp4 --calibrate` with matching page
geometry. Contact labels remain manual during playback.

## Check

```sh
python -m unittest discover -s tests -v
```

The tests exercise synthetic marker detection, perspective mapping, stroke
boundaries and stationary jitter. They do not establish physical accuracy.
See [the project notes](docs/mcr-scope.md) for requirements and remaining work.

References: [OpenCV HSV thresholding](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html),
[OpenCV perspective transforms](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html),
and [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

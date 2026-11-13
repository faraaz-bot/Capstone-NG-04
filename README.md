# NG04 pen tracking

Computer vision prototype for the vision-based touchless control project. It tracks
a selected ordinary pen or a coloured marker, maps its position onto an A4 page,
and records strokes.

## Run

Python 3.9 or newer. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pen_tracking.app --calibrate
```

Use an overhead webcam with a fixed mount and an A4 page in portrait orientation.
Keep the whole page visible. Click the page's
top-left, top-right, bottom-right and bottom-left corners, then press Enter.
Press R to redo the corners or Escape to cancel. Calibration is saved locally;
repeat it whenever the camera, page or resolution changes.

Next, click the physical pen tip, then a visible point on the pen body near the
tip. Choose a sharp, distinctive exposed section rather than a finger or paper.
The selected body point should be about 20–80 pixels from the tip in the displayed
image. The two clicks identify which object to track; no green tape or hand model
is needed in this default `--tracker pen` mode.

The tracker searches for the selected body patch under translation and rotation,
then verifies the nib in a small nearby region. It preserves its original reference
instead of learning paper or fingers after drift. It can recover a short tracking
gap locally; after 0.5 seconds without an accepted point, press R to select again.
Fully obscured pens and large pose/scale changes can still require reselection.

The camera requests 1920×1080 at 30 fps; the actual device may provide less.
On macOS, allow camera access for the terminal/application running Python when
prompted. Use `--camera 1` to try another camera.

In the camera window:

- D marks contact as down; U marks it up; X marks it unknown.
- R pauses on the current frame to select the tip and body point again.
- Q or Escape stops recording.
- Losing the pen, marker, hand ROI or page position resets contact to unknown.
  Press D again only when contact is confirmed.

The keyboard control is a development aid, not an automatic pen-contact detector.
Without reliable contact evidence the prototype cannot determine physical contact
from image position alone.

## Marker and pen tip

The original HSV mode remains available as `--tracker marker`. It requires a
distinctive coloured marker; it is not an ordinary-pen detector.

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
python -m pen_tracking.app --tracker marker --hand-model models/hand_landmarker.task
```

Install the optional dependency with `python -m pip install -e '.[hands]'` first.
MediaPipe detects up to two hands and displays their combined padded bounds as a
visual hint. It does not silently treat an arbitrary detected hand as the writing
hand. Set `--writing-hand left` or `--writing-hand right` to enable a strict search
region using MediaPipe's handedness labels. Check the box visually: handedness can
depend on image mirroring. In strict mode no matching hand means no accepted point. The hand
model is ignored in ordinary-pen mode, which tracks the explicitly selected pen.

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

Ordinary-pen records use `tip_px`, `appearance_score`, `tip_appearance_score` and
`tracking_state` (`tracking`, `uncertain` or `lost`). `marker_px` is null in that mode.
Scores are image-match diagnostics, not calibrated probabilities. The body-to-tip
offset is rotated in 2D and refined with a local nib match when the nib's appearance
remains similar. `tip_method` records `local appearance` or `body offset`; the
latter is an estimate from the currently observed body, not a directly localized
nib. Neither is a 3D pen-pose reconstruction. Pose/scale changes can affect accuracy.

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

The tests exercise synthetic marker and ordinary-pen tracking, rotation, occlusion,
loss/reselection, session recording, perspective mapping and stroke boundaries.
They do not establish physical accuracy.
See [the project notes](docs/mcr-scope.md) for requirements and remaining work.

Active video testing uses only bird's-eye footage looking down at the writing
surface. See the [bird's-eye trials](docs/birds-eye-trials.md) for the current set.
The [initial video trials](docs/internet-video-trials.md) and
[ordinary-pen follow-up](docs/ordinary-pen-trials.md) are historical records that
also contain angled clips excluded from the current scope.

References: [OpenCV HSV thresholding](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html),
[OpenCV perspective transforms](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html),
and [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

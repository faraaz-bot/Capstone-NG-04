# Ordinary-pen tracking follow-up

Historical record: the silver- and blue-pen clips below use angled views and are
excluded from the active bird's-eye-only evaluation. See
[the current trial set](birds-eye-trials.md). Their previous results are retained
for traceability and must not be included in current performance summaries.

## Cause and change

The first implementation was a green-marker detector, not an ordinary-pen
detector. Its largest-colour-contour rule selected the blue desk object when the
thresholds were changed to blue. Its one-hand MediaPipe setup also selected the
non-writing hand and used that box to exclude the actual pen.

The default camera mode now uses a selected pen's appearance. Click the physical
tip and a distinctive exposed part of the body near it. The tracker matches the
body reference under translation and rotation, rotates the selected tip offset,
and optionally refines the nib position with a nearby appearance match. The fixed
body reference avoids adapting a drifted tracker to paper. No green marker or hand
model is required. This is user-initialized tracking, not automatic pen discovery.

Missing body appearance produces no position. A local search can recover for up
to 0.5 seconds; a longer loss requires reselection with R. Contact resets to unknown
on every missing observation, so recovery cannot silently connect strokes.

The original marker mode is explicitly selected with `--tracker marker`. Hand
tracking now requests two hands, displays a hint by default, and only restricts
the marker search when `--writing-hand left/right` is supplied. This avoids an
arbitrary detected hand silently vetoing the pen. A real-model check of the first
overhead frame still detected only the non-writing hand; requesting two hands is
not a guarantee that both will be detected. Explicit strict selection fails closed
if the requested hand is absent.

## Results on the same downloaded clips

The new tracker processed every frame, initialized once on the first frame, with
the same settings across clips. Frames were resized to width 1280 for tracking.
The previous marker trials sampled every fifth frame, so their raw counts have
different denominators. All sampled original green-marker trials had zero candidates.

| Clip | Frames with estimate / total | Frame coverage | Manual spot checks at width 1280 |
| --- | ---: | ---: | --- |
| Black pen, 4668324 | 361 / 373 | 96.8% | 5/5 labelled frames returned estimates; median error 11.3 px, maximum 19.8 px |
| Silver pen, 5391294 | 740 / 754 | 98.1% | 4/5 labelled frames returned estimates; median error 10.4 px, maximum 13.2 px among returned estimates |
| Blue pen, 9573567 | 120 / 291 | 41.2% | Follows the initially visible pen, then loses lock around the pickup/occlusion; requires reselection afterwards |

The blue trial's six preview frames were inspected: its initial estimates stay on
the pen, with no estimate in the later previews instead of a lock on the desk
object. This is not a claim of correct tracking through occlusion. The black and
silver previews and five non-initialization frames per clip were also checked.

The spot labels are approximate visual estimates on unannotated source frames with
a coordinate grid, with about 5 px annotation uncertainty. They were created during
development and used to assess refinements, not held out. See
[pen-tip-spot-labels.json](pen-tip-spot-labels.json) and
[ordinary-pen-results.json](ordinary-pen-results.json) for coordinates, missing
estimates and errors. Spot-check coverage is reported separately from errors so
missing predictions do not disappear from the evaluation.

Frame coverage is not accuracy. This small check does not establish 1 mm precision,
performance on new footage, automatic pen contact, or recognition quality. The
rotated body-to-tip offset approximates 2D pose; large perspective, scale or grip
changes can require reselection and still cause spatial error. `tip_method` records
whether an estimate used local nib appearance or the observed body's rotated offset.

## Reproduce and watch

Sources and download URLs remain in [video-sources.json](video-sources.json).
From the project folder after activating `.venv`:

```sh
python -m pen_tracking.check_video test-media/pexels-4668324.mp4 --tracker pen --tip 1048.5 712.5 --anchor 1084.5 787.5 --preview-video --output reports/fixed-black
python -m pen_tracking.check_video test-media/pexels-5391294.mp4 --tracker pen --tip 3099 1716 --anchor 3036 1605 --preview-video --output reports/fixed-silver
python -m pen_tracking.check_video test-media/pexels-9573567.mp4 --tracker pen --tip 2432 976 --anchor 2272 1002 --preview-video --output reports/fixed-blue
python -m pen_tracking.evaluate_spots reports/fixed-black --clip-id 4668324
python -m pen_tracking.evaluate_spots reports/fixed-silver --clip-id 5391294
```

Tip and anchor arguments are in original video pixels. Each output folder includes
an annotated `tracking.mp4`, a six-frame preview sheet, per-frame records, and a
summary. `tip_px` in camera sessions and `centroid_original_px` in diagnostics are
pixel estimates before page homography. The selected blue-pen point is its right
endpoint. No page dimensions or physical contact labels were invented for these clips.

Thirteen automated checks pass, including translation/rotation of an ordinary pen,
background distractors, missing points under occlusion, local recovery and
reselection, and camera-session stroke boundaries. Physical webcam precision and
labelled pen-contact trials remain necessary for the MCR performance targets.

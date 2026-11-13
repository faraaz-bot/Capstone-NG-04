# Internet video trials

Historical record: current testing is restricted to bird's-eye footage. The
angled clips in this report are excluded from the active set; see
[the current trial set](birds-eye-trials.md).

Three Pexels clips were downloaded and tested on November 12, 2026. Source pages,
creators and exact download URLs are in [video-sources.json](video-sources.json).
The [Pexels license](https://www.pexels.com/license/) allows free download and use.
Videos and generated previews stay in ignored local folders rather than Git.

## Clips and suitability

- [4668324 — Vlada Karpovich](https://www.pexels.com/video/person-writing-on-a-paper-with-a-pen-4668324/):
  the best of these for overhead handwriting, with a black pen, visible pen lifts
  and two hands. The small page moves. It is not a calibrated A4 scene and has no
  coloured tracking marker.
- [5391294 — Gustavo Fring](https://www.pexels.com/video/a-person-writing-on-paper-5391294/):
  a silver pen from an oblique angle. Useful for motion and occlusion inspection,
  but unsuitable for validating green-marker tracking or page-scale accuracy.
- [9573567 — Tima Miroshnichenko](https://www.pexels.com/video/close-up-video-of-a-person-writing-on-a-paper-9573567/):
  a light-blue pen, substantial hand occlusion, and a blue/purple desk object.
  Much of the clip has the pen lying on the page. Useful for testing false matches
  when the selected colour also appears in the background.

## Measured observations

The existing `find_marker` function processed every fifth frame at a maximum width
of 1280 pixels. The contour-area threshold was scaled with image area. The sampled
frames are about 0.17–0.20 seconds apart; these are not full-rate tracking trials.

| Clip | Decoded frames | Sampled frames | Green candidates | Other observation |
| --- | ---: | ---: | ---: | --- |
| 4668324 | 373 | 75 | 0 | MediaPipe found one hand in all 75 samples |
| 5391294 | 754 | 151 | 0 | Silver pen has no green marker |
| 9573567 | 291 | 59 | 0 | Blue settings produced 59 candidates in 59 samples |

Visual inspection of six evenly spaced blue-trial previews showed the candidates
on the blue/purple desk object, not the pen. Therefore 59/59 candidates must not be
reported as successful pen tracking. Only those six previews were manually reviewed;
there are no per-frame pen-tip labels or accuracy scores.

The overhead clip's six MediaPipe previews show its ROI on the left, non-writing
hand. The current `num_hands=1` implementation does not identify which hand holds
the pen. Hand detection alone is not evidence of writing-hand tracking.

The green detector's zero candidate counts are consistent with these clips having
no green tracking marker. They do not establish that the detector works on positive
examples. No pen-tip reconstruction, contact classification, handwriting recognition,
millimetre accuracy or end-to-end latency was established by these tests.

## Replay

Download the source files to the paths listed in `video-sources.json`, then run:

```sh
source .venv/bin/activate
python -m pen_tracking.check_video test-media/pexels-4668324.mp4 --output reports/overhead-green
python -m pen_tracking.check_video test-media/pexels-4668324.mp4 --hand-model models/hand_landmarker.task --output reports/overhead-hands
python -m pen_tracking.check_video test-media/pexels-9573567.mp4 --config config/blue-pen-trial.json --output reports/blue-trial
```

Each output directory contains `summary.json`, `detections.jsonl` and
`detections.jpg`. Red circles mark colour candidates; blue boxes mark the hand ROI.
Coordinates in the JSON are mapped back to the original video's pixel dimensions.
This diagnostic deliberately avoids assigning page dimensions or inventing contact
labels for internet footage. The blue configuration is exploratory, not a validated
replacement for the project's green marker settings.

## Next steps supported by this trial

1. Record a positive example with a distinctive green marker, fixed overhead camera,
   stationary A4 page, and labelled pen lifts. The downloaded clips do not provide
   this setup.
2. Select the writing hand explicitly or evaluate both hands against marker evidence.
3. Reject similarly coloured background objects using size, shape and temporal
   consistency. Retuning hue alone did not solve this clip.

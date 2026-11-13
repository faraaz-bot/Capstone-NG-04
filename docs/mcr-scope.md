# MCR III camera work

Source: `Downloads/NG04_MCR_III_Semester2_10_Week_Plan (1) (1).docx`,
downloaded November 12, 2026. The other copy downloaded that day has identical bytes.
SHA-256: `43cc05aed7fc9f69f7fd196c44356cb225e5ce2982c9b39d608ac3669d41695c`.

This folder starts Mohammed Faraaz Mustafa's computer vision section. The MCR
describes preliminary implementation progress, but no existing implementation or
experimental records have been imported here. This repository begins a new prototype.

The selected approach is an overhead USB camera, MediaPipe hand landmarks, OpenCV
HSV marker segmentation, page homography, Kalman smoothing, and down/up/unknown
contact states. JSON records provide the handoff to the handwriting work.

## Current implementation

- Live webcam and video-file input.
- Four-corner A4 calibration with frame-size checks.
- Configurable HSV marker segmentation and fixed marker-to-tip pixel offset.
- Optional MediaPipe hand region, using the Tasks API and a separate model asset.
- Kalman smoothing, manual contact controls, and stroke breaks on missing points.
- Timestamped raw/processed JSON records and a PNG stroke preview.

## Next physical trials

1. Fix the camera and lighting; tune marker thresholds and measure the tip offset.
2. Track printed reference points distributed across the page. Save known positions
   and compare raw and filtered errors in mm, including different pen angles.
3. Label video for marker visibility and real contact. Compare hand ROI enabled
   versus disabled; count missed detections and false marker selections.
4. Develop automatic contact evidence and compare it against labels. Keep unknown
   as a real state rather than drawing through uncertainty.
5. Measure visible-frame tracking rate, false stroke connections at pen lifts,
   actual capture FPS, and end-to-end latency separately from processing time.

MCR targets, still unverified: preferred 1080p at at least 30 fps, median physical
tip error at most 1 mm, tracking on at least 95% of visible frames, and tracking
latency at most 100 ms. Shared digital-ink targets include 95th-percentile display
delay at most 150 ms and false connections on at most 1% of annotated pen-up
transitions. Do not report these as achieved without saved measurements.

Automatic contact classification, quantitative evaluation tooling, dynamic pen-tip
pose correction, and the agreed cross-team schema remain open. TrOCR, wearable
electronics, and the shared application belong to the other workstreams.

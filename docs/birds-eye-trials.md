# Bird's-eye video trials

Only footage looking down at the writing surface is included in active testing.
Inspect multiple frames before admitting a clip: stock-video descriptions using
"overhead" do not reliably establish a bird's-eye camera view.

## Active clips

- [4668324, Vlada Karpovich](https://www.pexels.com/video/person-writing-on-a-paper-with-a-pen-4668324/):
  overhead black-pen writing on a small page. Previously tested with 361/373 frames
  returning estimates. Five development spot checks had 11.3 px median error at a
  1280 px viewing width. This is coverage, not an accuracy score or proof of 1 mm precision.
- [3251846, fauxels](https://www.pexels.com/video/a-woman-writing-on-a-notebook-over-a-table-3251846/):
  bird's-eye flat-lay workspace with a notebook and silver pen. Camera view checked
  at 0, 3, 6 and 9 seconds before tracking. The unchanged tracker returned estimates
  on only 5/352 frames after first-frame initialization, then lost its appearance
  lock during the initial motion/grip change. Later writing also obscures the tip.
  This is an unresolved failure case, not a successful tracking demonstration.

Do not combine the two frame counts into an accuracy claim. No tip labels have
been collected for the second clip. Raw summaries are in
[birds-eye-results.json](birds-eye-results.json). Source URLs, hashes and the
active/excluded split are in [video-sources.json](video-sources.json).

## Excluded footage

5391294 and 9573567 are angled silver- and blue-pen views. Their older results stay
in historical reports and are excluded from current performance summaries.
Candidate 6930351 was downloaded for viewpoint inspection only: despite its
"overhead" description, frames show an oblique over-the-shoulder view. It was
rejected before tracking. No new tracking runs used these excluded clips.

## Replay

```sh
source .venv/bin/activate
python -m pen_tracking.check_video test-media/pexels-4668324.mp4 --tracker pen --tip 1048.5 712.5 --anchor 1084.5 787.5 --preview-video --output reports/fixed-black
python -m pen_tracking.check_video test-media/pexels-3251846.mp4 --tracker pen --tip 1908 1128 --anchor 2010 1146 --preview-video --output reports/birds-eye-3251846
```

The tip and anchor values are original video pixels. Local output folders contain
annotated MP4s, preview sheets and per-frame data. Downloads and rendered footage
remain outside Git. New clips must satisfy the bird's-eye restriction even if an
angled video would give better tracking scores.

# Travel-vlog candidate workflow

This fork adds a review-first selector, not a promise of automatic cinematic
storytelling. The upstream general-purpose pipeline remains unchanged; use
the dedicated travel entry point to avoid its per-file peak normalization.

1. Keep originals local. Create small FFmpeg proxies with correct display
   orientation, preserving source-relative seconds. A camera can physically
   roll mid-recording; split those ranges and record their source offsets.
2. Install semantic dependencies (`pip install -e '.[semantic]'`). CLIP model
   weights are downloaded once; analysis itself runs locally.
   The original S3 bucket is unavailable. This fork uses pinned Xenova CLIP
   quantized ONNX weights, verified with SHA256, without remote inference.
   Set `CLIPSHOW_MODEL_DIR` if the model cache should live on a data drive.
3. Run `python -m clipshow.travel proxy1.mp4 proxy2.mp4 --output review.json`.
   Scenic/canoeing/town prompts contrast with lens-obstruction, ground and
   poor-framing prompts. Output retains technical measurements and scores.
   Add `--photo-dir PATH --day 2026-08-21` to include same-day photos. EXIF
   capture time takes priority over DJI filename; copy/modification time is
   never used. Photos are candidates with a suggested four-second hold, not
   rendered movies. The downstream vlog editor handles motion/transitions.
4. Review entire candidate shots, including start/end and the actual 9:16 crop.
   Reject occlusion, excessive shake, incorrect rotation and near-duplicates.
   CLIP is not a reliable horizon/hand detector; 2 FPS can miss brief defects.
5. Group approved shots by the real day's activities and maintain chronology.
   Mix landscape, activity and human detail; use meaningful photos if needed.
   Do not repeat weak clips merely to fill five minutes.
6. Keep complete dialogue phrases, reduce wind conservatively, duck BGM and
   verify decoded audio coverage to the end (not just container duration).

The semantic fix interpolates measurements onto the output time base and
retains an absolute positive-minus-negative margin. Its sigmoid mapping is a
heuristic, NOT a calibrated probability or an aesthetic model. Thresholds
need evaluation on the actual footage. Unit tests alone do not validate art.

Never commit private footage, photos, BGM, model weights or local review JSON.

For an archive catalog, begin with `python -m clipshow.catalog_review --catalog
PATH --day YYYY-MM-DD --ffmpeg FFMPEG_PATH --output-dir LOCAL_REVIEW_PATH`.
The supplied day is an exact whitelist. Camera-clock corrections come from the
catalog's explicit `DateBasis`, never from file copy time. The coarse three-second
scan is for finding candidates; do a denser scan and full-shot playback before
approving final cuts.

# Four-engine integration validation

Validated locally on Windows with Python 3.12.14, NumPy 1.26.4 and OpenCV
contrib 4.11.0. Versions of external engines are pinned in
[upstream notices](vlog-upstreams.md). Original travel media, BGM, local YAML,
raw AI responses and generated media remain outside the Git commit.

Checks completed:

- `pip check`: no broken requirements.
- Existing application and new compiler tests, including the real FFmpeg
  synthetic renderer regression: 525 passed, 21 optional tests skipped.
- Wheel packaging: successful.
- Full repository Python lint and changed-file whitespace checks: successful.
- Actual Day2 analysis smoke test: three catalog videos and the same-day photo
  directory, real Katna keyframes matched back to timestamps, real ONNX CLIP
  inference, and actual VideoHighlighter/Ollama image inspection. Four machine
  candidates passed (one video and three photos); other footage was not forced
  into the test timeline.
- Actual AutoCut smoke test using the supplied, fully decodable local FLAC:
  30-second normalized test bed, 103.359375 BPM and 46 detected beats. The
  chronological review plan contains four shots and lasts about 26.36 seconds.
  This is intentionally a small **integration test**, not the full Day2 edit.
- Synthetic render: nine shots, two photos, protected source audio, two assembly
  chunks and a short BGM crossfaded repeatedly. Final audio spans the complete
  video; late-audio RMS, portrait size and full video/audio decode QC pass.
- Explicit dialogue edge cropping: AI inspection and rendering use the same
  reviewed framing. Unit tests reject invalid zoom and verify removal of an
  edge intrusion; the real FFmpeg regression exercises the modest crop.
- Visual response compatibility: known mixed labels retain their label list
  without changing an actual model veto. Unknown labels, non-boolean gates,
  non-finite confidence and empty summaries still fail structured validation.

The test fixtures validate mechanics, not highlight aesthetics, wind removal
quality or storytelling. The smoke-test inspection above was explicitly a
subset. The full production run below is separate from that integration test.

## Full Day2 production

The local production run analyzed all 34 catalog videos for the requested day
with actual Katna extraction and ONNX CLIP scoring, and scanned the local photo
directory using capture-date metadata. VideoHighlighter inspected 102 candidate
windows/photos through the installed loopback-only vision model; 101 candidates
passed the structured machine gates. Pinned AutoCut performed audio beat
analysis and generated the initial highlight timeline using the supplied FLAC.
No cloud media upload or model download was required.

An editorial subset retains 50 unique shots, including seven same-day photos
and four protected dialogue passages totaling 85.85 seconds. A roughly
three-second lake opening precedes capture chronology and a same-day closing
postcard. The reviewed, beat-realigned timeline lasts 577.7772 seconds; source
identity/date checks and non-overlap checks pass. Original media and earlier
exports are preserved.

Sampled source frames and actual rendered frames were inspected. One machine-
approved window had a material finger obstruction in its middle and was
removed after rendered-frame review. This is evidence that machine approval
does **not** certify every frame or replace editorial review. Sampled review
is not a claim of continuous viewing of every original shot.

The production core passes full video/audio decoding, 1080-by-1920 dimensions,
decoded-audio duration checks and five-second non-silence checks, including the
second half. Its video is 577.7772 seconds and its decoded audio is 577.77075
seconds. Source sound is muted outside protected dialogue. Speech uses
high/low-pass filtering, adaptive FFT denoising, compression and loudness
normalization; the supplied music bed is crossfaded to cover the whole edit
and ducked beneath dialogue. These technical checks do not guarantee perfect
restoration of wind-damaged speech or subjective highlight quality.

Finished master/share exports also pass the full decode, portrait and complete-
audio checks. Chapter/closing titles were reviewed from rendered frames. A
private finishing pass adds a 220 Hz high-pass only to the windier glass-walkway
dialogue, before music ducking/mixing; the music bass is not filtered. Interior
voice samples show approximately 5.85, 6.99, 8.94 and 11.86 dB lower 20--150 Hz
energy relative to the 300--3500 Hz band versus corresponding original voice
samples. This is a low-frequency rumble measurement, **not** proof that all wind
noise is removed or every word recovered. Local production evidence, EDL and
per-export QC remain outside Git along with the private media.

VideoHighlighter originally returned empty output because its markdown-fence
stop rule intercepted fenced JSON. The isolated adapter permits such fences
and still validates strict structured model output. Missing results and invalid
JSON fail closed, and transient HTTP retries are retained in the evidence.

`bd` is not installed on this Windows host. No beads synchronization is claimed.
Git commits/pushes and these verification records are used for this session.

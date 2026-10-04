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
quality or storytelling. The local Day2 inspection was explicitly a subset;
it does not certify all 34 original videos. A full Day2 production run and
whole-shot story/framing/dialogue review remain separate from tool integration.

VideoHighlighter originally returned empty output because its markdown-fence
stop rule intercepted fenced JSON. The isolated adapter permits such fences
and still validates strict structured model output. Missing results and invalid
JSON fail closed, and transient HTTP retries are retained in the evidence.

`bd` is not installed on this Windows host. No beads synchronization is claimed.
Git commits/pushes and these verification records are used for this session.

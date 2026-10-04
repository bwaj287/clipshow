# Local four-engine Vlog compiler

The Vlog entry point enables **all four real engines**. These are required
dependencies, not names assigned to substitute heuristics. Classic GUI and
`--auto` behavior is unchanged.

| Stage | Actual implementation | Used for |
| --- | --- | --- |
| Keyframes | Katna 0.9.2 `Video.extract_video_keyframes` | Selected frames and sharpness/technical gates; selected images are matched back to measured proxy timestamps |
| Semantic scoring | clipshow `SemanticDetector` + verified ONNX `LocalClip` | Full-window positive/negative travel scores and same-day photo scores |
| Content inspection | VideoHighlighter `LLMModule.query`, local Ollama **with an image** | Three-frame chronological contact sheet per candidate, or an EXIF-oriented photo; obstruction/usability veto and visual summary |
| Selection and music | AutoCut `AudioAnalyzer` + `BeatSyncTimelineGenerator` | Real beat timestamps, upstream quality ranking/selection and custom chronological phrase-length arrangement |

The adapter replaces AutoCut's one-beat-per-shot arrangement, disables
chronological shuffling, and measures alignment against actual output cuts.
It does not pretend every minimum-duration-clamped cut lands on a beat.
Protected dialogue ranges are not shortened to chase a beat or fill a target.
Upstream source remains unmodified and is not copied into this repository.

## Install

Use Python **3.12**, Git, FFmpeg and ffprobe. From this fork:

```bash
python scripts/setup_vlog.py
```

The helper creates `.vlog-tools/env` and pinned external checkouts in
`.vlog-tools`. It refuses to reset an existing checkout at a different revision.
The compiler also refuses modified tracked upstream source. The existing app
declares GUI packages as base dependencies, so they are installed for consistent
package metadata, but Vlog commands do not start a GUI. Torch and the large
VideoHighlighter desktop/OpenVINO stack are not required for this interface.

NumPy/OpenCV/scikit-learn versions are constrained for Katna and ONNX
compatibility. Several upstream packages declare different OpenCV distribution
names; the helper installs contrib last and `doctor` checks `cv2.saliency`.
Do not upgrade only one of these OpenCV packages in the engine environment.

Run an existing, locally installed Ollama **vision** model. Set its exact tag
in YAML; the example tag is not a requirement to download a 27B model. The
helper never downloads an Ollama model. Preflight checks the installed tag,
image capability and digest. An unreachable or text-only model is an error.
ONNX CLIP weights are fetched once if absent and verified by the existing
pinned/checksummed model backend; inference runs on the local machine.

## Configuration and stages

Use `examples/vlog-day.yaml` as a reference. Relative paths resolve beside the
YAML file, so update them if you save a copy elsewhere. Set `python` to the
engine environment's Python executable, and supply local paths for the catalog,
photos, output, engines, FFmpeg and ffprobe. Music can remain null while analyzing.
Keep personal YAML/media/reports outside Git or in ignored local directories.

```bash
clipshow-vlog vlog.local.yaml --stage doctor
clipshow-vlog vlog.local.yaml --stage analyze
clipshow-vlog vlog.local.yaml --stage plan
clipshow-vlog vlog.local.yaml --stage render
```

Equivalents:

```bash
python -m clipshow.vlog vlog.local.yaml --stage all
clipshow --vlog-config vlog.local.yaml --vlog-stage analyze
```

`all` performs preflight, analysis, planning and rendering. Every stage checks
all four dependencies. `--limit-inputs 3` or explicit `clip_ids` is an optional,
recorded **video** smoke-test subset, not a claim the full day was analyzed.
Photo scanning is controlled separately by `photo_dir`; `vision_limit` limits
content inspections, prioritizing protected ranges and then one candidate per
source. Uninspected candidates never count as approved machine results.

Catalog columns: `Day`, `LogicalCapture`, `ClipID`, `DateBasis`, `OriginalPath`,
`DurationSec`; optional `ProxyPath`. `Day` is an exact whitelist; explicit
catalog date evidence and DJI filename dates must agree. Same-day photos use
capture EXIF (including full capture time) or the explicit DJI timestamp
filename fallback, never copy/modification date. Exact duplicate photo bytes
are excluded. Photos retain the complete composition on a blurred background
with slow motion; video defaults to preserved composition too. `video_layout:
crop` is an explicit opt-in and still needs human framing review.

`target_seconds: 600` means **up to** ten minutes. Rejected footage is not
repeated or padded to hit a number. Too few acceptable candidates produces a
shorter review timeline, or stops if none exist. Thresholds and prompts are
heuristics to calibrate against actual material, not aesthetic probabilities.

For complete dialogue, explicitly review/list source bounds:

```yaml
keep_dialogue:
  - {clip_id: BANFF-D02-001, start: 10.0, end: 30.3}
```

Bounds must include the outgoing dissolve handle **after the last word**.
These manual ranges bypass coarse Katna/CLIP rejection but still receive local
vision inspection. The compiler does not yet transcribe, detect complete
sentences, auto-select dialogue, or invent subtitles. Dialogue uses conservative
wind/noise filtering, compression and BGM ducking; heavily clipped wind cannot
be perfectly repaired. Other source sound is muted rather than letting wind
dominate scenery shots.

## Outputs, privacy and failure behavior

All footage stays local. Content inference accepts only a loopback Ollama URL,
never a remote/cloud URL. No video, photo, BGM or model weight is committed by
this workflow. Originals and previous final exports are never overwritten.
Proxy/keyframe/scratch files are confined to the configured output directory;
new final MP4 names include a unique timestamp.

Outputs include:

- `engine_audit.json`: engine/version, success/failure/cache status and evidence
  paths; per-engine request, response and log files remain local.
- `analysis.json`: same-day source identities, capture evidence, inspected
  candidates and raw visual summaries; `coarse_engine_reports.json` retains
  Katna/CLIP results, including rejected candidates.
- `timeline.json`: actual beats/BPM, upstream selection result, source windows,
  phrase-length cuts and music/analysis identities.
- `Vlog_<day>_<timestamp>.mp4` and matching `.qc.json`: full video/audio decode,
  portrait dimensions, stream durations, decoded audio coverage and five-second
  RMS checks for unexpected silent sections (closing fade excepted).

Source/config/runtime/adapter fingerprints invalidate stale selection caches.
Planning and rendering verify source/catalog identities, reviewed candidate
membership and unchanged BGM. Missing engines, bad model JSON, decode errors,
unreviewed windows and audio ending early stop the pipeline. One transient
local HTTP failure can be retried and is recorded, not counted as successful
content inspection. Valid fenced JSON is accepted: VideoHighlighter's default
opening-code-fence stop rule is disabled only inside its isolated worker.

Frames are sampled, not every frame understood. Brief occlusions, camera roll,
duplicates across different files, pacing and daily storytelling still need
full-shot viewing. Machine gates are **review candidates**, not final artistic
approval or a guarantee of ten minutes of strong material.

## Verification

```bash
python -m pip check
python -m pytest -q
```

The synthetic FFmpeg regression needs FFmpeg/ffprobe on PATH, or environment
variables `CLIPSHOW_TEST_FFMPEG` and `CLIPSHOW_TEST_FFPROBE`. It exercises nine
shots/two photos, preserved source audio, a chunk boundary, short-music loops
and complete decode/late-audio QC. It does not substitute for real engine or
artistic footage validation. Real-engine smoke-test records remain local.

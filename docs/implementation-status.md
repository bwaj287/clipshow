# Implementation and validation status

Upstream evaluated: borgel/clipshow commit
`8f986d388da9de69b1db24b218784c14b6b1f1dd` (MIT).

Changes in this fork:

- Semantic sampling interpolates measured frames onto its timeline instead of
  inserting zero-valued holes. Interrupted/unobserved tails remain zero.
- Semantic output is an absolute prompt-contrast heuristic rather than a
  per-video maximum. A uniformly irrelevant video is no longer forced to 1.
- Dedicated local travel selector evaluates entire windows, technical vetoes,
  a low score quantile, and non-overlapping candidate ranges. It does not use
  the general-purpose pipeline's per-video peak normalization.
- Same-day photo candidates use capture EXIF, or an explicitly identified DJI
  filename fallback. File copy/modification dates never establish trip dates.
- The original model bucket returned HTTP 404 during a real installation test.
  A replacement local backend now uses pinned Xenova CLIP ONNX exports with
  SHA256 checks against the Hugging Face LFS metadata. Embeddings are explicitly
  normalized before cosine comparison. Real text/image inference passed locally.
- `python -m clipshow.catalog_review` filters the archive catalog to one day,
  analyzes low-resolution proxies, and exports full-day coarse contact sheets.
  These are review candidates, not approved shots or finished videos.
- Candidate review/export is separate from the existing five-minute 9:16 vlog
  renderer. This fork does not yet implement that renderer or audio mixing.

Unit tests validate mechanics, NOT artistic highlight quality. Real camera
roll, fingers appearing briefly, redundancy across clips and the final portrait
crop still need full-shot inspection. Local CLIP model download/inference is a
separate integration check; real local CLIP text/image inference has completed.

`bd` issue-tracking executable was not available in the Windows environment.
This status file records outstanding work without modifying upstream issues.

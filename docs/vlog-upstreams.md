# Vlog upstream boundaries and notices

This fork's source is MIT-licensed. Each separately installed dependency retains
its own license. No third-party implementation, model weight, footage or music
is vendored by the Vlog integration.

| Project | Validated version/revision | Upstream license | Boundary |
| --- | --- | --- | --- |
| [Katna](https://github.com/keplerlab/Katna) | PyPI `0.9.2` | MIT (`LICENSE.txt` in installed distribution) | Imported by the isolated keyframe worker |
| [AutoCut](https://github.com/PapaPandroni/AutoCut) | `d8adc47a527533ab39874214a7c3bb69d3b3ddfa` | MIT | Independent pinned checkout, imported by the isolated beat/selection worker |
| [VideoHighlighter](https://github.com/Aseiel/VideoHighlighter) | tag `0.12.0`, `0e6217a70a8d967b5da56aff664975c976a16a5c` | AGPL-3.0 | Independent pinned checkout, `LLMModule` invoked by an isolated worker; not imported by the classic GUI |
| [ClipShow](https://github.com/borgel/clipshow) | fork based on `8f986d388da9de69b1db24b218784c14b6b1f1dd` | MIT | Existing local ONNX CLIP scorer, enhanced travel selector and new compiler |

External checkout licenses stay intact. Do not relabel VideoHighlighter's code
as MIT. A process boundary is an architectural choice, **not** a conclusion
that AGPL obligations disappear; distribution or network deployment of a
combined work needs its applicable license obligations considered. This local
workflow does not publish such a combined bundle or service.

Local model and BGM licenses are separate from source-code licenses. Installation
does not grant rights to publish a model or copyrighted song. No KGM decryption
or DRM bypass is part of this repository.

"""Prepare pinned external engines and an isolated Windows/Linux Python env.

Does not download an Ollama model, upload media, touch source files, or alter
the original GUI environment. Use an already installed local vision model.
"""

from __future__ import annotations

import argparse
import subprocess
import venv
from pathlib import Path

ENGINES = {
    "AutoCut": (
        "https://github.com/PapaPandroni/AutoCut.git",
        "d8adc47a527533ab39874214a7c3bb69d3b3ddfa",
    ),
    "VideoHighlighter": (
        "https://github.com/Aseiel/VideoHighlighter.git",
        "0e6217a70a8d967b5da56aff664975c976a16a5c",
    ),
}


def run(*command):
    subprocess.run([str(c) for c in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = (args.tools_dir or root / ".vlog-tools").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in ENGINES.items():
        target = directory / name
        if target.exists():
            actual = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != revision:
                raise RuntimeError(f"Existing checkout {target} is not pinned; not resetting it")
        else:
            run("git", "init", target)
            run("git", "-C", target, "remote", "add", "origin", url)
            run("git", "-C", target, "fetch", "--depth", "1", "origin", revision)
            run("git", "-C", target, "switch", "--detach", "FETCH_HEAD")
    env = directory / "env"
    if not env.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    python = env / ("Scripts/python.exe" if (env / "Scripts").exists() else "bin/python")
    run(python, "-m", "pip", "install", "-r", root / "requirements-vlog.txt")
    # The existing application declares GUI packages as base dependencies.
    # Install their metadata-coherent environment, without starting any GUI.
    run(python, "-m", "pip", "install", "-c", root / "requirements-vlog.txt", "-e", root)
    run(
        python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "opencv-contrib-python==4.11.0.86",
    )
    run(python, "-m", "pip", "check")
    print(
        f"Python: {python}\n"
        "Use examples/vlog-day.yaml as a template; set local paths and python."
    )


if __name__ == "__main__":
    main()

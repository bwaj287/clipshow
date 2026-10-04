"""Entry point for ClipShow - handles CLI arg parsing."""

import argparse
import sys

from clipshow import __version__


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clipshow",
        description="Automatic highlight reel generator from video clips",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Video files to process",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run in automatic mode (no full UI)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Suppress all GUI (for scripting/batch use, implies --auto)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: highlight_reel.mp4)",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto = CPU count)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to a YAML pipeline configuration file",
    )
    parser.add_argument(
        "--vlog-config",
        type=str,
        default=None,
        help="Run the integrated local Vlog compiler with a YAML configuration",
    )
    parser.add_argument(
        "--vlog-stage",
        choices=["doctor", "analyze", "plan", "render", "all"],
        default="all",
        help="Integrated Vlog stage (default: all four engines plus local preview)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args(argv)
    if args.headless:
        args.auto = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.vlog_config:
        from clipshow.vlog import main as vlog_main

        return vlog_main([args.vlog_config, "--stage", args.vlog_stage])

    if args.auto:
        from clipshow.app import run_auto_mode

        # Load YAML config if provided
        config = None
        if args.config:
            from clipshow.yaml_config import load_pipeline_config

            config = load_pipeline_config(args.config)

        # Precedence: CLI positional args > YAML inputs
        files = args.files if args.files else (config.inputs if config else [])

        # Precedence: CLI --output > YAML output.path > default
        output = args.output or (config.output_path if config else None) or "highlight_reel.mp4"

        # Precedence: CLI --workers > YAML workers > default
        workers = args.workers if args.workers is not None else (
            config.workers if config and config.workers is not None else 0
        )

        return run_auto_mode(files, output, headless=args.headless, workers=workers, config=config)
    else:
        from clipshow.app import run_gui

        return run_gui(args.files)


if __name__ == "__main__":
    sys.exit(main())

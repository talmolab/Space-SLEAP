#!/usr/bin/env python
"""
Run all visualization scripts.

Generates all weighted phase visualizations from a given SLP file.
Outputs are saved to a subfolder named after the input clip.

Usage:
    python run_all.py [slp_path] [--scale SCALE]

Examples:
    python run_all.py                           # Default clip, scale=1.0
    python run_all.py /path/to/clip.slp         # Custom clip, scale=1.0
    python run_all.py --scale 1.5               # Default clip, scale=1.5
    python run_all.py /path/to/clip.slp --scale 1.5
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("render_polar.py", "polar_phase.mp4"),
    ("render_xz.py", "xz_overhead.mp4"),
    ("render_timeseries.py", "x_vs_time.mp4"),
    ("render_phase_vs_x.py", "phase_vs_x.mp4"),
    ("render_sinphase_vs_x.py", "sinphase_vs_x.mp4"),
    ("render_gforce.py", "gforce.mp4"),
    ("render_xz_gforce.py", "xz_gforce.mp4"),
]


def main():
    script_dir = Path(__file__).parent

    # Parse command line arguments
    slp_path = None
    scale = 1.0

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--scale" and i + 1 < len(args):
            scale = float(args[i + 1])
            i += 2
        elif not args[i].startswith("--"):
            slp_path = Path(args[i])
            i += 1
        else:
            i += 1

    # Default input file
    if slp_path is None:
        slp_path = script_dir.parent.parent / "clip0.smoothed.slp"

    if not slp_path.exists():
        print(f"ERROR: Input file not found: {slp_path}")
        sys.exit(1)

    # Create output subfolder based on input filename
    clip_name = slp_path.stem.replace(".smoothed", "")  # e.g., "clip0"
    output_dir = script_dir / f"outputs.{clip_name}"
    output_dir.mkdir(exist_ok=True)

    print(f"Input: {slp_path}")
    print(f"Output directory: {output_dir}")
    print(f"Scale: {scale}")
    print()

    failed = []

    for script, output_name in SCRIPTS:
        script_path = script_dir / script
        output_path = output_dir / output_name
        print(f"{'='*60}")
        print(f"Running: {script} -> {output_name}")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, str(script_path), str(slp_path), str(output_path), str(scale)],
            cwd=str(script_dir),
        )

        if result.returncode != 0:
            print(f"FAILED: {script}")
            failed.append(script)
        else:
            print(f"SUCCESS: {script}")
        print()

    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total scripts: {len(SCRIPTS)}")
    print(f"Successful: {len(SCRIPTS) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed scripts:")
        for s in failed:
            print(f"  - {s}")
        sys.exit(1)

    # List output files
    print(f"\nOutput files in {output_dir.name}/:")
    for f in sorted(output_dir.glob("*.mp4")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

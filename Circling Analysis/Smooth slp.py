#!/usr/bin/env python
"""
Preprocess a raw SLEAP .slp file into a smoothed version suitable for phase analysis.

Three steps applied to each keypoint coordinate (X and Y independently):
  1. Median filter  — removes single-frame spike outliers
  2. Linear interpolation — fills NaN gaps left by missed detections
  3. Gaussian smoothing — removes remaining high-frequency jitter

Output is saved as <stem>.smoothed.slp alongside the input file (or to a
specified path), with all frames guaranteed to have valid keypoint coordinates.

Usage:
    python smooth_slp.py <input.slp>
    python smooth_slp.py <input.slp> <output.slp>
    python smooth_slp.py <input.slp> --median-window 3 --sigma 1.5
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter, gaussian_filter1d
import sleap_io as sio


def smooth_slp(
    input_path: str,
    output_path: str | None = None,
    median_window: int = 3,
    sigma: float = 1.5,
) -> str:
    """
    Load a SLEAP .slp file, smooth all keypoint tracks, and save.

    Args:
        input_path: Path to raw .slp file.
        output_path: Where to save the result. Defaults to <stem>.smoothed.slp
                     in the same directory as the input.
        median_window: Window size for the median filter (frames). Must be odd.
                       Larger values remove longer spike sequences.
        sigma: Gaussian smoothing sigma (frames). Larger = smoother.

    Returns:
        Path to the saved output file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix("").with_suffix("") \
            .with_name(input_path.stem.replace(".smoothed", "") + ".smoothed.slp")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Median window: {median_window}  |  Gaussian sigma: {sigma}")
    print()

    # --- Load ---
    labels = sio.load_slp(str(input_path))
    n_frames = len(labels.labeled_frames)
    skeleton = labels.skeletons[0]
    node_names = [n.name for n in skeleton.nodes]
    n_nodes = len(node_names)
    print(f"Loaded {n_frames} frames, {n_nodes} nodes: {node_names}")

    # --- Extract raw coordinates ---
    # trx shape: (n_frames, n_tracks, n_nodes, 2)
    trx = labels.numpy()
    n_tracks = trx.shape[1]
    if n_tracks > 1:
        print(f"Warning: {n_tracks} tracks found — processing all tracks.")

    # Report validity before processing
    for track_idx in range(n_tracks):
        valid = np.sum(~np.isnan(trx[:, track_idx, 0, 0]))
        pct = valid / n_frames * 100
        print(f"  Track {track_idx}: {valid}/{n_frames} valid frames ({pct:.1f}%)")

    # --- Process each track and node ---
    trx_smoothed = trx.copy()

    for track_idx in range(n_tracks):
        for node_idx in range(n_nodes):
            for coord in range(2):  # 0=x, 1=y
                series = trx[:, track_idx, node_idx, coord].copy()

                # Step 1: Median filter (ignores NaNs by forward-filling first)
                # Fill NaNs temporarily so the median filter sees no gaps
                finite_mask = np.isfinite(series)
                if finite_mask.sum() == 0:
                    continue  # Entire track is NaN — skip

                # Temporary fill for median filter (will be re-NaN'd after)
                temp = series.copy()
                nan_mask = ~finite_mask
                # Forward-fill then back-fill NaNs
                idx = np.where(finite_mask)[0]
                temp = np.interp(np.arange(len(temp)), idx, temp[idx])
                temp = median_filter(temp, size=median_window)

                # Restore original NaN positions before interpolation
                temp[nan_mask] = np.nan

                # Step 2: Linear interpolation over NaN gaps
                finite_mask2 = np.isfinite(temp)
                if finite_mask2.sum() >= 2:
                    idx2 = np.where(finite_mask2)[0]
                    temp = np.interp(np.arange(len(temp)), idx2, temp[idx2])
                elif finite_mask2.sum() == 1:
                    # Only one valid point — fill everything with it
                    temp[:] = temp[finite_mask2][0]
                # If still all NaN, leave as-is

                # Step 3: Gaussian smoothing
                if np.all(np.isfinite(temp)):
                    temp = gaussian_filter1d(temp, sigma=sigma)

                trx_smoothed[:, track_idx, node_idx, coord] = temp

    # Report validity after processing
    print()
    for track_idx in range(n_tracks):
        valid = np.sum(~np.isnan(trx_smoothed[:, track_idx, 0, 0]))
        pct = valid / n_frames * 100
        print(f"  Track {track_idx} after smoothing: {valid}/{n_frames} valid frames ({pct:.1f}%)")

    # --- Write smoothed coordinates back into labels ---
    for frame_idx, lf in enumerate(labels.labeled_frames):
        for inst in lf.instances:
            track_idx = labels.tracks.index(inst.track) if inst.track in labels.tracks else 0
            for node_idx in range(n_nodes):
                x = trx_smoothed[frame_idx, track_idx, node_idx, 0]
                y = trx_smoothed[frame_idx, track_idx, node_idx, 1]
                inst[node_idx] = sio.Point(x, y) if np.isfinite(x) and np.isfinite(y) else sio.Point(np.nan, np.nan)

    # --- Save ---
    labels.save(str(output_path))
    print(f"\nSaved: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Smooth a SLEAP .slp file for phase analysis."
    )
    parser.add_argument("input", help="Path to input .slp file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Path to output .slp file (default: <stem>.smoothed.slp)")
    parser.add_argument("--median-window", type=int, default=3,
                        help="Median filter window size in frames (default: 3)")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian smoothing sigma in frames (default: 1.5)")
    args = parser.parse_args()

    smooth_slp(
        input_path=args.input,
        output_path=args.output,
        median_window=args.median_window,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()

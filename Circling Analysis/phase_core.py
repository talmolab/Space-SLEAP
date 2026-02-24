#!/usr/bin/env python
"""
Core Phase Computation Module

Confidence-weighted phase estimation for mouse circling behavior.
Phase is estimated by blending:
- X-derived phase (arcsin): high confidence near crossings (0deg, 180deg)
- Time-interpolated phase: fallback near extrema (90deg, 270deg)

Confidence is Gaussian-weighted by proximity to crossing phases.
Also outputs estimated Z coordinate assuming elliptical orbit.
"""

import numpy as np
import pandas as pd
import sleap_io as sio
from pathlib import Path
from sleap_io.rendering import get_palette

# Constants
VIDEO_WIDTH_PX = 640
VIDEO_HEIGHT_PX = 480
CAGE_WIDTH_CM = 22.0
CAGE_DEPTH_CM = 24.0  # Z dimension
FRAME_RATE = 30.0
MIDLINE_X_PX = VIDEO_WIDTH_PX / 2

PX_TO_CM_X = CAGE_WIDTH_CM / VIDEO_WIDTH_PX
CAGE_HEIGHT_CM = 36.0  # Y dimension (vertical)
PX_TO_CM_Y = CAGE_HEIGHT_CM / VIDEO_HEIGHT_PX


def compute_centroids(labels: sio.Labels) -> np.ndarray:
    """Compute centroid for each frame as mean of all keypoints."""
    trx = labels.numpy()
    centroids = np.nanmean(trx[:, 0, :, :], axis=1)
    return centroids


def detect_crossings(centroids_px: np.ndarray, midline: float = MIDLINE_X_PX) -> tuple:
    """Detect midline crossings and return half-cycle indices and crossing frames."""
    n_frames = len(centroids_px)
    half_cycle_idx = np.zeros(n_frames, dtype=int)
    x_coords = centroids_px[:, 0]

    first_valid = 0
    for i in range(n_frames):
        if not np.isnan(x_coords[i]):
            first_valid = i
            break

    current_half_cycle = 0
    last_side = "left" if x_coords[first_valid] < midline else "right"
    crossing_frames = []

    for i in range(n_frames):
        x = x_coords[i]
        if np.isnan(x):
            half_cycle_idx[i] = current_half_cycle
            continue

        if last_side == "left" and x >= midline:
            current_half_cycle += 1
            crossing_frames.append((i, "L_to_R"))
            last_side = "right"
        elif last_side == "right" and x <= midline:
            current_half_cycle += 1
            crossing_frames.append((i, "R_to_L"))
            last_side = "left"

        half_cycle_idx[i] = current_half_cycle

    return half_cycle_idx, crossing_frames


def gaussian_confidence(phase_deg: float, sigma_deg: float = 30.0) -> float:
    """
    Compute Gaussian-weighted confidence based on proximity to 0deg or 180deg.

    High confidence (near 1) when phase is close to 0deg or 180deg (crossings).
    Low confidence (near 0) when phase is close to 90deg or 270deg (extrema).

    Args:
        phase_deg: Phase in degrees [0, 360)
        sigma_deg: Standard deviation of Gaussian (controls falloff width)

    Returns:
        Confidence value in [0, 1]
    """
    # Distance to nearest crossing (0deg or 180deg)
    dist_to_0 = min(abs(phase_deg - 0), abs(phase_deg - 360))
    dist_to_180 = abs(phase_deg - 180)
    dist_to_crossing = min(dist_to_0, dist_to_180)

    # Gaussian falloff
    confidence = np.exp(-(dist_to_crossing ** 2) / (2 * sigma_deg ** 2))
    return confidence


def compute_weighted_phase(
    centroids_px: np.ndarray,
    half_cycle_idx: np.ndarray,
    crossing_frames: list,
    midline: float = MIDLINE_X_PX,
    sigma_deg: float = 30.0,
    z_amplitude_cm: float = 10.0,
) -> tuple:
    """
    Compute confidence-weighted phase and estimated Z.

    Blends X-derived phase (accurate near crossings) with time-interpolated
    phase (smooth near extrema) using Gaussian confidence weighting.

    Args:
        centroids_px: Centroid positions in pixels
        half_cycle_idx: Half-cycle index for each frame
        crossing_frames: List of (frame, direction) tuples
        midline: X midline in pixels
        sigma_deg: Gaussian sigma for confidence falloff
        z_amplitude_cm: Amplitude of Z oscillation (B in Z = B*cos(phase))

    Returns:
        phase_deg: Phase angle in degrees
        phase_confidence: Confidence in phase estimate
        z_cm: Estimated Z coordinate in cm
        full_cycle_idx: Full cycle index
        crossing_direction: Crossing direction for each frame
    """
    n_frames = len(centroids_px)
    phase_deg = np.zeros(n_frames, dtype=float)
    phase_confidence = np.zeros(n_frames, dtype=float)
    z_cm = np.zeros(n_frames, dtype=float)
    full_cycle_idx = np.zeros(n_frames, dtype=int)
    crossing_direction = [None] * n_frames

    x_coords = centroids_px[:, 0]
    x_centered = x_coords - midline  # X=0 at midline

    # Mark crossing directions
    for frame, direction in crossing_frames:
        crossing_direction[frame] = direction

    # Get L->R crossing frames for full cycle indexing
    l_to_r_frames = [f for f, d in crossing_frames if d == "L_to_R"]

    # Compute full cycle index
    for i in range(n_frames):
        l_to_r_count = sum(1 for f in l_to_r_frames if f <= i)
        full_cycle_idx[i] = max(0, l_to_r_count - 1)

    # Estimate X amplitude (A in X = A*sin(phase))
    valid_x = x_centered[~np.isnan(x_centered)]
    A = np.percentile(np.abs(valid_x), 95)

    # Build crossing timeline with phase values
    crossing_anchors = []
    phase_at_crossing = 0.0

    for frame, direction in crossing_frames:
        if direction == "L_to_R":
            phase_at_crossing = (len([c for c in crossing_anchors if c[2] == "L_to_R"])) * 360.0
        else:  # R_to_L
            l_to_r_count = len([c for c in crossing_anchors if c[2] == "L_to_R"])
            phase_at_crossing = (l_to_r_count - 1) * 360.0 + 180.0 if l_to_r_count > 0 else 180.0

        crossing_anchors.append((frame, phase_at_crossing, direction))

    # Process each half-cycle
    all_crossings = crossing_frames.copy()

    if len(all_crossings) == 0:
        for i in range(n_frames):
            phase_deg[i] = 180.0 * i / n_frames
            phase_confidence[i] = 0.5
        z_cm = z_amplitude_cm * np.cos(np.radians(phase_deg))
        return phase_deg % 360, phase_confidence, z_cm, full_cycle_idx, crossing_direction

    # For frames before first crossing
    first_crossing_frame, first_crossing_dir = all_crossings[0]
    if first_crossing_frame > 0:
        if first_crossing_dir == "L_to_R":
            phase_at_first = 0.0
        else:
            phase_at_first = 180.0

        for i in range(first_crossing_frame):
            t_frac = i / first_crossing_frame if first_crossing_frame > 0 else 0
            if first_crossing_dir == "L_to_R":
                phase_deg[i] = 270.0 + t_frac * 90.0
            else:
                phase_deg[i] = 90.0 + t_frac * 90.0
            phase_confidence[i] = 0.3

    # Process each pair of consecutive crossings
    for c_idx in range(len(all_crossings)):
        start_frame, start_dir = all_crossings[c_idx]

        if c_idx + 1 < len(all_crossings):
            end_frame, end_dir = all_crossings[c_idx + 1]
        else:
            end_frame = n_frames
            end_dir = None

        if start_dir == "L_to_R":
            phase_start = 0.0
            phase_end = 180.0
        else:
            phase_start = 180.0
            phase_end = 360.0

        for i in range(start_frame, end_frame):
            x = x_centered[i]

            if np.isnan(x):
                if end_frame > start_frame:
                    t_frac = (i - start_frame) / (end_frame - start_frame)
                else:
                    t_frac = 0
                phase_deg[i] = phase_start + t_frac * (phase_end - phase_start)
                phase_confidence[i] = 0.5
                continue

            if end_frame > start_frame:
                t_frac = (i - start_frame) / (end_frame - start_frame)
            else:
                t_frac = 0.5
            phase_time = phase_start + t_frac * (phase_end - phase_start)

            x_norm = np.clip(x / A, -1, 1)
            arcsin_val = np.degrees(np.arcsin(x_norm))

            if start_dir == "L_to_R":
                if t_frac <= 0.5:
                    phase_x = arcsin_val
                else:
                    phase_x = 180.0 - arcsin_val
            else:
                if t_frac <= 0.5:
                    phase_x = 180.0 - arcsin_val
                else:
                    phase_x = 360.0 + arcsin_val

            conf = gaussian_confidence(phase_time % 360, sigma_deg)
            phase_deg[i] = conf * phase_x + (1 - conf) * phase_time
            phase_confidence[i] = conf

    phase_deg = phase_deg % 360
    z_center = CAGE_DEPTH_CM / 2
    z_cm = z_center + z_amplitude_cm * np.cos(np.radians(phase_deg))

    return phase_deg, phase_confidence, z_cm, full_cycle_idx, crossing_direction


def px_to_cm_x(x_px: float) -> float:
    """Convert x pixel coordinate to cm (centered at midline)."""
    return (x_px - MIDLINE_X_PX) * PX_TO_CM_X


def px_to_cm_y(y_px: float) -> float:
    """Convert y pixel coordinate to cm (0 at top of frame)."""
    return y_px * PX_TO_CM_Y


def get_phase_color(phase_deg: float) -> tuple:
    """Get RGB color from colorwheel palette based on phase angle."""
    palette = get_palette("colorwheel", 256)
    idx = int((phase_deg / 360.0) * 255) % 256
    return palette[idx]


def load_and_compute_phase(slp_path: str, sigma_deg: float = 30.0, z_amplitude_cm: float = 10.0):
    """
    Load SLP file and compute all phase-related data.

    Returns:
        dict with all computed data and labels object
    """
    labels = sio.load_slp(slp_path)
    n_frames = len(labels.labeled_frames)

    centroids_px = compute_centroids(labels)
    centroids_cm_x = np.array([px_to_cm_x(c[0]) for c in centroids_px])
    centroids_cm_y = np.array([px_to_cm_y(c[1]) for c in centroids_px])

    half_cycle_idx, crossing_frames = detect_crossings(centroids_px)
    phase_deg, phase_confidence, z_cm, full_cycle_idx, crossing_direction = compute_weighted_phase(
        centroids_px, half_cycle_idx, crossing_frames,
        sigma_deg=sigma_deg, z_amplitude_cm=z_amplitude_cm
    )

    return {
        'labels': labels,
        'n_frames': n_frames,
        'centroids_px': centroids_px,
        'centroids_cm_x': centroids_cm_x,
        'centroids_cm_y': centroids_cm_y,
        'half_cycle_idx': half_cycle_idx,
        'crossing_frames': crossing_frames,
        'phase_deg': phase_deg,
        'phase_confidence': phase_confidence,
        'z_cm': z_cm,
        'full_cycle_idx': full_cycle_idx,
        'crossing_direction': crossing_direction,
    }


def save_phase_data(data: dict, output_path: str):
    """Save phase data to CSV."""
    n_frames = data['n_frames']
    df = pd.DataFrame({
        'frame': np.arange(n_frames),
        'time_s': np.arange(n_frames) / FRAME_RATE,
        'centroid_x_px': data['centroids_px'][:, 0],
        'centroid_y_px': data['centroids_px'][:, 1],
        'centroid_x_cm': data['centroids_cm_x'],
        'centroid_y_cm': data['centroids_cm_y'],
        'phase_deg': data['phase_deg'],
        'phase_confidence': data['phase_confidence'],
        'z_cm': data['z_cm'],
        'half_cycle_idx': data['half_cycle_idx'],
        'full_cycle_idx': data['full_cycle_idx'],
        'crossing_direction': data['crossing_direction'],
    })
    df.to_csv(output_path, index=False)
    print(f"Saved phase data: {output_path}")

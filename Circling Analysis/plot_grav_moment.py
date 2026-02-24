# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Gravitational moment (centripetal acceleration) visualization from phase data."""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.ndimage import median_filter, gaussian_filter1d
from pathlib import Path

# Set seaborn style
sns.set_context("talk")
sns.set_style("white")

# Physical constants
G_EARTH = 9.8  # m/s²
FRAME_RATE = 30.0
CAGE_HEIGHT_CM = 33.6
CAGE_DEPTH_CM = 24.0

# HSV colormap for cycles
hsv_cmap = colormaps["hsv"]


def compute_kinematics(
    centroids_cm_x: np.ndarray,
    centroids_cm_y: np.ndarray,
    z_cm: np.ndarray,
    fps: float = FRAME_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity and centripetal acceleration from 3D position data.
    a = v² / r

    Returns:
        (velocity_m_s, acceleration_m_s2)
    """
    n_frames = len(centroids_cm_x)

    # Convert cm to meters
    x_m = centroids_cm_x / 100.0
    y_m = centroids_cm_y / 100.0
    z_m = z_cm / 100.0

    # Centers for radius calculation
    y_center_m = (CAGE_HEIGHT_CM / 2) / 100.0
    z_center_m = (CAGE_DEPTH_CM / 2) / 100.0

    # Compute velocity (central difference, m/s)
    dt = 1.0 / fps
    vx = np.zeros(n_frames)
    vy = np.zeros(n_frames)
    vz = np.zeros(n_frames)

    # Central difference for interior points
    vx[1:-1] = (x_m[2:] - x_m[:-2]) / (2 * dt)
    vy[1:-1] = (y_m[2:] - y_m[:-2]) / (2 * dt)
    vz[1:-1] = (z_m[2:] - z_m[:-2]) / (2 * dt)

    # Forward/backward difference for endpoints
    vx[0] = (x_m[1] - x_m[0]) / dt
    vx[-1] = (x_m[-1] - x_m[-2]) / dt
    vy[0] = (y_m[1] - y_m[0]) / dt
    vy[-1] = (y_m[-1] - y_m[-2]) / dt
    vz[0] = (z_m[1] - z_m[0]) / dt
    vz[-1] = (z_m[-1] - z_m[-2]) / dt

    # Smooth velocity
    vx = gaussian_filter1d(median_filter(vx, size=3), sigma=1.5)
    vy = gaussian_filter1d(median_filter(vy, size=3), sigma=1.5)
    vz = gaussian_filter1d(median_filter(vz, size=3), sigma=1.5)

    # 3D velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # 3D radius from orbit center
    r = np.sqrt(x_m**2 + (y_m - y_center_m)**2 + (z_m - z_center_m)**2)
    r = np.maximum(r, 0.01)

    # Centripetal acceleration: a = v² / r
    accel = v_mag**2 / r

    return v_mag, accel


# Load data
data_dir = Path(__file__).parent.parent / "2026-01-26-weighted-phase-viz"
output_dir = Path(__file__).parent

dfs = []
for clip_dir in sorted(data_dir.glob("outputs.clip*")):
    csv_path = clip_dir / "phase_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["clip"] = clip_dir.name
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Compute kinematics for each clip
kinematics_list = []
for clip_name, clip_df in data.groupby("clip"):
    clip_df = clip_df.sort_values("frame").reset_index(drop=True)
    velocity, accel = compute_kinematics(
        clip_df["centroid_x_cm"].values,
        clip_df["centroid_y_cm"].values,
        clip_df["z_cm"].values
    )
    kinematics_list.append(pd.DataFrame({
        "clip": clip_name,
        "frame": clip_df["frame"].values,
        "time_s": clip_df["time_s"].values,
        "velocity_ms": velocity,
        "accel_ms2": accel,
        "phase_deg": clip_df["phase_deg"].values,
        "full_cycle_idx": clip_df["full_cycle_idx"].values,
    }))

data_kin = pd.concat(kinematics_list, ignore_index=True)

# Exclude first and last cycles (noisy)
for clip_name, clip_df in data_kin.groupby("clip"):
    max_cycle = clip_df["full_cycle_idx"].max()
    mask = (data_kin["clip"] == clip_name) & (
        (data_kin["full_cycle_idx"] == 0) | (data_kin["full_cycle_idx"] == max_cycle)
    )
    data_kin = data_kin[~mask]

data_kin = data_kin.reset_index(drop=True)

# --- Plot 1: Full acceleration timeseries with cycle segmentation (separate per clip, no legend) ---
for clip_name, clip_df in data_kin.groupby("clip"):
    clip_df = clip_df.sort_values("time_s").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Background shading by cycle using HSV
    unique_cycles = sorted(clip_df["full_cycle_idx"].unique())
    n_cycles = len(unique_cycles)
    for i, cycle_idx in enumerate(unique_cycles):
        cycle_mask = clip_df["full_cycle_idx"] == cycle_idx
        if cycle_mask.sum() == 0:
            continue
        cycle_times = clip_df.loc[cycle_mask, "time_s"]
        t_start, t_end = cycle_times.min(), cycle_times.max()
        color = hsv_cmap(i / max(1, n_cycles))
        ax.axvspan(t_start, t_end, alpha=0.5, color=color, linewidth=0)

    # Plot acceleration trace
    sns.lineplot(data=clip_df, x="time_s", y="accel_ms2", ax=ax, color="0.2", linewidth=2.5)

    # Reference line at 1g
    ax.axhline(G_EARTH, color="0.4", linestyle=":", linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Centripetal Acceleration (m/s²)")
    ax.set_title("Gravitational Moment Timeseries")
    ax.set_ylim(0, 15)
    sns.despine(ax=ax)

    plt.tight_layout()
    clip_num = clip_name.replace("outputs.clip", "")
    plt.savefig(output_dir / f"timeseries_accel_{clip_num}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / f"timeseries_accel_{clip_num}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved timeseries_accel_{clip_num}")

# --- Plot 1b: Velocity timeseries ---
for clip_name, clip_df in data_kin.groupby("clip"):
    clip_df = clip_df.sort_values("time_s").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Background shading by cycle using HSV
    unique_cycles = sorted(clip_df["full_cycle_idx"].unique())
    n_cycles = len(unique_cycles)
    for i, cycle_idx in enumerate(unique_cycles):
        cycle_mask = clip_df["full_cycle_idx"] == cycle_idx
        if cycle_mask.sum() == 0:
            continue
        cycle_times = clip_df.loc[cycle_mask, "time_s"]
        t_start, t_end = cycle_times.min(), cycle_times.max()
        color = hsv_cmap(i / max(1, n_cycles))
        ax.axvspan(t_start, t_end, alpha=0.5, color=color, linewidth=0)

    # Plot velocity trace
    sns.lineplot(data=clip_df, x="time_s", y="velocity_ms", ax=ax, color="0.2", linewidth=2.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity Timeseries")
    ax.set_ylim(0, None)
    sns.despine(ax=ax)

    plt.tight_layout()
    clip_num = clip_name.replace("outputs.clip", "")
    plt.savefig(output_dir / f"timeseries_velocity_{clip_num}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / f"timeseries_velocity_{clip_num}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved timeseries_velocity_{clip_num}")

# --- Plot 2: Distribution of gravitational moment (pooled, no hue) ---
fig, ax = plt.subplots(figsize=(6.5, 7.8))
sns.histplot(data=data_kin, x="accel_ms2", kde=True, ax=ax, alpha=0.6, stat="density",
             linewidth=2.5, color=sns.color_palette()[0], bins=50)

# Reference line at 1g with text annotation
ax.axvline(G_EARTH, color="0.4", linestyle=":", linewidth=2)
ax.text(G_EARTH + 0.3, ax.get_ylim()[1] * 0.95, "1g", fontsize=21, color="0.4", va="top")

ax.set_xlabel("Centripetal Acceleration (m/s²)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Gravitational Moment")
ax.set_xlim(0, 15)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(output_dir / "distribution.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig(output_dir / "distribution.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved distribution")

# --- Plot 3: Phase-averaged gravitational moment (pooled) ---
phase_bins = np.linspace(0, 360, 37)  # 10-degree bins
phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

data_kin["phase_bin"] = pd.cut(data_kin["phase_deg"], bins=phase_bins, labels=phase_centers)
data_kin["phase_bin"] = data_kin["phase_bin"].astype(float)

fig, ax = plt.subplots(figsize=(13, 7.8))
sns.lineplot(data=data_kin, x="phase_bin", y="accel_ms2", ax=ax, linewidth=3,
             errorbar="sd", color=sns.color_palette()[0])

# Reference line at 1g with text annotation
ax.axhline(G_EARTH, color="0.4", linestyle=":", linewidth=2)
ax.text(365, G_EARTH, "1g", fontsize=21, color="0.4", va="center")

ax.set_xlabel("Phase (degrees)")
ax.set_ylabel("Centripetal Acceleration (m/s²)")
ax.set_title("Phase-Averaged Gravitational Moment")
ax.set_xlim(0, 360)
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_ylim(0, None)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(output_dir / "phase_averaged.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig(output_dir / "phase_averaged.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved phase_averaged")

print(f"\nAll figures saved to {output_dir}")
print(f"Acceleration range: {data_kin['accel_ms2'].min():.2f} to {data_kin['accel_ms2'].max():.2f} m/s²")
print(f"Mean acceleration: {data_kin['accel_ms2'].mean():.2f} m/s² ({data_kin['accel_ms2'].mean()/G_EARTH:.2f}g)")
print(f"Velocity range: {data_kin['velocity_ms'].min():.2f} to {data_kin['velocity_ms'].max():.2f} m/s")
print(f"Mean velocity: {data_kin['velocity_ms'].mean():.2f} m/s")

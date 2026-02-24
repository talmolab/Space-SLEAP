# Circling Analysis Pipeline

Analysis of mouse circling behavior in NASA's AEM-X habitat cage from SLEAP pose tracking data. Estimates 3D orbit trajectory from 2D video using a confidence-weighted phase algorithm, detects cycles, and computes centripetal acceleration.

Our analyses rely on the only RR-1 sample video with a singular mouse displaying continuous circling:
- `279_06-54-24_5002_Feeder_1.mp4` - Source video
- `279_06-54-24_5002_Feeder_1.solo.predictions.slp` - SLEAP predictions file


---

## Quick Start

```bash
# Step 0: Preprocess raw tracking into a smoothed SLP file
python smooth_slp.py /path/to/clip.slp

# Run all visualizations for a clip (outputs saved to outputs.<clip_name>/)
python run_all.py /path/to/clip.smoothed.slp

# Higher resolution output (1.5x scale)
python run_all.py /path/to/clip.smoothed.slp --scale 1.5

# Run a single visualization
python render_xz.py /path/to/clip.smoothed.slp output.mp4

# Static analysis figures (gravitational moment)
python plot_grav_moment.py

# 3D animated trajectory
python 01_matplotlib_3d.py
```

---

## Pipeline Overview

The pipeline has three stages: **preprocessing** (smooth raw tracking), **phase estimation** (core algorithm), and **visualization** (rendering scripts that call it).

### Stage 0 — Preprocessing (`smooth_slp.py`)

Input: a raw SLEAP `.slp` file from the tracker (typically ~40–60% valid frames due to missed detections).

```bash
python smooth_slp.py clip.slp                          # → clip.smoothed.slp
python smooth_slp.py clip.slp --median-window 5 --sigma 2.0   # custom settings
```

Three steps applied independently to each keypoint coordinate:

1. **Median filter** (window=3 frames) — removes single-frame spike outliers without blurring edges
2. **Linear interpolation** — fills NaN gaps left by missed detections
3. **Gaussian smoothing** (σ=1.5 frames) — removes remaining high-frequency jitter

Output is saved as `<stem>.smoothed.slp` with 100% valid frames, which is what all downstream scripts require.

### Stage 1 — Phase Estimation (`phase_core.py`)

Input: a smoothed SLP file with 100% valid frames.

The algorithm estimates where the mouse is in its circular orbit (0–360°) for every frame, then derives a Z depth coordinate from that phase.

**How it works:**

1. **Detect midline crossings.** The cage midline (X = 320 px) divides the orbit into half-cycles. Each crossing (L→R or R→L) is a phase anchor.

2. **Compute two phase estimates per frame:**
   - *X-derived phase* — `arcsin(x / amplitude)`, accurate when the mouse is near the midline (X changing fast)
   - *Time-interpolated phase* — linear interpolation between crossings, accurate near the orbit extrema (X changing slowly)

3. **Blend with Gaussian confidence.** Confidence is high near 0°/180° (crossings) and low near 90°/270° (extrema), so the blend automatically picks the better estimate at each point in the orbit:

   ```
   confidence = exp(−dist_to_crossing² / 2σ²)    [σ = 30°]
   phase = confidence × phase_x + (1 − confidence) × phase_time
   ```

4. **Estimate Z from phase.** Assuming an elliptical orbit:
   ```
   Z = Z_center + Z_amplitude × cos(phase)        [Z_amplitude = 10 cm]
   ```

**Phase convention:**
| Phase | Meaning |
|-------|---------|
| 0° | L→R crossing (proximal, near camera) |
| 90° | Rightmost position |
| 180° | R→L crossing (distal, far from camera) |
| 270° | Leftmost position |

**Output** (returned as a dict, also saved as `phase_data.csv`):

| Column | Description |
|--------|-------------|
| `phase_deg` | Orbit phase, 0–360° |
| `phase_confidence` | Blend confidence, 0–1 |
| `z_cm` | Estimated depth (cm) |
| `centroid_x_cm`, `centroid_y_cm` | Measured position (cm) |
| `half_cycle_idx` | Increments at each midline crossing |
| `full_cycle_idx` | Increments at each L→R crossing |

### Stage 2 — Visualization

All render scripts share the same layout: **left panel** = video frame + pose skeleton + centroid trail (colored by phase), **right panel** = the plot described below.

| Script | Output | Right panel |
|--------|--------|-------------|
| `render_polar.py` | `polar_phase.mp4` | Phase as angular position on a circle; green/red zones show Gaussian confidence |
| `render_xz.py` | `xz_overhead.mp4` | Overhead view of estimated 3D orbit (X measured, Z from phase) |
| `render_xz_gforce.py` | `xz_gforce.mp4` | XZ overhead (top ⅔) + centripetal acceleration timeseries (bottom ⅓) |
| `render_timeseries.py` | `x_vs_time.mp4` | X position vs time, scrolling 20-second window |
| `render_gforce.py` | `gforce.mp4` | Centripetal acceleration `a = v²/r` vs time; orange 1g reference line |
| `render_phase_vs_x.py` | `phase_vs_x.mp4` | Phase (°) vs X position — shows S-curve sine relationship |
| `render_sinphase_vs_x.py` | `sinphase_vs_x.mp4` | sin(phase) vs X — should be linear if X = A·sin(θ) |

**run_all.py** runs all seven in sequence and saves outputs to `outputs.<clip_name>/`.

### Static Analysis

`plot_grav_moment.py` — reads `phase_data.csv` from completed render runs and produces:
- Centripetal acceleration timeseries with per-cycle shading
- Distribution of gravitational moment (pooled across clips)
- Phase-averaged gravitational moment (0–360°, 10° bins)

`01_matplotlib_3d.py` — reads `phase_data.csv` and renders an animated 3D trajectory (X, Y, Z) with phase coloring and slow camera rotation.

---

## Coordinate System

```
Cage dimensions:   X = 22 cm (left–right), Z = 24 cm (near–far), Y = 36 cm (vertical)
Video:             640 × 480 px, 30 fps
Midline:           X = 320 px = 0 cm

Overhead view (XZ):
        FAR (Z=24 cm)
   ┌─────────────────┐
   │    ← orbit →    │
   │  X=-11     X=+11│
   └─────────────────┘
        NEAR (Z=0)
          CAMERA
```

---

## Dependencies

```
python = ">=3.10"
sleap-io
skia-python
numpy
pandas
scipy
matplotlib
seaborn
tqdm
```

---

## Known Limitations

- **Z amplitude is fixed** at 10 cm, not measured. Actual orbit depth may differ.
- **Orbit assumed elliptical** — real mouse movement is more complex than X = A·sin(θ), Z = B·cos(θ).
- **Phase can briefly reverse** (~19 frames total per clip) near orbit extrema where X-derived phase overshoots.
- Requires a fully interpolated/smoothed SLP file — raw tracking with NaN gaps will break phase continuity.

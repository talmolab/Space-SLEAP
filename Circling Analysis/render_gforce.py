#!/usr/bin/env python
"""
Centripetal Acceleration (G-Force) Visualization

Shows the gravitational moment (centripetal acceleration) from circular motion.
Acceleration computed as a = v²/r where v is velocity and r is orbit radius.
Uses 3D trajectory (X, Y measured; Z estimated from phase).
"""

import numpy as np
import skia
from pathlib import Path
from tqdm import tqdm
import sleap_io as sio
from scipy.ndimage import median_filter, gaussian_filter1d

from phase_core import (
    load_and_compute_phase, get_phase_color,
    VIDEO_WIDTH_PX, VIDEO_HEIGHT_PX, FRAME_RATE, CAGE_DEPTH_CM, CAGE_HEIGHT_CM
)

# Physical constants
G_EARTH = 9.8  # m/s²


def compute_centripetal_acceleration(
    centroids_cm_x: np.ndarray,
    centroids_cm_y: np.ndarray,
    z_cm: np.ndarray,
    fps: float = FRAME_RATE,
    smooth_velocity: bool = True,
    median_window: int = 3,
    gaussian_sigma: float = 1.5,
) -> np.ndarray:
    """
    Compute centripetal acceleration from 3D position data.

    a = v² / r

    Where:
        v = instantaneous velocity magnitude (m/s) in 3D
        r = distance from orbit center (m) in 3D

    Args:
        centroids_cm_x: X position in cm (centered at midline)
        centroids_cm_y: Y position in cm (vertical, from top of frame)
        z_cm: Z position in cm (estimated from phase)
        fps: Frame rate
        smooth_velocity: Whether to smooth velocity components
        median_window: Median filter window size (frames)
        gaussian_sigma: Gaussian filter sigma (frames)

    Returns:
        Acceleration in m/s².
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

    # Smooth velocity if requested
    if smooth_velocity:
        vx = median_filter(vx, size=median_window)
        vy = median_filter(vy, size=median_window)
        vz = median_filter(vz, size=median_window)
        vx = gaussian_filter1d(vx, sigma=gaussian_sigma)
        vy = gaussian_filter1d(vy, sigma=gaussian_sigma)
        vz = gaussian_filter1d(vz, sigma=gaussian_sigma)

    # 3D velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # 3D radius from orbit center (X centered at 0, Y and Z at cage centers)
    r = np.sqrt(x_m**2 + (y_m - y_center_m)**2 + (z_m - z_center_m)**2)
    r = np.maximum(r, 0.01)  # Avoid division by zero

    # Centripetal acceleration: a = v² / r
    accel = v_mag**2 / r

    return accel


def render_video(
    slp_path: str,
    output_path: str,
    fps: float = 30.0,
    crf: int = 18,
    trail_frames: int = 60,
    time_window_sec: float = 10.0,
    sigma_deg: float = 30.0,
    scale: float = 1.0,
):
    print(f"Loading: {slp_path}")
    data = load_and_compute_phase(slp_path, sigma_deg=sigma_deg)

    labels = data['labels']
    video = labels.video
    n_frames = data['n_frames']
    centroids_px = data['centroids_px']
    centroids_cm_x = data['centroids_cm_x']
    centroids_cm_y = data['centroids_cm_y']
    times_sec = np.arange(n_frames) / FRAME_RATE
    phase_deg = data['phase_deg']
    z_cm = data['z_cm']
    full_cycle_idx = data['full_cycle_idx']

    # Compute centripetal acceleration (3D, smoothed velocity)
    accel = compute_centripetal_acceleration(centroids_cm_x, centroids_cm_y, z_cm)

    print(f"  Found {data['half_cycle_idx'].max() + 1} half-cycles, {full_cycle_idx.max() + 1} full cycles")
    print(f"  Acceleration range: {np.nanmin(accel):.2f} to {np.nanmax(accel):.2f} m/s²")
    print(f"  Mean acceleration: {np.nanmean(accel):.2f} m/s² ({np.nanmean(accel)/G_EARTH:.2f} g)")

    skeleton = labels.skeletons[0]
    node_names = [n.name.lower().replace(" ", "_") for n in skeleton.nodes]
    trx = labels.numpy()[:, 0, :, :]

    node_colors = [
        skia.Color(66, 133, 244, 255),
        skia.Color(52, 168, 83, 255),
        skia.Color(234, 67, 53, 255),
    ]

    # Apply scale factor for higher resolution rendering
    s = scale
    plot_width = int(VIDEO_HEIGHT_PX * s)
    video_width_scaled = int(VIDEO_WIDTH_PX * s)
    video_height_scaled = int(VIDEO_HEIGHT_PX * s)
    combined_width = video_width_scaled + plot_width
    combined_height = video_height_scaled

    plot_padding = int(50 * s)
    plot_x_start = video_width_scaled + plot_padding
    plot_x_end = combined_width - plot_padding
    plot_y_start = plot_padding
    plot_y_end = combined_height - plot_padding
    plot_inner_width = plot_x_end - plot_x_start
    plot_inner_height = plot_y_end - plot_y_start

    # Y-axis range for acceleration (m/s²)
    accel_min, accel_max = 0, max(15, np.nanmax(accel) * 1.1)
    accel_range = accel_max - accel_min

    def accel_to_plot_y(a):
        normalized = (a - accel_min) / accel_range
        return plot_y_end - normalized * plot_inner_height

    print(f"Rendering {n_frames} frames...")

    with sio.VideoWriter(output_path, fps=fps, crf=crf) as writer:
        for frame_idx in tqdm(range(n_frames)):
            current_time = frame_idx / FRAME_RATE
            current_cycle = full_cycle_idx[frame_idx]
            current_accel = accel[frame_idx]

            surface = skia.Surface(combined_width, combined_height)
            canvas = surface.getCanvas()
            canvas.clear(skia.Color(0, 0, 0, 255))

            # === LEFT PANEL: Video + Pose ===
            img = video[frame_idx]
            if img.shape[2] == 3:
                img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
            img = np.ascontiguousarray(img)
            img_skia = skia.Image.fromarray(img, colorType=skia.ColorType.kRGBA_8888_ColorType)
            canvas.drawImageRect(img_skia, skia.Rect(0, 0, video_width_scaled, video_height_scaled))

            points_2d = []
            for node_idx in range(len(node_names)):
                x, y = trx[frame_idx, node_idx, :]
                if not np.isnan(x) and not np.isnan(y):
                    points_2d.append((float(x * s), float(y * s)))
                    paint = skia.Paint(Color=node_colors[node_idx], AntiAlias=True)
                    canvas.drawCircle(x * s, y * s, 5 * s, paint)

            if len(points_2d) >= 2:
                paint_edge = skia.Paint(Color=skia.Color(255, 255, 255, 180), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                for i in range(len(points_2d) - 1):
                    canvas.drawLine(points_2d[i][0], points_2d[i][1], points_2d[i + 1][0], points_2d[i + 1][1], paint_edge)

            # Centroid trail on video
            trail_start = max(0, frame_idx - trail_frames)
            for ti in range(trail_start, frame_idx):
                cx1, cy1 = centroids_px[ti]
                cx2, cy2 = centroids_px[ti + 1]
                if not np.isnan(cx1) and not np.isnan(cx2):
                    age = frame_idx - ti
                    alpha = int(255 * (1 - age / trail_frames))
                    r, g, b = get_phase_color(phase_deg[ti])
                    paint = skia.Paint(Color=skia.Color(r, g, b, alpha), AntiAlias=True, StrokeWidth=3 * s, Style=skia.Paint.kStroke_Style)
                    canvas.drawLine(cx1 * s, cy1 * s, cx2 * s, cy2 * s, paint)

            cx, cy = centroids_px[frame_idx]
            if not np.isnan(cx):
                r, g, b = get_phase_color(phase_deg[frame_idx])
                canvas.drawCircle(cx * s, cy * s, 7 * s, skia.Paint(Color=skia.Color(r, g, b, 200), AntiAlias=True))

            # === RIGHT PANEL: Acceleration vs Time ===
            paint_bg = skia.Paint(Color=skia.Color(30, 30, 30, 255))
            canvas.drawRect(skia.Rect(video_width_scaled, 0, combined_width, combined_height), paint_bg)

            # Time window (scrolling)
            t_max = current_time
            t_min = max(0, t_max - time_window_sec)

            def time_to_plot_x(t):
                if t_max == t_min:
                    return plot_x_start
                normalized = (t - t_min) / (t_max - t_min)
                return plot_x_start + normalized * plot_inner_width

            # Grid lines
            paint_grid = skia.Paint(Color=skia.Color(50, 50, 50, 255), AntiAlias=True, StrokeWidth=1 * s, Style=skia.Paint.kStroke_Style)

            # Horizontal grid (acceleration levels)
            for a_val in [0, 5, 10, 15]:
                if accel_min <= a_val <= accel_max:
                    py = accel_to_plot_y(a_val)
                    canvas.drawLine(plot_x_start, py, plot_x_end, py, paint_grid)

            # Vertical grid (time)
            for t in range(int(t_min), int(t_max) + 2, 2):
                if t_min <= t <= t_max:
                    px = time_to_plot_x(t)
                    canvas.drawLine(px, plot_y_start, px, plot_y_end, paint_grid)

            # 1g reference line (9.8 m/s²)
            if accel_min <= G_EARTH <= accel_max:
                paint_1g = skia.Paint(Color=skia.Color(255, 200, 100, 200), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                py_1g = accel_to_plot_y(G_EARTH)
                canvas.drawLine(plot_x_start, py_1g, plot_x_end, py_1g, paint_1g)

                # Label for 1g
                font_small = skia.Font(skia.Typeface('Arial'), 10 * s)
                paint_1g_text = skia.Paint(Color=skia.Color(255, 200, 100, 255), AntiAlias=True)
                canvas.drawString("1g", plot_x_end + 5 * s, py_1g + 4 * s, font_small, paint_1g_text)

            # Draw acceleration trace
            window_start_frame = int(t_min * FRAME_RATE)
            for i in range(window_start_frame, frame_idx):
                if i + 1 >= n_frames:
                    continue
                if np.isnan(accel[i]) or np.isnan(accel[i + 1]):
                    continue

                r, g, b = get_phase_color(phase_deg[i])
                paint = skia.Paint(Color=skia.Color(r, g, b, 200), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                canvas.drawLine(
                    time_to_plot_x(times_sec[i]), accel_to_plot_y(accel[i]),
                    time_to_plot_x(times_sec[i + 1]), accel_to_plot_y(accel[i + 1]),
                    paint
                )

            # Current position marker
            if not np.isnan(current_accel):
                r, g, b = get_phase_color(phase_deg[frame_idx])
                px_curr = time_to_plot_x(current_time)
                py_curr = accel_to_plot_y(current_accel)
                canvas.drawCircle(px_curr, py_curr, 6 * s, skia.Paint(Color=skia.Color(r, g, b, 255), AntiAlias=True))
                paint_outline = skia.Paint(Color=skia.Color(255, 255, 255, 200), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2 * s)
                canvas.drawCircle(px_curr, py_curr, 6 * s, paint_outline)

            # Labels
            font = skia.Font(skia.Typeface('Arial'), 12 * s)
            paint_text = skia.Paint(Color=skia.Color(180, 180, 180, 255), AntiAlias=True)

            canvas.drawString("Centripetal Acceleration", video_width_scaled + 60 * s, 20 * s, font, paint_text)
            label_y = 38 * s
            canvas.drawString(f"Cycle {current_cycle:2d}", video_width_scaled + 60 * s, label_y, font, paint_text)
            canvas.drawString(f"Phase {phase_deg[frame_idx]:5.1f}\u00b0", video_width_scaled + 140 * s, label_y, font, paint_text)
            canvas.drawString(f"a = {current_accel:5.2f} m/s\u00b2 ({current_accel/G_EARTH:.2f}g)", video_width_scaled + 250 * s, label_y, font, paint_text)

            font_small = skia.Font(skia.Typeface('Arial'), 10 * s)

            # Y-axis labels (acceleration)
            for a_val in [0, 5, 10, 15]:
                if accel_min <= a_val <= accel_max:
                    py = accel_to_plot_y(a_val)
                    canvas.drawString(f"{a_val}", plot_x_start - 20 * s, py + 4 * s, font_small, paint_text)

            canvas.drawString("m/s\u00b2", plot_x_start - 25 * s, plot_y_start - 10 * s, font_small, paint_text)

            # X-axis labels (time)
            canvas.drawString(f"{t_min:.1f}s", plot_x_start - 5 * s, plot_y_end + 15 * s, font_small, paint_text)
            canvas.drawString(f"{t_max:.1f}s", plot_x_end - 15 * s, plot_y_end + 15 * s, font_small, paint_text)
            canvas.drawString("Time", plot_x_start + plot_inner_width / 2 - 15 * s, plot_y_end + 30 * s, font_small, paint_text)

            # Frame/time info
            canvas.drawString(f"Frame: {frame_idx}", video_width_scaled + 10 * s, combined_height - 25 * s, font_small, paint_text)
            canvas.drawString(f"Time: {current_time:.2f}s", video_width_scaled + 10 * s, combined_height - 10 * s, font_small, paint_text)

            # Divider
            canvas.drawLine(video_width_scaled, 0, video_width_scaled, combined_height,
                           skia.Paint(Color=skia.Color(100, 100, 100, 255), StrokeWidth=2 * s))

            writer.write_frame(surface.toarray()[:, :, :3])

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import sys
    script_dir = Path(__file__).parent

    if len(sys.argv) >= 3:
        slp_path = sys.argv[1]
        output_path = sys.argv[2]
        scale = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0
    else:
        slp_path = str(script_dir.parent.parent / "clip0.smoothed.slp")
        output_path = str(script_dir / "outputs.clip0" / "gforce.mp4")
        scale = 1.0

    render_video(slp_path, output_path, scale=scale)

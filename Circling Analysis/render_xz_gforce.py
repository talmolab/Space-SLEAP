#!/usr/bin/env python
"""
Combined X-Z Overhead + G-Force Visualization

Top 2/3: XZ overhead view of orbit
Bottom 1/3: Centripetal acceleration time series
"""

import numpy as np
import skia
from pathlib import Path
from tqdm import tqdm
import sleap_io as sio
from scipy.ndimage import median_filter, gaussian_filter1d

from phase_core import (
    load_and_compute_phase, get_phase_color,
    VIDEO_WIDTH_PX, VIDEO_HEIGHT_PX, FRAME_RATE, CAGE_WIDTH_CM, CAGE_DEPTH_CM, CAGE_HEIGHT_CM
)

G_EARTH = 9.8  # m/s²


def compute_centripetal_acceleration(
    centroids_cm_x: np.ndarray,
    centroids_cm_y: np.ndarray,
    z_cm: np.ndarray,
    fps: float = FRAME_RATE,
) -> np.ndarray:
    """Compute centripetal acceleration from 3D position data with smoothed positions and velocity."""
    n_frames = len(centroids_cm_x)

    # Convert cm to meters
    x_m = centroids_cm_x / 100.0
    y_m = centroids_cm_y / 100.0
    z_m = z_cm / 100.0

    # Smooth positions first (especially Z which is derived from phase)
    x_m = gaussian_filter1d(median_filter(x_m, size=3), sigma=1.5)
    y_m = gaussian_filter1d(median_filter(y_m, size=3), sigma=1.5)
    z_m = gaussian_filter1d(median_filter(z_m, size=3), sigma=1.5)

    y_center_m = (CAGE_HEIGHT_CM / 2) / 100.0
    z_center_m = (CAGE_DEPTH_CM / 2) / 100.0

    dt = 1.0 / fps
    vx = np.zeros(n_frames)
    vy = np.zeros(n_frames)
    vz = np.zeros(n_frames)

    vx[1:-1] = (x_m[2:] - x_m[:-2]) / (2 * dt)
    vy[1:-1] = (y_m[2:] - y_m[:-2]) / (2 * dt)
    vz[1:-1] = (z_m[2:] - z_m[:-2]) / (2 * dt)

    vx[0] = (x_m[1] - x_m[0]) / dt
    vx[-1] = (x_m[-1] - x_m[-2]) / dt
    vy[0] = (y_m[1] - y_m[0]) / dt
    vy[-1] = (y_m[-1] - y_m[-2]) / dt
    vz[0] = (z_m[1] - z_m[0]) / dt
    vz[-1] = (z_m[-1] - z_m[-2]) / dt

    # Smooth velocity as well
    vx = gaussian_filter1d(median_filter(vx, size=3), sigma=1.5)
    vy = gaussian_filter1d(median_filter(vy, size=3), sigma=1.5)
    vz = gaussian_filter1d(median_filter(vz, size=3), sigma=1.5)

    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    r = np.sqrt(x_m**2 + (y_m - y_center_m)**2 + (z_m - z_center_m)**2)
    r = np.maximum(r, 0.01)

    return v_mag**2 / r


def render_video(
    slp_path: str,
    output_path: str,
    fps: float = 30.0,
    crf: int = 18,
    trail_frames: int = 60,
    n_cycles_visible: int = 5,
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

    accel = compute_centripetal_acceleration(centroids_cm_x, centroids_cm_y, z_cm)

    print(f"  Found {data['half_cycle_idx'].max() + 1} half-cycles, {full_cycle_idx.max() + 1} full cycles")
    print(f"  X range: {np.nanmin(centroids_cm_x):.1f} to {np.nanmax(centroids_cm_x):.1f} cm")
    print(f"  Z range: {np.nanmin(z_cm):.1f} to {np.nanmax(z_cm):.1f} cm")
    print(f"  Accel range: {np.nanmin(accel):.2f} to {np.nanmax(accel):.2f} m/s²")

    skeleton = labels.skeletons[0]
    node_names = [n.name.lower().replace(" ", "_") for n in skeleton.nodes]
    trx = labels.numpy()[:, 0, :, :]

    node_colors = [
        skia.Color(66, 133, 244, 255),
        skia.Color(52, 168, 83, 255),
        skia.Color(234, 67, 53, 255),
    ]

    # Apply scale factor for higher resolution rendering
    s = scale  # shorthand
    plot_width = int(VIDEO_HEIGHT_PX * s)
    video_width_scaled = int(VIDEO_WIDTH_PX * s)
    video_height_scaled = int(VIDEO_HEIGHT_PX * s)
    combined_width = video_width_scaled + plot_width
    combined_height = video_height_scaled

    # Layout: top 2/3 for XZ, bottom 1/3 for g-force
    plot_padding = int(40 * s)
    divider_y = int(combined_height * 0.67)

    # XZ plot region (top)
    xz_x_start = video_width_scaled + plot_padding
    xz_x_end = combined_width - plot_padding
    xz_y_start = plot_padding
    xz_y_end = divider_y - int(20 * s)
    xz_width = xz_x_end - xz_x_start
    xz_height = xz_y_end - xz_y_start

    x_min_cm, x_max_cm = -12, 12
    x_range_cm = x_max_cm - x_min_cm
    z_min_cm, z_max_cm = 0, CAGE_DEPTH_CM
    z_range_cm = z_max_cm - z_min_cm

    def cm_to_xz_x(x_cm):
        return xz_x_start + ((x_cm - x_min_cm) / x_range_cm) * xz_width

    def cm_to_xz_y(z_val):
        return xz_y_end - ((z_val - z_min_cm) / z_range_cm) * xz_height

    # G-force plot region (bottom)
    gf_x_start = video_width_scaled + plot_padding
    gf_x_end = combined_width - plot_padding
    gf_y_start = divider_y + int(10 * s)
    gf_y_end = combined_height - int(25 * s)
    gf_width = gf_x_end - gf_x_start
    gf_height = gf_y_end - gf_y_start

    accel_min, accel_max = 0, 11  # Clipped y-axis

    def accel_to_gf_y(a):
        a_clipped = np.clip(a, accel_min, accel_max)
        return gf_y_end - ((a_clipped - accel_min) / (accel_max - accel_min)) * gf_height

    print(f"Rendering {n_frames} frames...")

    with sio.VideoWriter(output_path, fps=fps, crf=crf) as writer:
        for frame_idx in tqdm(range(n_frames)):
            current_time = frame_idx / FRAME_RATE
            current_cycle = full_cycle_idx[frame_idx]
            current_x = centroids_cm_x[frame_idx]
            current_z = z_cm[frame_idx]
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
            # Scale video to fit
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

            # === RIGHT PANEL BACKGROUND ===
            paint_bg = skia.Paint(Color=skia.Color(30, 30, 30, 255))
            canvas.drawRect(skia.Rect(video_width_scaled, 0, combined_width, combined_height), paint_bg)

            # === TOP: XZ Overhead View ===
            paint_plot_bg = skia.Paint(Color=skia.Color(20, 20, 20, 255))
            canvas.drawRect(skia.Rect(xz_x_start, xz_y_start, xz_x_end, xz_y_end), paint_plot_bg)

            # Grid
            paint_grid = skia.Paint(Color=skia.Color(50, 50, 50, 255), AntiAlias=True, StrokeWidth=1 * s, Style=skia.Paint.kStroke_Style)
            for x_cm in [-10, -5, 0, 5, 10]:
                px = cm_to_xz_x(x_cm)
                canvas.drawLine(px, xz_y_start, px, xz_y_end, paint_grid)
            for z_val in [0, 6, 12, 18, 24]:
                py = cm_to_xz_y(z_val)
                canvas.drawLine(xz_x_start, py, xz_x_end, py, paint_grid)

            # Center lines
            paint_center = skia.Paint(Color=skia.Color(100, 100, 100, 255), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
            canvas.drawLine(cm_to_xz_x(0), xz_y_start, cm_to_xz_x(0), xz_y_end, paint_center)
            canvas.drawLine(xz_x_start, cm_to_xz_y(CAGE_DEPTH_CM / 2), xz_x_end, cm_to_xz_y(CAGE_DEPTH_CM / 2), paint_center)

            # Cage boundary
            paint_cage = skia.Paint(Color=skia.Color(80, 80, 80, 255), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
            cage_left = cm_to_xz_x(-CAGE_WIDTH_CM / 2)
            cage_right = cm_to_xz_x(CAGE_WIDTH_CM / 2)
            cage_near = cm_to_xz_y(0)
            cage_far = cm_to_xz_y(CAGE_DEPTH_CM)
            canvas.drawRect(skia.Rect(cage_left, cage_far, cage_right, cage_near), paint_cage)

            # XZ trajectory
            min_visible_cycle = max(0, current_cycle - n_cycles_visible + 1)
            for cycle in range(min_visible_cycle, current_cycle + 1):
                cycle_frames = [i for i in range(n_frames) if full_cycle_idx[i] == cycle and i <= frame_idx]
                if len(cycle_frames) < 2:
                    continue
                cycle_age = current_cycle - cycle
                base_alpha = int(255 * (1 - cycle_age / n_cycles_visible))
                for i in range(len(cycle_frames) - 1):
                    f1, f2 = cycle_frames[i], cycle_frames[i + 1]
                    x1, x2 = centroids_cm_x[f1], centroids_cm_x[f2]
                    z1, z2 = z_cm[f1], z_cm[f2]
                    if np.isnan(x1) or np.isnan(x2):
                        continue
                    px1, py1 = cm_to_xz_x(x1), cm_to_xz_y(z1)
                    px2, py2 = cm_to_xz_x(x2), cm_to_xz_y(z2)
                    r, g, b = get_phase_color(phase_deg[f1])
                    paint = skia.Paint(Color=skia.Color(r, g, b, base_alpha), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                    canvas.drawLine(px1, py1, px2, py2, paint)

            # Current position marker (XZ)
            if not np.isnan(current_x):
                px_curr = cm_to_xz_x(current_x)
                py_curr = cm_to_xz_y(current_z)
                r, g, b = get_phase_color(phase_deg[frame_idx])
                canvas.drawCircle(px_curr, py_curr, 6 * s, skia.Paint(Color=skia.Color(r, g, b, 255), AntiAlias=True))
                canvas.drawCircle(px_curr, py_curr, 6 * s, skia.Paint(Color=skia.Color(255, 255, 255, 200), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2 * s))

            # === BOTTOM: G-Force Trace ===
            paint_gf_bg = skia.Paint(Color=skia.Color(20, 20, 20, 255))
            canvas.drawRect(skia.Rect(gf_x_start, gf_y_start, gf_x_end, gf_y_end), paint_gf_bg)

            t_max = current_time
            t_min = max(0, t_max - time_window_sec)

            def time_to_gf_x(t):
                if t_max == t_min:
                    return gf_x_start
                return gf_x_start + ((t - t_min) / (t_max - t_min)) * gf_width

            # Grid for g-force
            for a_val in [0, 5, 10]:
                py = accel_to_gf_y(a_val)
                canvas.drawLine(gf_x_start, py, gf_x_end, py, paint_grid)

            # 1g reference line
            if accel_min <= G_EARTH <= accel_max:
                paint_1g = skia.Paint(Color=skia.Color(255, 200, 100, 180), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                py_1g = accel_to_gf_y(G_EARTH)
                canvas.drawLine(gf_x_start, py_1g, gf_x_end, py_1g, paint_1g)

            # G-force trace
            window_start_frame = int(t_min * FRAME_RATE)
            for i in range(window_start_frame, frame_idx):
                if i + 1 >= n_frames or np.isnan(accel[i]) or np.isnan(accel[i + 1]):
                    continue
                r, g, b = get_phase_color(phase_deg[i])
                paint = skia.Paint(Color=skia.Color(r, g, b, 200), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                canvas.drawLine(
                    time_to_gf_x(times_sec[i]), accel_to_gf_y(accel[i]),
                    time_to_gf_x(times_sec[i + 1]), accel_to_gf_y(accel[i + 1]),
                    paint
                )

            # Current marker (g-force)
            if not np.isnan(current_accel):
                r, g, b = get_phase_color(phase_deg[frame_idx])
                px_gf = time_to_gf_x(current_time)
                py_gf = accel_to_gf_y(current_accel)
                canvas.drawCircle(px_gf, py_gf, 5 * s, skia.Paint(Color=skia.Color(r, g, b, 255), AntiAlias=True))

            # === LABELS ===
            font = skia.Font(skia.Typeface('Arial'), 11 * s)
            font_small = skia.Font(skia.Typeface('Arial'), 9 * s)
            paint_text = skia.Paint(Color=skia.Color(180, 180, 180, 255), AntiAlias=True)

            # Top labels
            canvas.drawString("Overhead View (X vs Z)", video_width_scaled + 10 * s, 15 * s, font, paint_text)
            canvas.drawString(f"Cycle {current_cycle:2d}  Phase {phase_deg[frame_idx]:5.1f}\u00b0  X {current_x:+5.1f}cm  Z {current_z:4.1f}cm",
                            video_width_scaled + 10 * s, 30 * s, font_small, paint_text)

            # XZ axis labels
            canvas.drawString("NEAR", xz_x_end - 25 * s, xz_y_end - 5 * s, font_small, paint_text)
            canvas.drawString("FAR", xz_x_end - 20 * s, xz_y_start + 12 * s, font_small, paint_text)

            # G-force labels
            canvas.drawString(f"Centripetal Accel: {current_accel:.2f} m/s\u00b2 ({current_accel/G_EARTH:.2f}g)",
                            video_width_scaled + 10 * s, divider_y + 5 * s, font_small, paint_text)

            # 1g label
            paint_1g_text = skia.Paint(Color=skia.Color(255, 200, 100, 255), AntiAlias=True)
            canvas.drawString("1g", gf_x_end + 3 * s, accel_to_gf_y(G_EARTH) + 3 * s, font_small, paint_1g_text)

            # Y-axis labels for g-force
            for a_val in [0, 5, 10]:
                py = accel_to_gf_y(a_val)
                canvas.drawString(f"{a_val}", gf_x_start - 15 * s, py + 3 * s, font_small, paint_text)

            # Frame info
            canvas.drawString(f"Frame {frame_idx} | {current_time:.2f}s", video_width_scaled + 10 * s, combined_height - 8 * s, font_small, paint_text)

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
        output_path = str(script_dir / "outputs.clip0" / "xz_gforce.mp4")
        scale = 1.0

    render_video(slp_path, output_path, scale=scale)

#!/usr/bin/env python
"""
X vs Z (Overhead View) Visualization

Shows the estimated orbit path from above using weighted phase Z estimation.
"""

import numpy as np
import skia
from pathlib import Path
from tqdm import tqdm
import sleap_io as sio

from phase_core import (
    load_and_compute_phase, get_phase_color,
    VIDEO_WIDTH_PX, VIDEO_HEIGHT_PX, FRAME_RATE, CAGE_WIDTH_CM, CAGE_DEPTH_CM
)


def render_video(
    slp_path: str,
    output_path: str,
    fps: float = 30.0,
    crf: int = 18,
    trail_frames: int = 60,
    n_cycles_visible: int = 5,
    sigma_deg: float = 30.0,
    z_amplitude_cm: float = 10.0,
    scale: float = 1.0,
):
    print(f"Loading: {slp_path}")
    data = load_and_compute_phase(slp_path, sigma_deg=sigma_deg, z_amplitude_cm=z_amplitude_cm)

    labels = data['labels']
    video = labels.video
    n_frames = data['n_frames']
    centroids_px = data['centroids_px']
    centroids_cm_x = data['centroids_cm_x']
    phase_deg = data['phase_deg']
    z_cm = data['z_cm']
    full_cycle_idx = data['full_cycle_idx']

    print(f"  Found {data['half_cycle_idx'].max() + 1} half-cycles, {full_cycle_idx.max() + 1} full cycles")
    print(f"  X range: {np.nanmin(centroids_cm_x):.1f} to {np.nanmax(centroids_cm_x):.1f} cm")
    print(f"  Z range: {np.nanmin(z_cm):.1f} to {np.nanmax(z_cm):.1f} cm")

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

    # XZ plot parameters - overhead view
    plot_padding = int(50 * s)
    plot_x_start = video_width_scaled + plot_padding
    plot_x_end = combined_width - plot_padding
    plot_y_start = plot_padding
    plot_y_end = combined_height - plot_padding
    plot_inner_width = plot_x_end - plot_x_start
    plot_inner_height = plot_y_end - plot_y_start

    x_min_cm, x_max_cm = -12, 12
    x_range_cm = x_max_cm - x_min_cm
    z_min_cm, z_max_cm = 0, CAGE_DEPTH_CM
    z_range_cm = z_max_cm - z_min_cm

    def cm_to_plot_x(x_cm):
        normalized = (x_cm - x_min_cm) / x_range_cm
        return plot_x_start + normalized * plot_inner_width

    def cm_to_plot_y(z_cm_val):
        normalized = (z_cm_val - z_min_cm) / z_range_cm
        return plot_y_end - normalized * plot_inner_height

    print(f"Rendering {n_frames} frames...")

    with sio.VideoWriter(output_path, fps=fps, crf=crf) as writer:
        for frame_idx in tqdm(range(n_frames)):
            current_time = frame_idx / FRAME_RATE
            current_cycle = full_cycle_idx[frame_idx]
            current_x = centroids_cm_x[frame_idx]
            current_z = z_cm[frame_idx]

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

            # === RIGHT PANEL: X vs Z (Overhead View) ===
            paint_bg = skia.Paint(Color=skia.Color(30, 30, 30, 255))
            canvas.drawRect(skia.Rect(video_width_scaled, 0, combined_width, combined_height), paint_bg)

            paint_plot_bg = skia.Paint(Color=skia.Color(20, 20, 20, 255))
            canvas.drawRect(skia.Rect(plot_x_start, plot_y_start, plot_x_end, plot_y_end), paint_plot_bg)

            # Grid lines
            paint_grid = skia.Paint(Color=skia.Color(50, 50, 50, 255), AntiAlias=True, StrokeWidth=1 * s, Style=skia.Paint.kStroke_Style)
            for x_cm in [-10, -5, 0, 5, 10]:
                px = cm_to_plot_x(x_cm)
                canvas.drawLine(px, plot_y_start, px, plot_y_end, paint_grid)
            for z_val in [0, 6, 12, 18, 24]:
                py = cm_to_plot_y(z_val)
                canvas.drawLine(plot_x_start, py, plot_x_end, py, paint_grid)

            # Center lines
            paint_center = skia.Paint(Color=skia.Color(100, 100, 100, 255), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
            canvas.drawLine(cm_to_plot_x(0), plot_y_start, cm_to_plot_x(0), plot_y_end, paint_center)
            canvas.drawLine(plot_x_start, cm_to_plot_y(CAGE_DEPTH_CM / 2), plot_x_end, cm_to_plot_y(CAGE_DEPTH_CM / 2), paint_center)

            # Cage boundary
            paint_cage = skia.Paint(Color=skia.Color(80, 80, 80, 255), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
            cage_left = cm_to_plot_x(-CAGE_WIDTH_CM / 2)
            cage_right = cm_to_plot_x(CAGE_WIDTH_CM / 2)
            cage_near = cm_to_plot_y(0)
            cage_far = cm_to_plot_y(CAGE_DEPTH_CM)
            canvas.drawRect(skia.Rect(cage_left, cage_far, cage_right, cage_near), paint_cage)

            # Draw XZ trajectory for visible cycles
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

                    px1, py1 = cm_to_plot_x(x1), cm_to_plot_y(z1)
                    px2, py2 = cm_to_plot_x(x2), cm_to_plot_y(z2)

                    r, g, b = get_phase_color(phase_deg[f1])
                    paint = skia.Paint(Color=skia.Color(r, g, b, base_alpha), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                    canvas.drawLine(px1, py1, px2, py2, paint)

            # Current position marker
            if not np.isnan(current_x):
                px_curr = cm_to_plot_x(current_x)
                py_curr = cm_to_plot_y(current_z)
                r, g, b = get_phase_color(phase_deg[frame_idx])

                paint_marker = skia.Paint(Color=skia.Color(r, g, b, 255), AntiAlias=True)
                canvas.drawCircle(px_curr, py_curr, 8 * s, paint_marker)
                paint_outline = skia.Paint(Color=skia.Color(255, 255, 255, 200), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2 * s)
                canvas.drawCircle(px_curr, py_curr, 8 * s, paint_outline)

            # Labels
            font = skia.Font(skia.Typeface('Arial'), 12 * s)
            paint_text = skia.Paint(Color=skia.Color(180, 180, 180, 255), AntiAlias=True)

            canvas.drawString("Overhead View (X vs Z)", video_width_scaled + 10 * s, 20 * s, font, paint_text)
            label_y = 38 * s
            canvas.drawString(f"Cycle {current_cycle:2d}", video_width_scaled + 10 * s, label_y, font, paint_text)
            canvas.drawString(f"Phase {phase_deg[frame_idx]:5.1f}\u00b0", video_width_scaled + 90 * s, label_y, font, paint_text)
            canvas.drawString(f"X {current_x:+6.1f} cm", video_width_scaled + 190 * s, label_y, font, paint_text)
            canvas.drawString(f"Z {current_z:5.1f} cm", video_width_scaled + 300 * s, label_y, font, paint_text)

            font_small = skia.Font(skia.Typeface('Arial'), 10 * s)

            for x_cm in [-10, 0, 10]:
                px = cm_to_plot_x(x_cm)
                canvas.drawString(f"{x_cm}", px - 8 * s, plot_y_end + 15 * s, font_small, paint_text)
            canvas.drawString("X (cm)", plot_x_start + plot_inner_width / 2 - 15 * s, plot_y_end + 30 * s, font_small, paint_text)

            for z_val in [0, 12, 24]:
                py = cm_to_plot_y(z_val)
                canvas.drawString(f"{z_val}", plot_x_start - 20 * s, py + 4 * s, font_small, paint_text)

            canvas.drawString("NEAR", plot_x_end - 30 * s, plot_y_end - 5 * s, font_small, paint_text)
            canvas.drawString("FAR", plot_x_end - 25 * s, plot_y_start + 15 * s, font_small, paint_text)

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

    # Accept command line args: slp_path output_path [scale]
    if len(sys.argv) >= 3:
        slp_path = sys.argv[1]
        output_path = sys.argv[2]
        scale = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0
    else:
        slp_path = str(script_dir.parent.parent / "clip0.smoothed.slp")
        output_path = str(script_dir / "xz_overhead.mp4")
        scale = 1.0

    render_video(slp_path, output_path, n_cycles_visible=5, scale=scale)

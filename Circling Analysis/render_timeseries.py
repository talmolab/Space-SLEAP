#!/usr/bin/env python
"""
X Position vs Time Visualization

Shows X centroid position over time with phase coloring.
"""

import numpy as np
import skia
from pathlib import Path
from tqdm import tqdm
import sleap_io as sio

from phase_core import (
    load_and_compute_phase, get_phase_color,
    VIDEO_WIDTH_PX, VIDEO_HEIGHT_PX, FRAME_RATE
)


def render_video(
    slp_path: str,
    output_path: str,
    fps: float = 30.0,
    crf: int = 18,
    trail_frames: int = 60,
    time_window_sec: float = 20.0,
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
    times_sec = np.arange(n_frames) / FRAME_RATE
    phase_deg = data['phase_deg']
    z_cm = data['z_cm']
    full_cycle_idx = data['full_cycle_idx']

    print(f"  Found {data['half_cycle_idx'].max() + 1} half-cycles, {full_cycle_idx.max() + 1} full cycles")

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

    x_min_cm, x_max_cm = -12, 12
    x_range_cm = x_max_cm - x_min_cm

    def cm_to_plot_x(x_cm):
        normalized = (x_cm - x_min_cm) / x_range_cm
        return plot_x_start + normalized * plot_inner_width

    print(f"Rendering {n_frames} frames...")

    with sio.VideoWriter(output_path, fps=fps, crf=crf) as writer:
        for frame_idx in tqdm(range(n_frames)):
            current_time = frame_idx / FRAME_RATE
            current_cycle = full_cycle_idx[frame_idx]

            surface = skia.Surface(combined_width, combined_height)
            canvas = surface.getCanvas()
            canvas.clear(skia.Color(0, 0, 0, 255))

            # Left panel
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

            # Right panel: X vs Time
            paint_bg = skia.Paint(Color=skia.Color(30, 30, 30, 255))
            canvas.drawRect(skia.Rect(video_width_scaled, 0, combined_width, combined_height), paint_bg)

            t_max = current_time
            t_min = max(0, t_max - time_window_sec)

            def time_to_plot_y(t):
                if t_max == t_min:
                    return plot_y_end
                normalized = (t - t_min) / (t_max - t_min)
                return plot_y_end - normalized * plot_inner_height

            paint_grid = skia.Paint(Color=skia.Color(50, 50, 50, 255), AntiAlias=True, StrokeWidth=1 * s, Style=skia.Paint.kStroke_Style)
            for x_cm in [-10, -5, 0, 5, 10]:
                canvas.drawLine(cm_to_plot_x(x_cm), plot_y_start, cm_to_plot_x(x_cm), plot_y_end, paint_grid)
            for t in range(int(t_min), int(t_max) + 5, 5):
                if t_min <= t <= t_max:
                    canvas.drawLine(plot_x_start, time_to_plot_y(t), plot_x_end, time_to_plot_y(t), paint_grid)

            paint_center = skia.Paint(Color=skia.Color(100, 100, 100, 255), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
            canvas.drawLine(cm_to_plot_x(0), plot_y_start, cm_to_plot_x(0), plot_y_end, paint_center)

            window_start_frame = int(t_min * FRAME_RATE)
            for i in range(window_start_frame, frame_idx):
                if i + 1 >= n_frames or np.isnan(centroids_cm_x[i]) or np.isnan(centroids_cm_x[i + 1]):
                    continue
                r, g, b = get_phase_color(phase_deg[i])
                paint = skia.Paint(Color=skia.Color(r, g, b, 200), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                canvas.drawLine(cm_to_plot_x(centroids_cm_x[i]), time_to_plot_y(times_sec[i]),
                               cm_to_plot_x(centroids_cm_x[i + 1]), time_to_plot_y(times_sec[i + 1]), paint)

            if not np.isnan(centroids_cm_x[frame_idx]):
                r, g, b = get_phase_color(phase_deg[frame_idx])
                canvas.drawCircle(cm_to_plot_x(centroids_cm_x[frame_idx]), time_to_plot_y(current_time), 6 * s,
                                 skia.Paint(Color=skia.Color(r, g, b, 255), AntiAlias=True))

            font = skia.Font(skia.Typeface('Arial'), 12 * s)
            paint_text = skia.Paint(Color=skia.Color(180, 180, 180, 255), AntiAlias=True)
            canvas.drawString("X vs Time", video_width_scaled + 10 * s, 20 * s, font, paint_text)
            canvas.drawString(f"Cycle {current_cycle} | Phase: {phase_deg[frame_idx]:.0f}\u00b0 | Z: {z_cm[frame_idx]:.1f}cm", video_width_scaled + 10 * s, 38 * s, font, paint_text)

            font_small = skia.Font(skia.Typeface('Arial'), 10 * s)
            for x_cm in [-10, 0, 10]:
                canvas.drawString(f"{x_cm}", cm_to_plot_x(x_cm) - 8 * s, plot_y_end + 15 * s, font_small, paint_text)
            canvas.drawString(f"{t_max:.1f}s", plot_x_start - 30 * s, plot_y_start + 4 * s, font_small, paint_text)
            canvas.drawString(f"{t_min:.1f}s", plot_x_start - 30 * s, plot_y_end + 4 * s, font_small, paint_text)

            canvas.drawLine(video_width_scaled, 0, video_width_scaled, combined_height, skia.Paint(Color=skia.Color(100, 100, 100, 255), StrokeWidth=2 * s))

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
        output_path = str(script_dir / "x_vs_time.mp4")
        scale = 1.0

    render_video(slp_path, output_path, scale=scale)

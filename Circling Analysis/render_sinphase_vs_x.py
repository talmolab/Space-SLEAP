#!/usr/bin/env python
"""
sin(Phase) vs X Position Visualization

Shows sin(phase) vs X position - should be linear if X ~ A*sin(phase).
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
    n_cycles_visible: int = 3,
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
    phase_deg = data['phase_deg']
    phase_confidence = data['phase_confidence']
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

    def sinphase_to_plot_y(sin_phase):
        normalized = (sin_phase + 1) / 2.0
        return plot_y_end - normalized * plot_inner_height

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

            # Right panel: X vs sin(Phase)
            paint_bg = skia.Paint(Color=skia.Color(30, 30, 30, 255))
            canvas.drawRect(skia.Rect(video_width_scaled, 0, combined_width, combined_height), paint_bg)

            paint_grid = skia.Paint(Color=skia.Color(50, 50, 50, 255), AntiAlias=True, StrokeWidth=1 * s, Style=skia.Paint.kStroke_Style)
            for x_cm in [-10, -5, 0, 5, 10]:
                canvas.drawLine(cm_to_plot_x(x_cm), plot_y_start, cm_to_plot_x(x_cm), plot_y_end, paint_grid)
            for sin_val in [-1, -0.5, 0, 0.5, 1]:
                canvas.drawLine(plot_x_start, sinphase_to_plot_y(sin_val), plot_x_end, sinphase_to_plot_y(sin_val), paint_grid)

            paint_center = skia.Paint(Color=skia.Color(100, 100, 100, 255), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
            canvas.drawLine(cm_to_plot_x(0), plot_y_start, cm_to_plot_x(0), plot_y_end, paint_center)
            canvas.drawLine(plot_x_start, sinphase_to_plot_y(0), plot_x_end, sinphase_to_plot_y(0), paint_center)

            # Reference diagonal line (perfect X = A*sin(phase) relationship)
            paint_ref = skia.Paint(Color=skia.Color(80, 120, 80, 255), AntiAlias=True, StrokeWidth=1 * s, Style=skia.Paint.kStroke_Style)
            canvas.drawLine(cm_to_plot_x(-8), sinphase_to_plot_y(-1), cm_to_plot_x(8), sinphase_to_plot_y(1), paint_ref)

            min_visible_cycle = max(0, current_cycle - n_cycles_visible + 1)
            for cycle in range(min_visible_cycle, current_cycle + 1):
                cycle_frames = [i for i in range(n_frames) if full_cycle_idx[i] == cycle and i <= frame_idx]
                if len(cycle_frames) < 2:
                    continue
                cycle_age = current_cycle - cycle
                base_alpha = int(255 * (1 - cycle_age / n_cycles_visible))

                for i in range(len(cycle_frames) - 1):
                    f1, f2 = cycle_frames[i], cycle_frames[i + 1]
                    if np.isnan(centroids_cm_x[f1]):
                        continue
                    sin1 = np.sin(np.radians(phase_deg[f1]))
                    sin2 = np.sin(np.radians(phase_deg[f2]))
                    r, g, b = get_phase_color(phase_deg[f1])
                    paint = skia.Paint(Color=skia.Color(r, g, b, base_alpha), AntiAlias=True, StrokeWidth=2 * s, Style=skia.Paint.kStroke_Style)
                    canvas.drawLine(cm_to_plot_x(centroids_cm_x[f1]), sinphase_to_plot_y(sin1),
                                   cm_to_plot_x(centroids_cm_x[f2]), sinphase_to_plot_y(sin2), paint)

            if not np.isnan(centroids_cm_x[frame_idx]):
                sin_curr = np.sin(np.radians(phase_deg[frame_idx]))
                r, g, b = get_phase_color(phase_deg[frame_idx])
                canvas.drawCircle(cm_to_plot_x(centroids_cm_x[frame_idx]), sinphase_to_plot_y(sin_curr), 6 * s,
                                 skia.Paint(Color=skia.Color(r, g, b, 255), AntiAlias=True))

            font = skia.Font(skia.Typeface('Arial'), 12 * s)
            paint_text = skia.Paint(Color=skia.Color(180, 180, 180, 255), AntiAlias=True)
            canvas.drawString("X vs sin(Phase)", video_width_scaled + 10 * s, 20 * s, font, paint_text)
            canvas.drawString(f"Cycle {current_cycle} | Phase: {phase_deg[frame_idx]:.0f}\u00b0 | Conf: {phase_confidence[frame_idx]:.2f}", video_width_scaled + 10 * s, 38 * s, font, paint_text)

            font_small = skia.Font(skia.Typeface('Arial'), 10 * s)
            for x_cm in [-10, 0, 10]:
                canvas.drawString(f"{x_cm}", cm_to_plot_x(x_cm) - 8 * s, plot_y_end + 15 * s, font_small, paint_text)
            for sin_val in [-1, 0, 1]:
                canvas.drawString(f"{sin_val:.0f}", plot_x_start - 20 * s, sinphase_to_plot_y(sin_val) + 4 * s, font_small, paint_text)

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
        output_path = str(script_dir / "sinphase_vs_x.mp4")
        scale = 1.0

    render_video(slp_path, output_path, scale=scale)

#!/usr/bin/env python
"""
Polar Phase Diagram Visualization

Shows phase as angular position around a circle with confidence zones.
"""

import numpy as np
import skia
from pathlib import Path
from tqdm import tqdm
import sleap_io as sio

from phase_core import (
    load_and_compute_phase, save_phase_data, get_phase_color, gaussian_confidence,
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
    z_amplitude_cm: float = 10.0,
    scale: float = 1.0,
):
    """Render side-by-side video with polar phase plot and confidence visualization."""

    print(f"Loading: {slp_path}")
    data = load_and_compute_phase(slp_path, sigma_deg=sigma_deg, z_amplitude_cm=z_amplitude_cm)

    labels = data['labels']
    video = labels.video
    n_frames = data['n_frames']
    centroids_px = data['centroids_px']
    phase_deg = data['phase_deg']
    phase_confidence = data['phase_confidence']
    z_cm = data['z_cm']
    full_cycle_idx = data['full_cycle_idx']

    print(f"  Frames: {n_frames}")
    print(f"  Half-cycles: {data['half_cycle_idx'].max() + 1}")
    print(f"  Full cycles: {full_cycle_idx.max() + 1}")

    # Save phase data
    output_dir = Path(output_path).parent
    save_phase_data(data, str(output_dir / "phase_data.csv"))

    # Get skeleton for pose overlay
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

    # Polar plot parameters
    polar_center_x = video_width_scaled + plot_width / 2
    polar_center_y = combined_height / 2 - 30 * s
    polar_radius = min(plot_width, combined_height) / 2 - 80 * s
    orbit_radius = polar_radius * 0.75

    def phase_to_polar(phase_degrees: float, radius: float = orbit_radius) -> tuple:
        """Convert phase angle to polar coordinates (0deg at top, clockwise)."""
        angle_rad = np.radians(phase_degrees - 90)
        x = polar_center_x + radius * np.cos(angle_rad)
        y = polar_center_y + radius * np.sin(angle_rad)
        return x, y

    print(f"Rendering {n_frames} frames...")

    with sio.VideoWriter(output_path, fps=fps, crf=crf) as writer:
        for frame_idx in tqdm(range(n_frames)):
            current_time = frame_idx / FRAME_RATE
            current_cycle = full_cycle_idx[frame_idx]
            current_phase = phase_deg[frame_idx]
            current_conf = phase_confidence[frame_idx]
            current_z = z_cm[frame_idx]

            surface = skia.Surface(combined_width, combined_height)
            canvas = surface.getCanvas()
            canvas.clear(skia.Color(0, 0, 0, 255))

            # === LEFT PANEL: Video + Pose + Centroid ===
            img = video[frame_idx]
            if img.shape[2] == 3:
                img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
            img = np.ascontiguousarray(img)
            img_skia = skia.Image.fromarray(img, colorType=skia.ColorType.kRGBA_8888_ColorType)
            canvas.drawImageRect(img_skia, skia.Rect(0, 0, video_width_scaled, video_height_scaled))

            # Draw skeleton
            points_2d = []
            for node_idx, node_name in enumerate(node_names):
                x, y = trx[frame_idx, node_idx, :]
                if not np.isnan(x) and not np.isnan(y):
                    points_2d.append((float(x * s), float(y * s)))
                    paint = skia.Paint(Color=node_colors[node_idx], AntiAlias=True)
                    canvas.drawCircle(x * s, y * s, 5 * s, paint)
                    paint_outline = skia.Paint(
                        Color=skia.Color(0, 0, 0, 255),
                        AntiAlias=True,
                        Style=skia.Paint.kStroke_Style,
                        StrokeWidth=1 * s
                    )
                    canvas.drawCircle(x * s, y * s, 5 * s, paint_outline)

            if len(points_2d) >= 2:
                paint_edge = skia.Paint(
                    Color=skia.Color(255, 255, 255, 180),
                    AntiAlias=True,
                    StrokeWidth=2 * s,
                    Style=skia.Paint.kStroke_Style
                )
                for i in range(len(points_2d) - 1):
                    canvas.drawLine(
                        points_2d[i][0], points_2d[i][1],
                        points_2d[i + 1][0], points_2d[i + 1][1],
                        paint_edge
                    )

            # Draw centroid trail
            trail_start = max(0, frame_idx - trail_frames)
            trail_points = []
            for ti in range(trail_start, frame_idx + 1):
                cx, cy = centroids_px[ti]
                if not np.isnan(cx) and not np.isnan(cy):
                    trail_points.append((ti, float(cx * s), float(cy * s), phase_deg[ti]))

            if len(trail_points) > 1:
                for i in range(len(trail_points) - 1):
                    ti1, x1, y1, ph1 = trail_points[i]
                    ti2, x2, y2, ph2 = trail_points[i + 1]
                    age = frame_idx - (ti1 + ti2) / 2
                    alpha = int(255 * (1 - age / trail_frames))
                    r, g, b = get_phase_color(ph1)
                    paint_trail = skia.Paint(
                        Color=skia.Color(r, g, b, alpha),
                        AntiAlias=True,
                        StrokeWidth=3 * s,
                        Style=skia.Paint.kStroke_Style
                    )
                    canvas.drawLine(x1, y1, x2, y2, paint_trail)

            # Draw current centroid
            cx, cy = centroids_px[frame_idx]
            if not np.isnan(cx) and not np.isnan(cy):
                r, g, b = get_phase_color(current_phase)
                paint_centroid = skia.Paint(Color=skia.Color(r, g, b, 200), AntiAlias=True)
                canvas.drawCircle(cx * s, cy * s, 7 * s, paint_centroid)
                paint_outline = skia.Paint(
                    Color=skia.Color(0, 0, 0, 180),
                    AntiAlias=True,
                    Style=skia.Paint.kStroke_Style,
                    StrokeWidth=1.5 * s
                )
                canvas.drawCircle(cx * s, cy * s, 7 * s, paint_outline)

            # === RIGHT PANEL: Polar Phase Plot + Confidence ===
            paint_bg = skia.Paint(Color=skia.Color(30, 30, 30, 255))
            canvas.drawRect(skia.Rect(video_width_scaled, 0, combined_width, combined_height), paint_bg)

            # Grid circles
            paint_grid = skia.Paint(
                Color=skia.Color(50, 50, 50, 255),
                AntiAlias=True,
                StrokeWidth=1 * s,
                Style=skia.Paint.kStroke_Style
            )
            for r_frac in [0.25, 0.5, 0.75, 1.0]:
                canvas.drawCircle(polar_center_x, polar_center_y, polar_radius * r_frac, paint_grid)

            # Orbit circle
            paint_orbit = skia.Paint(
                Color=skia.Color(80, 80, 80, 255),
                AntiAlias=True,
                StrokeWidth=2 * s,
                Style=skia.Paint.kStroke_Style
            )
            canvas.drawCircle(polar_center_x, polar_center_y, orbit_radius, paint_orbit)

            # Confidence zones (green = high confidence, red = low)
            for angle in range(0, 360, 5):
                conf = gaussian_confidence(angle, sigma_deg)
                r_col = int(255 * (1 - conf))
                g_col = int(255 * conf)

                x1, y1 = phase_to_polar(angle, orbit_radius * 0.9)
                x2, y2 = phase_to_polar(angle, orbit_radius * 1.1)

                paint_conf = skia.Paint(
                    Color=skia.Color(r_col, g_col, 50, 80),
                    AntiAlias=True,
                    StrokeWidth=8 * s,
                    Style=skia.Paint.kStroke_Style
                )
                canvas.drawLine(x1, y1, x2, y2, paint_conf)

            # Radial lines
            paint_radial = skia.Paint(
                Color=skia.Color(60, 60, 60, 255),
                AntiAlias=True,
                StrokeWidth=1 * s,
                Style=skia.Paint.kStroke_Style
            )
            for angle in [0, 90, 180, 270]:
                x_end, y_end = phase_to_polar(angle, polar_radius)
                canvas.drawLine(polar_center_x, polar_center_y, x_end, y_end, paint_radial)

            # Highlight crossing lines (high confidence regions)
            paint_crossing = skia.Paint(
                Color=skia.Color(100, 200, 100, 255),
                AntiAlias=True,
                StrokeWidth=2 * s,
                Style=skia.Paint.kStroke_Style
            )
            for angle in [0, 180]:
                x_end, y_end = phase_to_polar(angle, polar_radius)
                canvas.drawLine(polar_center_x, polar_center_y, x_end, y_end, paint_crossing)

            # Draw trajectory for visible cycles
            min_visible_cycle = max(0, current_cycle - n_cycles_visible + 1)

            for cycle in range(min_visible_cycle, current_cycle + 1):
                cycle_frames = [i for i in range(n_frames) if full_cycle_idx[i] == cycle and i <= frame_idx]
                if len(cycle_frames) < 2:
                    continue

                cycle_age = current_cycle - cycle
                base_alpha = int(255 * (1 - cycle_age / n_cycles_visible))

                for i in range(len(cycle_frames) - 1):
                    f1, f2 = cycle_frames[i], cycle_frames[i + 1]
                    ph1 = phase_deg[f1]
                    ph2 = phase_deg[f2]

                    if abs(ph2 - ph1) > 180:
                        continue

                    x1, y1 = phase_to_polar(ph1)
                    x2, y2 = phase_to_polar(ph2)

                    r, g, b = get_phase_color(ph1)
                    paint_line = skia.Paint(
                        Color=skia.Color(r, g, b, base_alpha),
                        AntiAlias=True,
                        StrokeWidth=3 * s,
                        Style=skia.Paint.kStroke_Style
                    )
                    canvas.drawLine(x1, y1, x2, y2, paint_line)

            # Current position marker
            curr_x, curr_y = phase_to_polar(current_phase)
            r, g, b = get_phase_color(current_phase)
            paint_marker = skia.Paint(Color=skia.Color(r, g, b, 255), AntiAlias=True)
            canvas.drawCircle(curr_x, curr_y, 8 * s, paint_marker)
            paint_marker_outline = skia.Paint(
                Color=skia.Color(255, 255, 255, 200),
                AntiAlias=True,
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=2 * s
            )
            canvas.drawCircle(curr_x, curr_y, 8 * s, paint_marker_outline)

            # === Confidence bar at bottom ===
            bar_y = combined_height - 60 * s
            bar_height = 15 * s
            bar_x_start = video_width_scaled + 50 * s
            bar_x_end = combined_width - 50 * s
            bar_width = bar_x_end - bar_x_start

            paint_bar_bg = skia.Paint(Color=skia.Color(50, 50, 50, 255))
            canvas.drawRect(skia.Rect(bar_x_start, bar_y, bar_x_end, bar_y + bar_height), paint_bar_bg)

            conf_width = bar_width * current_conf
            conf_r = int(255 * (1 - current_conf))
            conf_g = int(255 * current_conf)
            paint_conf_fill = skia.Paint(Color=skia.Color(conf_r, conf_g, 50, 255))
            canvas.drawRect(skia.Rect(bar_x_start, bar_y, bar_x_start + conf_width, bar_y + bar_height), paint_conf_fill)

            paint_bar_border = skia.Paint(
                Color=skia.Color(100, 100, 100, 255),
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=1 * s
            )
            canvas.drawRect(skia.Rect(bar_x_start, bar_y, bar_x_end, bar_y + bar_height), paint_bar_border)

            # Labels
            font = skia.Font(skia.Typeface('Arial'), 12 * s)
            paint_text = skia.Paint(Color=skia.Color(180, 180, 180, 255), AntiAlias=True)

            canvas.drawString("Polar Phase (Gaussian conf.)", video_width_scaled + 10 * s, 20 * s, font, paint_text)
            canvas.drawString(f"Cycle {current_cycle} | Phase: {current_phase:.0f}\u00b0", video_width_scaled + 10 * s, 38 * s, font, paint_text)
            canvas.drawString(f"Z: {current_z:.1f} cm", video_width_scaled + 10 * s, 56 * s, font, paint_text)

            font_small = skia.Font(skia.Typeface('Arial'), 10 * s)

            labels_pos = [
                (0, "0\u00b0", 0, -polar_radius - 15 * s),
                (90, "90\u00b0", polar_radius + 10 * s, 4 * s),
                (180, "180\u00b0", 0, polar_radius + 20 * s),
                (270, "270\u00b0", -polar_radius - 25 * s, 4 * s),
            ]
            for angle, label, dx, dy in labels_pos:
                lx = polar_center_x + dx
                ly = polar_center_y + dy
                canvas.drawString(label, lx - len(label) * 2.5 * s, ly, font_small, paint_text)

            canvas.drawString(f"Confidence: {current_conf:.2f}", bar_x_start, bar_y - 5 * s, font_small, paint_text)
            canvas.drawString(f"Frame: {frame_idx}", video_width_scaled + 10 * s, combined_height - 25 * s, font_small, paint_text)
            canvas.drawString(f"Time: {current_time:.2f}s", video_width_scaled + 10 * s, combined_height - 10 * s, font_small, paint_text)

            # Divider
            paint_div = skia.Paint(Color=skia.Color(100, 100, 100, 255), StrokeWidth=2 * s)
            canvas.drawLine(video_width_scaled, 0, video_width_scaled, combined_height, paint_div)

            img_array = surface.toarray()[:, :, :3]
            writer.write_frame(img_array)

    print(f"Saved video: {output_path}")


def main():
    import sys
    script_dir = Path(__file__).parent

    # Accept command line args: slp_path output_path [scale]
    if len(sys.argv) >= 3:
        slp_path = sys.argv[1]
        output_path = sys.argv[2]
        scale = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0
    else:
        slp_path = str(script_dir.parent.parent / "clip0.smoothed.slp")
        output_path = str(script_dir / "polar_phase.mp4")
        scale = 1.0

    render_video(
        slp_path=slp_path,
        output_path=output_path,
        fps=30.0,
        crf=18,
        trail_frames=60,
        n_cycles_visible=3,
        sigma_deg=30.0,
        z_amplitude_cm=10.0,
        scale=scale,
    )


if __name__ == "__main__":
    main()

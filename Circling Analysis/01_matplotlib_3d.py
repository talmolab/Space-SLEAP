# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "numpy",
#     "sleap-io",
# ]
# ///
"""
3D trajectory plot with phase coloring using matplotlib.
Real-time frame-aligned animation with slow rotation.
Dark mode styling to match scratch/2026-01-26-weighted-phase-viz.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path
from sleap_io.rendering import get_palette

# Dark mode style
plt.style.use('dark_background')
BG_COLOR = (30/255, 30/255, 30/255)  # Match 2D viz: rgb(30, 30, 30)

# Load data
data_path = Path(__file__).parent.parent / "2026-01-26-weighted-phase-viz/outputs.clip0/phase_data.csv"
df = pd.read_csv(data_path)

# Extract coordinates and phase
x = df["centroid_x_cm"].values
y = df["centroid_y_cm"].values  # Y in camera coords
z = df["z_cm"].values
phase = df["phase_deg"].values
n_frames = len(x)

# Get colorwheel palette and map phase to colors
palette = get_palette("colorwheel", 256)
palette_rgb = np.array([[c[0]/255, c[1]/255, c[2]/255] for c in palette])
color_indices = (phase / 360.0 * 255).astype(int) % 256
colors = palette_rgb[color_indices]

# Create figure with dark background
fig = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
ax = fig.add_subplot(111, projection='3d', facecolor=BG_COLOR)

# Set up axis limits (fixed throughout animation) - tighter bounds
max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2 * 1.0
mid_x = (x.max() + x.min()) / 2
mid_y = (y.max() + y.min()) / 2
mid_z = (z.max() + z.min()) / 2
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_z - max_range, mid_z + max_range)
ax.set_zlim(mid_y - max_range, mid_y + max_range)

# Dark mode axis styling
ax.set_xlabel('X (cm)', color='white')
ax.set_ylabel('Z (cm)', color='white')
ax.set_zlabel('Y (cm)', color='white')
ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
ax.grid(True, alpha=0.3, color='gray')

# Initialize empty scatter and line
# Base dot size: 10 * 1.3 = 13, current dot: 200% bigger (3x)
DOT_SIZE = 13
CURRENT_DOT_SIZE = 78  # 26 * 3
OUTLINE_SIZE = 140  # White outline behind current dot (50% wider margin)
scatter = ax.scatter([], [], [], s=DOT_SIZE, alpha=0.55)  # 40% lower alpha for trail
# Current point: white outline first, then colored dot on top
current_outline = ax.scatter([], [], [], s=OUTLINE_SIZE, c='white', alpha=0.9, zorder=10)
current_dot = ax.scatter([], [], [], s=CURRENT_DOT_SIZE, alpha=0.9, zorder=11)
line, = ax.plot([], [], [], color='white', alpha=0.3, linewidth=3)

# Rotation: 30% slower (0.7 rotations over entire clip)
rotation_per_frame = (360.0 * 0.7) / n_frames
initial_azim = 45
initial_elev = 20

output_dir = Path(__file__).parent

def init():
    scatter._offsets3d = ([], [], [])
    current_outline._offsets3d = ([], [], [])
    current_dot._offsets3d = ([], [], [])
    line.set_data_3d([], [], [])
    return scatter, current_outline, current_dot, line

def update(frame):
    # Show all points up to current frame (real-time)
    n_show = frame + 1

    # Update trail scatter (all points except current)
    if n_show > 1:
        scatter._offsets3d = (x[:n_show-1], z[:n_show-1], y[:n_show-1])
        scatter.set_color(colors[:n_show-1])
    else:
        scatter._offsets3d = ([], [], [])

    # Update current point (with white outline)
    current_outline._offsets3d = ([x[frame]], [z[frame]], [y[frame]])
    current_dot._offsets3d = ([x[frame]], [z[frame]], [y[frame]])
    current_dot.set_color([colors[frame]])

    # Update line
    line.set_data_3d(x[:n_show], z[:n_show], y[:n_show])

    # Slow rotation during drawing
    azim = initial_azim + frame * rotation_per_frame
    ax.view_init(elev=initial_elev, azim=azim)

    return scatter, current_outline, current_dot, line

# Create animation - one frame per data point
print(f"Rendering {n_frames} frames (frame-aligned at 30fps)...")
anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                     interval=33, blit=False)  # 30fps

# Save animation
output_path = output_dir / "trajectory_3d_animated.mp4"
anim.save(output_path, writer='ffmpeg', fps=30, dpi=150,
          savefig_kwargs={'facecolor': BG_COLOR})
print(f"Saved animated trajectory to {output_path}")

# Also save a static final frame
fig2 = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
ax2 = fig2.add_subplot(111, projection='3d', facecolor=BG_COLOR)
ax2.scatter(x, z, y, c=colors, s=DOT_SIZE, alpha=0.55)
ax2.plot(x, z, y, color='white', alpha=0.3, linewidth=3)
ax2.set_xlim(mid_x - max_range, mid_x + max_range)
ax2.set_ylim(mid_z - max_range, mid_z + max_range)
ax2.set_zlim(mid_y - max_range, mid_y + max_range)
ax2.set_xlabel('X (cm)', color='white')
ax2.set_ylabel('Z (cm)', color='white')
ax2.set_zlabel('Y (cm)', color='white')
ax2.tick_params(colors='white')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('gray')
ax2.yaxis.pane.set_edgecolor('gray')
ax2.zaxis.pane.set_edgecolor('gray')
ax2.grid(True, alpha=0.3, color='gray')
ax2.view_init(elev=initial_elev, azim=initial_azim + 360 * 0.7)
fig2.savefig(output_dir / "trajectory_3d_static.png", dpi=150,
             bbox_inches='tight', facecolor=BG_COLOR)
print(f"Saved static plot to {output_dir / 'trajectory_3d_static.png'}")

plt.close('all')

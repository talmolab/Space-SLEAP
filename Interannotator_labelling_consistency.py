!pip install sleap-io

!git clone https://github.com/LeoMeow123/spacecage-undistort
%cd spacecage-undistort
!git switch -c master origin/master
!pip install .
%cd ..

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import itertools
import copy
import sleap_io as sio
from scipy.optimize import linear_sum_assignment
from spacecage_undistort import transform_slp_coordinates


#Apply undistortion to annotations of the original SLEAP files of each annotator (annotations were performed on original unedited videos)
#Group annotators
gdrive_root_prefix = Path("/content/drive/MyDrive/nasa")
save_dir = gdrive_root_prefix / "NASA Project/2025-11-19 - Consistency analysis v6"
save_dir.mkdir(parents=True, exist_ok=True)

annotators = [
    "Amisihi",
    "Blake",
    "Fred",
    "Julian",
    "Shambhabi",
    "Svanik",
    "Mariela",
    "Arthur",
    "Marlu",
]

slp_files = [save_dir / "annotators" / f"{ann}.slp" for ann in annotators]
slp_files

#remove predictions in case any were accidentally made, clean frames and instances
slps = [sio.load_file(slp_file) for slp_file in slp_files]
for slp in slps:
    slp.remove_predictions()
    slp.clean(frames=True, empty_instances=True)


# Point to calibration file produced by the spacecage-undistort pipeline (https://github.com/talmolab/spacecage-undistort)
calibration_file = "/content/drive/MyDrive/nasa/NASA Project/Trimmed Videos & Labels/Downsampled Videos v3 Undistorted/initial_calibration.yml"

# Create a directory for the undistorted SLP files
undistorted_slp_dir = save_dir / "annotators_undistorted"
undistorted_slp_dir.mkdir(parents=True, exist_ok=True)

# Define the prefix mapping for video paths
old_video_prefix = "/content/drive/MyDrive/nasa/NASA Project/Trimmed Videos & Labels/Downsampled Videos v3/Flight"
new_video_prefix = "/content/drive/MyDrive/nasa/NASA Project/Trimmed Videos & Labels/Downsampled Videos Undistorted v4/Flight"

# List to store the newly created undistorted Labels objects
undistorted_slps = []

for i, slp_file in enumerate(slp_files):
    # Define the output path for the undistorted SLP file
    undistorted_slp_file = undistorted_slp_dir / f"{slp_file.stem}_undistorted.slp"

    # Load the original labels to get the video paths
    original_labels = sio.load_file(slp_file)

    # Prepare an ordered list of new filenames for this specific SLP file
    # This ensures a 1:1 correspondence with original_labels.videos, even if filenames are duplicated.
    ordered_new_filenames = []
    for video in original_labels.videos:
        original_video_path = Path(video.filename)
        try:
            relative_path = original_video_path.relative_to(old_video_prefix)
            new_video_path = Path(new_video_prefix) / relative_path
            ordered_new_filenames.append(str(new_video_path))
        except ValueError:
            # If the video path doesn't match the old prefix, keep the original path
            # This ensures that the number of new filenames matches the number of videos.
            print(f"Warning: Video path '{original_video_path}' in '{slp_file.name}' does not start with expected prefix '{old_video_prefix}'. Keeping original path.")
            ordered_new_filenames.append(str(original_video_path))

    # Transform coordinates for the current SLP file
    # The function saves the output to disk and likely returns None, based on previous execution.
    transform_slp_coordinates(
        slp_input_path=slp_file,
        slp_output_path=undistorted_slp_file,
        calibration_path=calibration_file,
        video_path_mapping=ordered_new_filenames, # Pass the explicit list of filenames
    )

    # Explicitly load the newly saved undistorted SLP file
    # Since the transform function itself doesn't return the labels, we load it after it's saved.
    try:
        loaded_transformed_labels = sio.load_file(undistorted_slp_file)
        undistorted_slps.append(loaded_transformed_labels)
    except Exception as e:
        print(f"Error loading transformed file {undistorted_slp_file.name}: {e}")

# Update the 'slps' variable to point to the undistorted labels for subsequent cells
slps = undistorted_slps

print("Undistorted SLEAP labels:")
print(slps)


#Find corresponding frames among labelers
corresponding_frame_sets = {}
for slp_ind, slp in enumerate(slps):
    for lf_ind, lf in enumerate(slp):
        key = (lf.video.filename, lf.frame_idx)
        if key not in corresponding_frame_sets:
            corresponding_frame_sets[key] = []
        corresponding_frame_sets[key].append({
            "slp_ind": slp_ind,
            "annotator": annotators[slp_ind],
            "lf_ind": lf_ind,
            "frame_idx": lf.frame_idx,
            "video_ind": slp.videos.index(lf.video),
            "video_filename": lf.video.filename,
            "instances": lf.instances,
        })

# Filter to ones that have at least 2 annotations.
corresponding_frame_sets = {k: v for k, v in corresponding_frame_sets.items() if len(v) > 1}

print("Number of corresponding frames:")
len(corresponding_frame_sets)

#Visualize corresponding frames as a dataframe
frame_set_counts = pd.DataFrame([
    {annotator: sum([frame["annotator"] == annotator for frame in frame_set]) for annotator in annotators}
    for frame_set in corresponding_frame_sets.values()
], index=list(corresponding_frame_sets.keys()))
frame_set_counts["total"] = frame_set_counts.sum(axis=1)
frame_set_counts

#def for finding corresponding instances within frame sets
def score_hypothesis(frame_set, hypothesis):
    """Return a score for a hypothesis of instance set correspondences.

    Args:
        frame_set: List of dictionaries describing the frame set.
        hypothesis: List of instance indices of the same length as `frame_set`.

    Returns:
        A score for the hypothesis.

        This is computed by finding the total distance between the instance points and
        the consensus (average) of all of the instance points.
    """
    pts = np.stack([frame["instances"][inst_ind].numpy() for frame, inst_ind in zip(frame_set, hypothesis)], axis=0)
    consensus = np.nanmean(pts, axis=0, keepdims=True)
    dxy = pts - consensus
    err = np.sqrt(np.nansum(dxy ** 2, axis=-1))
    is_nan = np.isnan(dxy).any(axis=-1)
    err[is_nan] = np.nan
    return np.mean(err[~is_nan])

#def for matching the frame sets
def match_frame_set(frame_set):
    """Match all instances within the frame set into instance sets.

    Args:
        frame_set: List of dictionaries describing the frame set.

    Returns:
        A list of instance sets.

        Each instance set is a list of dictionaries describing corresponding instances.

        Each instance set may be of different length if not all instances in the same
        frame were annotated across annotators.

        The total number of instance sets for this frame set is the maximum number of
        instances that were annotated on the frame.

        In cases where there is only one instance per frame in the set, we assume that
        they correspond to the same instance.
    """
    frame_set = copy.deepcopy(frame_set)
    frame_set = [frame for frame in frame_set if len(frame["instances"]) > 0]
    for i, frame in enumerate(frame_set):
        frame["instance_ind"] = list(range((len(frame["instances"]))))

    instance_sets = []
    instance_counts = [len(frame["instances"]) for frame in frame_set]
    need_to_match = max(instance_counts) > 1
    while need_to_match:
        all_possibilities = [list(range(len(frame["instances"]))) for frame in frame_set]
        all_hypotheses = list(itertools.product(*all_possibilities))
        all_scores = [score_hypothesis(frame_set, hypothesis) for hypothesis in all_hypotheses]
        best_hypothesis = all_hypotheses[np.argmin(all_scores)]

        instance_set = []
        for inst_ind, frame in zip(best_hypothesis, frame_set):
            matched_inst = {}
            for k in ["slp_ind", "annotator", "lf_ind", "frame_idx", "video_ind", "video_filename", "frame_set_ind"]:
                matched_inst[k] = frame[k]
            matched_inst["instance"] = frame["instances"][inst_ind]
            matched_inst["instance_ind"] = frame["instance_ind"][inst_ind]
            matched_inst["within_frame_set_ind"] = len(instance_sets)
            instance_set.append(matched_inst)
        instance_sets.append(instance_set)

        for i, inst_ind in enumerate(best_hypothesis):
            frame_set[i]["instances"].pop(inst_ind)
            frame_set[i]["instance_ind"].pop(inst_ind)

        frame_set = [frame for frame in frame_set if len(frame["instances"]) > 0]
        need_to_match = max(len(frame["instances"]) for frame in frame_set) > 1

    if len(frame_set) > 0:
        instance_set = []
        for frame in frame_set:
            inst = {}
            for k in ["slp_ind", "annotator", "lf_ind", "frame_idx", "video_ind", "video_filename", "frame_set_ind"]:
                inst[k] = frame[k]
            inst["instance"] = frame["instances"][0]
            inst["instance_ind"] = frame["instance_ind"][0]
            inst["within_frame_set_ind"] = len(instance_sets)
            instance_set.append(inst)
        instance_sets.append(instance_set)

    return instance_sets


# Match all instance sets
instance_sets = []
for frame_set_ind, frame_set in enumerate(corresponding_frame_sets.values()):
    for frame in frame_set:
        frame["frame_set_ind"] = frame_set_ind
    frame_instance_sets = match_frame_set(frame_set)
    for instance_set in frame_instance_sets:
        for inst in instance_set:
            inst["instance_set_ind"] = len(instance_sets)
        instance_sets.append(instance_set)


# Only keep instance sets with enough replicates
instance_sets = [instance_set for instance_set in instance_sets if len(instance_set) > 3]

# Compute consensus poses
consensus_poses = [np.nanmean([inst["instance"].numpy() for inst in instance_set], axis=0) for instance_set in instance_sets]

len(instance_sets), [len(instance_set) for instance_set in instance_sets]


#convert to dataframe for reporting
df_rows = []
for instance_set_idx, instance_set in enumerate(instance_sets):
    for instance_dict in instance_set:
        row = {}
        for k, v in instance_dict.items():
            if k == "instance":
                row["points"] = v.numpy().tolist()
            else:
                row[k] = v
        df_rows.append(row)

df = pd.DataFrame(df_rows)

node_names = slps[0].skeleton.node_names

expanded_points = df['points'].apply(lambda x: [coord for sublist in x for coord in sublist])

new_col_names = []
for node_name in node_names:
    new_col_names.append(f'{node_name}_x')
    new_col_names.append(f'{node_name}_y')

expanded_points_df = pd.DataFrame(expanded_points.tolist(), columns=new_col_names, index=df.index)

df = pd.concat([df, expanded_points_df], axis=1)

df = df.drop(columns=['points'])

display(df)

df.to_csv("/path/to/your/drive/interannotator_consistency_dataset.csv", index=False)


#Visualize consensus point and points for all annotators
def plot_instance_set(instance_set):
    lf = slps[instance_set[0]["slp_ind"]][instance_set[0]["lf_ind"]]

    consensus = np.nanmean([inst["instance"].numpy() for inst in instance_set], axis=0)

    plt.figure(figsize=(6, 4), dpi=200)
    plt.imshow(lf.image, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    plt.plot(consensus[:, 0], consensus[:, 1], "+-", ms=10, mew=1, lw=1, c="r")

    for inst in instance_set:
        pts = inst["instance"].numpy()
        plt.plot(pts[:, 0], pts[:, 1], ".", ms=4, mew=0.5, mec="w", alpha=0.7)

#Visualize by individualframe
plot_instance_set(instance_sets[0])


#Calculate distances to consensus per annotation
all_poses, all_consensus, all_instance_set_inds = [], [], []
for instance_set_ind, (consensus, instance_set) in enumerate(zip(consensus_poses, instance_sets)):
    for frame in instance_set:
        pose = frame["instance"].numpy()
        all_poses.append(pose)
        all_consensus.append(consensus)
        all_instance_set_inds.append(instance_set_ind)

all_poses = np.stack(all_poses, axis=0)
all_consensus = np.stack(all_consensus, axis=0)
all_instance_set_inds = np.array(all_instance_set_inds)

all_poses.shape, all_consensus.shape, all_instance_set_inds.shape

delta = np.linalg.norm(all_poses - all_consensus, axis=-1)
np.nanmean(delta, axis=0), np.nanstd(delta, axis=0)

#Find average distance to consensus
all_instance_set_inds[np.argsort(delta.mean(axis=1))[::-1]]

consensus_df = pd.DataFrame({"instance_set": all_instance_set_inds, "delta": delta.mean(axis=1)}).groupby("instance_set").mean().sort_values("delta")
consensus_df


#Visualize all labels, consensus points, and average distance to consensus for every label set
for instance_set_ind, delta_inst_set in consensus_df.iterrows():
    plot_instance_set(instance_sets[instance_set_ind])
    plt.text(5, 5, f"Instance set: {instance_set_ind} / Error: {delta_inst_set.iloc[0]:.2f} px", color="w", va="top", fontsize=10);


#Visualize all annotations into one represenetative image by translating annotation coords
delta_signed = all_poses - all_consensus

instance_set = instance_sets[15]

lf = slps[instance_set[0]["slp_ind"]][instance_set[0]["lf_ind"]]

consensus = np.nanmean([inst["instance"].numpy() for inst in instance_set], axis=0)

plt.figure(figsize=(6, 4), dpi=200)
plt.imshow(lf.image, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

plt.plot(consensus[:, 0], consensus[:, 1], "+-", ms=10, mew=1, lw=1, c="r")
plt.plot(consensus[None, :, 0] + delta_signed[:, :, 0], consensus[None, :, 1] + delta_signed[:, :, 1], ".", ms=2, mew=0.5, mec="w", alpha=0.7);


#Visualize percentile distances to consensus as radial distances over a representative image
percentiles = [75, 90, 99]
prcs = np.nanpercentile(delta, percentiles, axis=0)


instance_set = instance_sets[15]

lf = slps[instance_set[0]["slp_ind"]][instance_set[0]["lf_ind"]]

consensus = np.nanmean([inst["instance"].numpy() for inst in instance_set], axis=0)

plt.figure(figsize=(6, 4), dpi=200)
plt.imshow(lf.image, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())


pal = sns.color_palette("Set2", n_colors=len(consensus))
for xy, col in zip(consensus, pal):
    plt.plot(xy[0], xy[1], "x", ms=12, mew=2, lw=1, c=col)


cmap = sns.color_palette("viridis", n_colors=len(prcs))
for percentile, prc, col in zip(percentiles, prcs, cmap):
    first_pass = True
    for p, xy in zip(prc, consensus):
        plt.gca().add_patch(plt.Circle(
            xy,
            p,
            fill=False,
            ec=col,
            lw=2,
            alpha=0.8,
            label=f"{percentile}%" if first_pass else None,
            zorder=999,
        ))
        first_pass = False

plt.legend(loc="upper right", title="Variability", fontsize=14);


#Plot error distances per keypoint as a bar graph
node_names = slps[0].skeleton.node_names
df = pd.DataFrame(delta, columns=node_names)
df = df.melt(var_name="Keypoint", value_name="Distance")

sns.set_context("talk")

pal = sns.color_palette("Set2", n_colors=len(node_names))

plt.figure(figsize=(3, 6), dpi=120)

sns.barplot(data=df, x="Keypoint", y="Distance", hue="Keypoint", palette=pal, linewidth=1, edgecolor=".5")
sns.swarmplot(data=df, x="Keypoint", y="Distance", size=3, color=".25", alpha=0.5)
plt.ylim([0, 50])
plt.ylabel("Distance to consensus (pixels)");


#Plot error distances per keypoint as a violin plot
node_names = slps[0].skeleton.node_names
df = pd.DataFrame(delta, columns=node_names)
df = df.melt(var_name="Keypoint", value_name="Distance")

sns.set_context("talk")

pal = sns.color_palette("Set2", n_colors=len(node_names))

plt.figure(figsize=(3, 6), dpi=120)
sns.violinplot(data=df, x="Keypoint", y="Distance", hue="Keypoint", palette=pal, inner=None, linewidth=1)
sns.swarmplot(data=df, x="Keypoint", y="Distance", size=3, color=".25", alpha=0.5)
plt.ylim([0, 50])
plt.ylabel("Distance to consensus (pixels)");


#Visualize error distance distribution per keypoint as a kernel density estimate plot
node_names = slps[0].skeleton.node_names
df = pd.DataFrame(delta, columns=node_names)
df = df.melt(var_name="Keypoint", value_name="Distance")

sns.set_context("talk")

pal = sns.color_palette("Set2", n_colors=len(node_names))

g = sns.FacetGrid(df, row="Keypoint", hue="Keypoint",
                  aspect=4, height=1.5, palette=pal,
                  subplot_kws={"facecolor": (0, 0, 0, 0)})

g.map(sns.kdeplot, "Distance",
      bw_adjust=0.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5,
      clip=(0, 50))

g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

def label(x, color, label):
    ax = plt.gca()
    ax.text(1, 0.3, label, fontweight="bold", color=color,
            ha="right", va="center", transform=ax.transAxes)

g.map(label, "Keypoint")

g.figure.subplots_adjust(hspace=-.25)

g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.set(xlabel="Distance to consensus (pixels)")
g.figure.suptitle("Interannotator variability", ha="center", x=0.6);


#Visualize error distance by frame quadrant
np.nanmin(all_poses[..., 0]), np.nanmax(all_poses[..., 0]), np.nanmin(all_poses[..., 1]), np.nanmax(all_poses[..., 1])

slps[0].videos[0].shape

x_min, x_max = 0, 640
y_min, y_max = 0, 480

x_edges = np.linspace(x_min, x_max, 4)
y_edges = np.linspace(y_min, y_max, 4)

x_bin_indices = np.digitize(all_poses[..., 0], bins=x_edges) - 1  # subtract 1 to zero-index
y_bin_indices = np.digitize(all_poses[..., 1], bins=y_edges) - 1

bin_indices = np.stack((x_bin_indices, y_bin_indices), axis=-1)  # ((0, 2), (0, 2))
linear_indices = (bin_indices[..., 0] * (len(x_edges) - 1)) + bin_indices[..., 1]  # (0, 8)
linear_indices[np.isnan(all_poses).any(axis=-1)] = -1
linear_indices.shape

df = pd.DataFrame({
    "Bin": linear_indices.flatten(),
    "Row": bin_indices[..., 0].flatten(),
    "Column": bin_indices[..., 1].flatten(),
    "Distance": delta.flatten()
    })
df = df[df["Bin"] != -1]
bin_descriptions = {
    0: "Top-left",
    1: "Top-center",
    2: "Top-right",
    3: "Middle-left",
    4: "Center",
    5: "Middle-right",
    6: "Bottom-left",
    7: "Bottom-center",
    8: "Bottom-right"
}
df["Location"] = df["Bin"].map(bin_descriptions)
df

df[["Location"]].groupby("Location").value_counts()


#Visualize quadrant-based error distances as a barplot
sns.barplot(df, y="Location", x="Distance")


#Visualize quadrant-based error distances as a heatmap overlaid onto a representative frame
loc_df = df[["Row", "Column", "Distance"]].groupby(["Row", "Column"]).mean()

count_df = df[["Row", "Column", "Distance"]].groupby(["Row", "Column"]).count()

min_observations = 5 #adjust desired minmum observations per quadrant if needed

dist_map = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan)
for (r, c), d in loc_df.iterrows():
    n = count_df.loc[r, c].iloc[0]
    dist_map[r, c] = d.iloc[0] if n > min_observations else np.nan
dist_map

instance_set = instance_sets[15]

lf = slps[instance_set[0]["slp_ind"]][instance_set[0]["lf_ind"]]
img = lf.image
consensus = np.nanmean([inst["instance"].numpy() for inst in instance_set], axis=0)

plt.figure(figsize=(6, 4.5), dpi=200)
plt.imshow(img, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

plt.imshow(dist_map, aspect="auto", interpolation="nearest", extent=[0, img.shape[1], img.shape[0], 0], cmap="RdBu_r", alpha=0.5)
plt.colorbar(label="Distance to consensus (px)")

for x in x_edges[1:-1]:
    plt.axvline(x, c="w", lw=1, ls="--")
for y in y_edges[1:-1]:
    plt.axhline(y, c="w", lw=1, ls="--")

# Plot value as text in the center of each grid square
for c in range(len(x_edges) - 1):
    x = (x_edges[c] + x_edges[c + 1]) / 2
    for r in range(len(y_edges) - 1):
        y = (y_edges[r] + y_edges[r + 1]) / 2
        val = dist_map[r, c]
        if not np.isnan(val):
            plt.text(x, y, f"{val:.1f}", ha="center", va="center", color="w", fontsize=12)


#Calculate keypoint bounding parameters for reporting
node_names = slps[0].skeleton.node_names

percentiles = [50, 75, 90, 95]
prcs = np.nanpercentile(delta, percentiles, axis=0)

print("Mean:")
print(np.nanmean(delta, axis=0))
print(np.nanmean(delta))
print()
print("STD:")
print(np.nanstd(delta, axis=0))
print(np.nanstd(delta))
print()
print("Percentiles:")
pd.DataFrame(prcs, columns=node_names, index=percentiles)

def compute_instance_area(points: np.ndarray) -> np.ndarray:
    """Compute the area of the bounding box of a set of keypoints.

    Args:
        points: A numpy array of coordinates.

    Returns:
        The area of the bounding box of the points.
    """
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)

    min_pt = np.nanmin(points, axis=-2)
    max_pt = np.nanmax(points, axis=-2)

    return np.prod(max_pt - min_pt, axis=-1)


# https://cocodataset.org/#keypoints-eval:~:text=box/segment%20detection.-,1.3.%20Tuning%20OKS,-We%20tune%20the
# sigma_i ** 2 = mean(d_i ** 2 / scale ** 2)
# kappa_i = 2 * sigma_i
all_consensus_areas = compute_instance_area(all_consensus)

# "Each ground truth object also has a scale s which we define as the square root of the object segment area."
# sigmas = np.sqrt(np.nanmean((delta ** 2) / (all_consensus_areas.reshape(-1, 1) ** 2), axis=0))
sigmas = np.sqrt(np.nanmean((delta ** 2) / (all_consensus_areas.reshape(-1, 1)), axis=0))
print(sigmas)
print(2 * sigmas) #<--These values will be used for comparing model performance to human perfomance based on interannotator consistencies

#Generate results dataframe
keep_keys = ["slp_ind", "annotator", "lf_ind", "frame_idx", "video_ind", "video_filename", "instance_ind", "frame_set_ind", "within_frame_set_ind", "instance_set_ind"]
consistency_df = []
for instance_set_ind, instance_set in enumerate(instance_sets):
    consensus = np.nanmean([inst["instance"].numpy() for inst in instance_set], axis=0)

    for inst in instance_set:
        obs = {k: inst[k] for k in keep_keys}
        obs["slp_filename"] = slp_files[inst["slp_ind"]]
        slp = slps[inst["slp_ind"]]
        video_height, video_width = slp.videos[inst["video_ind"]].shape[1:3]
        obs["video_width"] = video_width
        obs["video_height"] = video_height
        pts = inst["instance"].numpy()

        dxy = pts - consensus
        err = np.sqrt(np.nansum(dxy ** 2, axis=-1))
        is_nan = np.isnan(dxy).any(axis=-1)
        err[is_nan] = np.nan
        delta_inst = np.mean(err[~is_nan])

        obs["pts"] = pts.tolist()
        obs["area"] = float(compute_instance_area(pts).squeeze())
        obs["delta"] = delta_inst
        obs["consensus"] = consensus.tolist()
        consistency_df.append(obs)

consistency_df = pd.DataFrame(consistency_df)
consistency_df

consistency_df.to_csv("/Your/Path/Here/Archival Misc/consistency.csv")


#Visualize mean distance to consensus by labeller
consistency_df.groupby("annotator")["delta"].mean().sort_values()


#Visualize distances to consensus by label bounding box area
sns.set_context("talk")
sns.scatterplot(data=consistency_df, x="area", y="delta")


#Visualize distances to consensus as one histogram
sns.set_context("talk")
sns.histplot(data=consistency_df, x="delta", bins=30)


#Merge all annotators labels and consensus points into one concise .slp file with embedded frames
unique_video_filenames = consistency_df["video_filename"].unique().tolist()
unique_videos = {x: sio.load_video(x, grayscale=False) for x in unique_video_filenames}

skel = slps[0].skeleton

lfs = []
for _, frame_set in consistency_df.groupby("frame_set_ind"):
    frame_set_df = frame_set.groupby("instance_set_ind").first()[["video_filename", "frame_idx", "consensus"]]

    video_filename = frame_set_df["video_filename"].iloc[0]
    frame_idx = frame_set_df["frame_idx"].iloc[0]
    all_frame_set_pts = frame_set_df["consensus"].tolist()

    insts = []
    for pts in all_frame_set_pts:
        insts.append(sio.Instance.from_numpy(pts, skeleton=skel))
    lfs.append(sio.LabeledFrame(video=unique_videos[video_filename], frame_idx=frame_idx, instances=insts))
labels_consensus = sio.Labels(lfs)
labels_consensus

labels_consensus.save(f"{save_dir}/labels_consensus.slp")
labels_consensus.save(f"{save_dir}/labels_consensus.pkg.slp", embed="user")

#Workflow for analyzing SLEAP model performance, and calibrated for humn error

!pip install "sleap-nn[torch] @ git+https://github.com/talmolab/sleap-nn.git" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128

!git clone https://github.com/talmolab/spacecage-undistort
%cd spacecage-undistort
!git switch -c master origin/master
!pip install .
%cd ..

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sleap_io as sio
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from scipy.optimize import linear_sum_assignment


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


def compute_oks(
    points_gt: np.ndarray,
    points_pr: np.ndarray,
    scale: float | None = None,
    stddev: float = 0.025,
    use_cocoeval: bool = True,
) -> np.ndarray:
    """Compute the object keypoints similarity between sets of points.

    Args:
        points_gt: Ground truth instances of shape (n_gt, n_nodes, n_ed),
            where n_nodes is the number of body parts/keypoint types, and n_ed
            is the number of Euclidean dimensions (typically 2 or 3). Keypoints
            that are missing/not visible should be represented as NaNs.
        points_pr: Predicted instance of shape (n_pr, n_nodes, n_ed).
        use_cocoeval: Indicates whether the OKS score is calculated like cocoeval
            method or not. True indicating the score is calculated using the
            cocoeval method (widely used and the code can be found here at
            https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L192C5-L233C20)
            and False indicating the score is calculated using the method exactly
            as given in the paper referenced in the Notes below.
        scale: Size scaling factor to use when weighing the scores, typically
            the area of the bounding box of the instance (in pixels). This
            should be of the length n_gt. If a scalar is provided, the same
            number is used for all ground truth instances. If set to None, the
            bounding box area of the ground truth instances will be calculated.
        stddev: The standard deviation associated with the spread in the
            localization accuracy of each node/keypoint type. This should be of
            the length n_nodes. "Easier" keypoint types will have lower values
            to reflect the smaller spread expected in localizing it.

    Returns:
        The object keypoints similarity between every pair of ground truth and
        predicted instance, a numpy array of of shape (n_gt, n_pr) in the range
        of [0, 1.0], with 1.0 denoting a perfect match.

    Notes:
        It's important to set the stddev appropriately when accounting for the
        difficulty of each keypoint type. For reference, the median value for
        all keypoint types in COCO is 0.072. The "easiest" keypoint is the left
        eye, with stddev of 0.025, since it is easy to precisely locate the
        eyes when labeling. The "hardest" keypoint is the left hip, with stddev
        of 0.107, since it's hard to locate the left hip bone without external
        anatomical features and since it is often occluded by clothing.

        The implementation here is based off of the descriptions in:
        Ronchi & Perona. "Benchmarking and Error Diagnosis in Multi-Instance Pose
        Estimation." ICCV (2017).
    """
    if points_gt.ndim == 2:
        points_gt = np.expand_dims(points_gt, axis=0)
    if points_pr.ndim == 2:
        points_pr = np.expand_dims(points_pr, axis=0)

    if scale is None:
        scale = compute_instance_area(points_gt)

    n_gt, n_nodes, n_ed = points_gt.shape  # n_ed = 2 or 3 (euclidean dimensions)
    n_pr = points_pr.shape[0]

    # If scalar scale was provided, use the same for each ground truth instance.
    if np.isscalar(scale):
        scale = np.full(n_gt, scale)

    # If scalar standard deviation was provided, use the same for each node.
    if np.isscalar(stddev):
        stddev = np.full(n_nodes, stddev)
    stddev = np.array(stddev)

    # Compute displacement between each pair.
    displacement = np.reshape(points_gt, (n_gt, 1, n_nodes, n_ed)) - np.reshape(
        points_pr, (1, n_pr, n_nodes, n_ed)
    )
    assert displacement.shape == (n_gt, n_pr, n_nodes, n_ed)

    # Convert to pairwise Euclidean distances.
    distance = (displacement**2).sum(axis=-1)  # (n_gt, n_pr, n_nodes)
    assert distance.shape == (n_gt, n_pr, n_nodes)

    # Compute the normalization factor per keypoint.
    if use_cocoeval:
        # If use_cocoeval is True, then compute normalization factor according to cocoeval.
        spread_factor = (2 * stddev) ** 2
        scale_factor = 2 * (scale + np.spacing(1))
    else:
        # If use_cocoeval is False, then compute normalization factor according to the paper.
        spread_factor = stddev**2
        scale_factor = 2 * ((scale + np.spacing(1)) ** 2)
    normalization_factor = np.reshape(spread_factor, (1, 1, n_nodes)) * np.reshape(
        scale_factor, (n_gt, 1, 1)
    )
    assert normalization_factor.shape == (n_gt, 1, n_nodes)

    # Since a "miss" is considered as KS < 0.5, we'll set the
    # distances for predicted points that are missing to inf.
    missing_pr = np.any(np.isnan(points_pr), axis=-1)  # (n_pr, n_nodes)
    assert missing_pr.shape == (n_pr, n_nodes)
    distance[:, missing_pr] = np.inf

    # Compute the keypoint similarity as per the top of Eq. 1.
    ks = np.exp(-(distance / normalization_factor))  # (n_gt, n_pr, n_nodes)
    assert ks.shape == (n_gt, n_pr, n_nodes)

    # Set the KS for missing ground truth points to 0.
    # This is equivalent to the visibility delta function of the bottom
    # of Eq. 1.
    missing_gt = np.any(np.isnan(points_gt), axis=-1)  # (n_gt, n_nodes)
    assert missing_gt.shape == (n_gt, n_nodes)
    ks[np.expand_dims(missing_gt, axis=1)] = 0

    # Compute the OKS.
    n_visible_gt = np.sum(
        (~missing_gt).astype("float32"), axis=-1, keepdims=True
    )  # (n_gt, 1)
    oks = np.sum(ks, axis=-1) / n_visible_gt
    assert oks.shape == (n_gt, n_pr)

    return oks


def match_instances(
    frame_gt: sio.LabeledFrame,
    frame_pr: sio.LabeledFrame,
    stddev: float = 0.025,
    scale: float | None = None,
    threshold: float = 0,
    user_labels_only: bool = True,
) -> tuple[list[tuple[sio.Instance, sio.PredictedInstance, float]], list[sio.Instance]]:
    """Match pairs of instances between ground truth and predictions in a frame.

    Args:
        frame_gt: A `sleap.LabeledFrame` with ground truth instances.
        frame_pr: A `sleap.LabeledFrame` with predicted instances.
        stddev: The expected spread of coordinates for OKS computation.
        scale: The scale for normalizing the OKS. If not set, the bounding box area will
            be used.
        threshold: The minimum OKS between a candidate pair of instances to be
            considered a match.
        user_labels_only: If False, predicted instances in the ground truth frame may be
            considered for matching.

    Returns:
        A tuple of (`positive_pairs`, `false_negatives`).

        `positive_pairs` is a list of 3-tuples of the form
        `(instance_gt, instance_pr, oks)` containing the matched pair of instances and
        their OKS.

        `false_negatives` is a list of ground truth `sleap.Instance`s that could not be
        matched.

    Notes:
        This function uses the approach from the PASCAL VOC scoring procedure. Briefly,
        predictions are sorted descending by their instance-level prediction scores and
        greedily matched to ground truth instances which are then removed from the pool
        of available instances.

        Ground truth instances that remain unmatched are considered false negatives.
    """
    # Sort predicted instances by score.
    scores_pr = np.array(
        [
            instance.score
            for instance in frame_pr.instances
            if hasattr(instance, "score")
        ]
    )
    idxs_pr = np.argsort(-scores_pr, kind="mergesort")  # descending
    scores_pr = scores_pr[idxs_pr]

    if user_labels_only:
        available_instances_gt = frame_gt.user_instances
    else:
        available_instances_gt = frame_gt.instances
    available_instances_gt_idxs = list(range(len(available_instances_gt)))

    positive_pairs = []
    for idx_pr in idxs_pr:
        # Pull out predicted instance.
        instance_pr = frame_pr.instances[idx_pr]

        # Convert instances to point arrays.
        points_pr = np.expand_dims(instance_pr.numpy(), axis=0)
        points_gt = np.stack(
            [
                available_instances_gt[idx].numpy()
                for idx in available_instances_gt_idxs
            ],
            axis=0,
        )

        # Find the best match by computing OKS.
        oks = compute_oks(points_gt, points_pr, stddev=stddev, scale=scale)
        oks = np.squeeze(oks, axis=1)
        assert oks.shape == (len(points_gt),)

        oks[oks <= threshold] = np.nan
        best_match_gt_idx = np.argsort(-oks, kind="mergesort")[0]
        best_match_oks = oks[best_match_gt_idx]
        if np.isnan(best_match_oks):
            continue

        # Remove matched ground truth instance and add as a positive pair.
        instance_gt_idx = available_instances_gt_idxs.pop(best_match_gt_idx)
        instance_gt = available_instances_gt[instance_gt_idx]
        positive_pairs.append((instance_gt, instance_pr, best_match_oks))

        # Stop matching lower scoring instances if we run out of candidates in the
        # ground truth.
        if not available_instances_gt_idxs:
            break

    # Any remaining ground truth instances are considered false negatives.
    false_negatives = [
        available_instances_gt[idx] for idx in available_instances_gt_idxs
    ]

    return positive_pairs, false_negatives


def match_frame_pairs(
    frame_pairs: list[tuple[sio.LabeledFrame, sio.LabeledFrame]],
    stddev: float = 0.025,
    scale: float | None = None,
    threshold: float = 0,
    user_labels_only: bool = True,
) -> tuple[list[tuple[sio.Instance, sio.PredictedInstance, float]], list[sio.Instance]]:
    """Match all ground truth and predicted instances within each pair of frames.

    This is a wrapper for `match_instances()` but operates on lists of frames.

    Args:
        frame_pairs: A list of pairs of `sleap.LabeledFrame`s in the form
            `(frame_gt, frame_pr)`. These can be obtained with `find_frame_pairs()`.
        stddev: The expected spread of coordinates for OKS computation.
        scale: The scale for normalizing the OKS. If not set, the bounding box area will
            be used.
        threshold: The minimum OKS between a candidate pair of instances to be
            considered a match.
        user_labels_only: If False, predicted instances in the ground truth frame may be
            considered for matching.

    Returns:
        A tuple of (`positive_pairs`, `false_negatives`).

        `positive_pairs` is a list of 4-tuples of the form
        `(instance_gt, instance_pr, oks, frame_pair_ind)` containing the matched pair of
        instances and their OKS.

        `false_negatives` is a list of ground truth `sleap.Instance`s that could not be
        matched.
    """
    positive_pairs = []
    false_negatives = []
    for frame_pair_ind, (frame_gt, frame_pr) in enumerate(frame_pairs):
        positive_pairs_frame, false_negatives_frame = match_instances(
            frame_gt,
            frame_pr,
            stddev=stddev,
            scale=scale,
            threshold=threshold,
            user_labels_only=user_labels_only,
        )
        positive_pairs_frame = [(*pp, frame_pair_ind) for pp in positive_pairs_frame]
        positive_pairs.extend(positive_pairs_frame)
        false_negatives.extend(false_negatives_frame)

    return positive_pairs, false_negatives


def compute_generalized_voc_metrics(
    positive_pairs: list[tuple[sio.Instance, sio.PredictedInstance, float, int]],
    false_negatives: list[sio.Instance],
    match_scores: list[float],
    match_score_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),  # 0.5:0.05:0.95
    recall_thresholds: np.ndarray = np.linspace(0, 1, 101),  # 0.0:0.01:1.00
    name: str = "gvoc",
) -> dict[str, float | np.ndarray]:
    """Compute VOC metrics given matched pairs of instances.

    Args:
        positive_pairs: A list of tuples of the form `(instance_gt, instance_pr, _, _)`
            containing the matched pair of instances.
        false_negatives: A list of unmatched instances.
        match_scores: The score obtained in the matching procedure for each matched pair
            (e.g., OKS).
        match_score_thresholds: Score thresholds at which to consider matches as a true
            positive match.
        recall_thresholds: Recall thresholds at which to evaluate Average Precision.
        name: Name to use to prefix returned metric keys.

    Returns:
        A dictionary of VOC metrics.
    """
    detection_scores = np.array([pp[1].score for pp in positive_pairs])

    inds = np.argsort(-detection_scores, kind="mergesort")
    detection_scores = detection_scores[inds]
    match_scores = match_scores[inds]

    precisions = []
    recalls = []

    npig = len(positive_pairs) + len(false_negatives)  # total number of GT instances

    for match_score_threshold in match_score_thresholds:

        tp = np.cumsum(match_scores >= match_score_threshold)
        fp = np.cumsum(match_scores < match_score_threshold)

        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))

        recall = rc[-1]  # best recall at this OKS threshold

        # Ensure strictly decreasing precisions.
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]

        # Find best precision at each recall threshold.
        rc_inds = np.searchsorted(rc, recall_thresholds, side="left")
        precision = np.zeros(rc_inds.shape)
        is_valid_rc_ind = rc_inds < len(pr)
        precision[is_valid_rc_ind] = pr[rc_inds[is_valid_rc_ind]]

        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    AP = precisions.mean(
        axis=1
    )  # AP = average precision over fixed set of recall thresholds
    AR = recalls  # AR = max recall given a fixed number of detections per image

    mAP = precisions.mean()  # mAP = mean over all OKS thresholds
    mAR = recalls.mean()  # mAR = mean over all OKS thresholds

    return {
        name + ".match_score_thresholds": match_score_thresholds,
        name + ".recall_thresholds": recall_thresholds,
        name + ".match_scores": match_scores,
        name + ".precisions": precisions,
        name + ".recalls": recalls,
        name + ".AP": AP,
        name + ".AR": AR,
        name + ".mAP": mAP,
        name + ".mAR": mAR,
    }


def compute_dists(
    positive_pairs: list[tuple[sio.Instance, sio.PredictedInstance, float]]
) -> np.ndarray:
    """Compute Euclidean distances between matched pairs of instances.

    Args:
        positive_pairs: A list of tuples of the form `(instance_gt, instance_pr, _, _)`
            containing the matched pair of instances.

    Returns:
        dists: An array of pairwise distances of shape `(n_positive_pairs, n_nodes)`
    """
    dists = []
    for instance_gt, instance_pr, _, _ in positive_pairs:
        points_gt = instance_gt.numpy()
        points_pr = instance_pr.numpy()

        dists.append(np.linalg.norm(points_pr - points_gt, axis=-1))

    dists = np.array(dists)

    return dists


def compute_dist_metrics(
    dists: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute the Euclidean distance error at different percentiles.

    Args:
        dists: An array of pairwise distances of shape `(n_positive_pairs, n_nodes)`.

    Returns:
        A dictionary of distance metrics.
    """
    results = {
        "dist.dists": dists,
        "dist.avg": np.nanmean(dists),
        "dist.p50": np.nan,
        "dist.p75": np.nan,
        "dist.p90": np.nan,
        "dist.p95": np.nan,
        "dist.p99": np.nan,
    }

    is_non_nan = ~np.isnan(dists)
    if np.any(is_non_nan):
        non_nans = dists[is_non_nan]
        for ptile in (50, 75, 90, 95, 99):
            results[f"dist.p{ptile}"] = np.percentile(non_nans, ptile)

    return results


#Load files
labels_gt_path = '/Path/to/Ground_Truth_Labels.slp' #Manual annotations
labels_pr_path = '/Path/to/Prediction_Labels.slp' #Model predictions over full dataset, test split

sigmas = [0.20617885, 0.25968609, 0.20690924] # 2*s from https://github.com/talmolab/Space-SLEAP/blob/main/Interannotator_labelling_consistency.py 

labels_gt = sio.load_slp(labels_gt_path)
labels_pr = sio.load_slp(labels_pr_path)
labels_gt, labels_pr


#Match frames among manual labels and predicted labels
frame_pairs = []
for lf_gt, lf_pr in zip(labels_gt, labels_pr):
    assert lf_gt.video.filename == lf_pr.video.filename
    assert lf_gt.frame_idx == lf_pr.frame_idx
    frame_pairs.append((lf_gt, lf_pr))

print("Frame pairs:", len(frame_pairs))

#Display metrics
positive_pairs, false_negatives = match_frame_pairs(frame_pairs, stddev=sigmas)

print("Positive pairs:", len(positive_pairs))
print("False negatives:", len(false_negatives))

pair_oks = np.array([oks for _, _, oks, _ in positive_pairs])
mOKS = pair_oks.mean()

print(f"mOKS: {mOKS}")

voc_metrics = compute_generalized_voc_metrics(positive_pairs, false_negatives, match_scores=pair_oks, name="oks_voc")

for k in ["oks_voc.mAP", "oks_voc.mAR"]:
    print(f"{k}: {voc_metrics[k]}")

dist_metrics = compute_dist_metrics(compute_dists(positive_pairs))

for k in ["dist.p50", "dist.p75", "dist.p90", "dist.p95", "dist.p99"]:
    print(f"{k}: {dist_metrics[k]}")


#Plot uncalibrated probability of OKS
sns.set_context("talk")

plt.figure(figsize=(6, 4), dpi=120)
sns.histplot(pair_oks, kde=True, stat="probability", bins=np.linspace(0, 1.0, 20), element="step")
plt.xlabel("Accuracy (Object Keypoint Similarity)")
plt.title("Uncalibrated model accuracy (per detection)")
sns.despine()


#Plot uncalibrated localization errors
node_names = labels_gt.skeleton.node_names
df = pd.DataFrame(dist_metrics["dist.dists"], columns=node_names)
df = df.melt(var_name="Keypoint", value_name="Distance")

sns.set_context("talk")

pal = sns.color_palette("Set2", n_colors=len(node_names))

g = sns.FacetGrid(df, row="Keypoint", hue="Keypoint",
                  aspect=4, height=1.5, palette=pal,
                  subplot_kws={"facecolor": (0, 0, 0, 0)})

g.map(sns.kdeplot, "Distance",
      bw_adjust=0.1, clip_on=False,
      fill=True, alpha=1, linewidth=1.5,
      clip=(0, 100))
g.map(sns.kdeplot, "Distance", clip_on=False, color="w", lw=2.5, bw_adjust=0.1, clip=(0, 100))

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
g.set(xlabel="Error distance (pixels)")
g.figure.suptitle("Localization error", ha="center", x=0.6);


#Plot uncalibrated error distances
node_names = labels_gt.skeleton.node_names
df = pd.DataFrame(dist_metrics["dist.dists"], columns=node_names)
df = df.melt(var_name="Keypoint", value_name="Distance")

sns.set_context("talk")

pal = sns.color_palette("Set2", n_colors=len(node_names))

plt.figure(figsize=(3, 6), dpi=120)
sns.barplot(data=df, x="Keypoint", y="Distance", hue="Keypoint", palette=pal, linewidth=1, edgecolor=".5")
sns.swarmplot(data=df, x="Keypoint", y="Distance", size=3, color=".25", alpha=0.5)
plt.ylim([0, 100])
plt.ylabel("Error distance (pixels)");


#Plot uncalibrated error distances as percentile radial distances from ground truth
percentiles = [50, 75, 90]
prcs = np.nanpercentile(dist_metrics["dist.dists"], percentiles, axis=0)


lf = labels_gt[1] #Edit this to find a representative image
inst = lf[0]
ref_pts = inst.numpy()
img = lf.image
print(img.shape)

plt.figure(figsize=(6, 4), dpi=200)
plt.imshow(img, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())


pal = sns.color_palette("Set2", n_colors=len(ref_pts))
for xy, col in zip(ref_pts, pal):
    plt.plot(xy[0], xy[1], "x", ms=12, mew=2, lw=1, c=col)


cmap = sns.color_palette("viridis", n_colors=len(prcs))
for percentile, prc, col in zip(percentiles, prcs, cmap):
    first_pass = True
    for p, xy in zip(prc, ref_pts):
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

plt.legend(loc="lower right", title="Error");


#Plot uncalibrated precision/recall curcve
plt.figure(figsize=(6, 4), dpi=120, facecolor="w")
for precision, thresh in zip(voc_metrics["oks_voc.precisions"][::2], voc_metrics["oks_voc.match_score_thresholds"][::2]):
    plt.plot(voc_metrics["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="upper right");
plt.title("Model accuracy (overall)")
sns.despine()


#Plot uncalibrated true and false positive area distirbutions
tp_areas = [compute_instance_area(inst.numpy()) for inst, _, _, _ in positive_pairs]
tp_areas = np.array(tp_areas)

print(f"Mean true positive area: {tp_areas.mean()}, std: {tp_areas.std()}")

fn_areas = [compute_instance_area(inst.numpy()) for inst in false_negatives]
fn_areas = np.array(fn_areas)

print(f"Mean false negative area: {fn_areas.mean()}, std: {fn_areas.std()}")

print(f"Mean area ratio: {fn_areas.mean() / tp_areas.mean()}")

plt.figure()
plt.hist(tp_areas, label="TP", alpha=0.5)
plt.hist(fn_areas, label="FN", alpha=0.5)
plt.legend()


#Visualize predictions overlaid onto ground truths
best_pp_inds = np.argsort(pair_oks)[::-1]
best_fp_inds = np.array([pp[3] for pp in positive_pairs])[best_pp_inds]

i = 69 #Choose image here

lf_gt, lf_pr = frame_pairs[best_fp_inds[i]]

img = lf_gt.image

plt.figure(figsize=(6, 4), dpi=200)
plt.imshow(img, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

lw = 3

for inst in lf_gt:
    pts = inst.numpy()
    plt.plot(pts[:, 0], pts[:, 1], "g+", mew=lw)

for inst in lf_pr:
    pts = inst.numpy()
    plt.plot(pts[:, 0], pts[:, 1], "rx", mew=lw, alpha=0.6)

plt.plot(np.nan, np.nan, "g+", label="Ground truth", mew=lw)
plt.plot(np.nan, np.nan, "rx", label="Predicted", mew=lw, alpha=0.6)
plt.legend(loc="lower left")


#Longitudinal model analyses
mission_time_df = []

for inst_gt, inst_pr, oks_pp, fp_ind in positive_pairs:
    lf_gt, lf_pr = frame_pairs[fp_ind]

    p = Path(lf_gt.video.filename)
    day = int(p.parent.name.split(" ")[1])
    cycle = Path(lf_gt.video.filename).parent.parent.name.split(" ")[0]

    pts_gt = inst_gt.numpy()
    pts_pr = inst_pr.numpy()
    dxy = pts_gt - pts_pr
    err = np.sqrt(np.sum(dxy ** 2, axis=-1))
    err = np.nanmean(err)
    frac_vis = ((~np.isnan(pts_pr[:, 0])) & (~np.isnan(pts_gt[:, 0]))).sum() / ((~np.isnan(pts_gt[:, 0])).sum())

    mission_time_df.append({
        "day": day,
        "cycle": cycle,
        "oks": oks_pp,
        "dist": err,
        "frac_vis": frac_vis,
    })

mission_time_df = pd.DataFrame(mission_time_df)
mission_time_df


#Plot uncalibrated longitudinal error distances
sns.set_context("talk")
plt.figure()
sns.scatterplot(x="day", y="dist", data=mission_time_df)

plt.figure()
sns.lineplot(x="day", y="dist", data=mission_time_df)

plt.figure()
sns.pointplot(x="day", y="dist", data=mission_time_df)


#Plot uncalibrated longitudinal Object Keypoint Similarities
sns.set_context("talk")
plt.figure()
sns.scatterplot(x="day", y="oks", data=mission_time_df)

plt.figure()
sns.lineplot(x="day", y="oks", data=mission_time_df)

plt.figure()
sns.pointplot(x="day", y="oks", data=mission_time_df)


#Plot uncalibrated longitudinal fraction visualized
sns.set_context("talk")
plt.figure()
sns.scatterplot(x="day", y="frac_vis", data=mission_time_df)

plt.figure()
sns.lineplot(x="day", y="frac_vis", data=mission_time_df)

plt.figure()
sns.pointplot(x="day", y="frac_vis", data=mission_time_df)


#Plot uncalibrated logitudinal Object Keypoint Similaritie by light cycle
sns.set_context("talk")
plt.figure()
sns.scatterplot(x="day", y="oks", hue="cycle", data=mission_time_df)

plt.figure()
sns.lineplot(x="day", y="oks", hue="cycle", data=mission_time_df)

plt.figure()
sns.pointplot(x="day", y="oks", hue="cycle", data=mission_time_df)


#Plot the longitudinal Object Keypoint Similarity
sns.set_context("talk")
plt.figure(figsize=(6, 4), dpi=120)
sns.pointplot(x="day", y="oks", data=mission_time_df)
plt.xlabel("Mission time (days)")
sns.despine()
plt.ylabel("Accuracy (OKS)");


#Model performance calibrated to human error begins here
#Load interannotator consistency file
labels_consistency_gt_path = "/Path/to/labels_consensus.slp" #Final output of https://github.com/talmolab/Space-SLEAP/blob/main/Interannotator_labelling_consistency.py 
#^contains all labels from each annotator and the consensus labels

#Load model inferred inferences specific to just the interannotator label set
labels_consistency_pr_paths = [
    "/Path/to/Individual Video Inferences/278_05-56-58_5002_feeder_2.predictions.slp",
    "/Path/to/Individual Video Inferences/281_06-00-00_1389_Feeder_2.predictions.slp",
    "/Path/to/Individual Video Inferences/270_06-02-15_1389_Feeder_2.predictions.slp"
]

labels_consistency_gt = sio.load_slp(labels_consistency_gt_path)
labels_consistency_pr_list = [sio.load_slp(path) for path in labels_consistency_pr_paths]
labels_consistency_pr = labels_consistency_pr_list[0]

for labels_pr in labels_consistency_pr_list[1:]:
    labels_consistency_pr.labeled_frames.extend(labels_pr.labeled_frames)

labels_consistency_pr.labeled_frames.sort(
    key=lambda lf: (lf.video.filename, lf.frame_idx)
)

print(f"Ground truth frames: {len(labels_consistency_gt)}")
print(f"Prediction frames (combined): {len(labels_consistency_pr)}")

labels_consistency_gt, labels_consistency_pr

node_names = labels_gt.skeleton.node_names


#Calibration
all_dxy = np.stack((consistency_df["pts"] - consistency_df["consensus"]).to_numpy(), axis=0)
all_err = np.sqrt(np.sum(all_dxy ** 2, axis=-1))
per_node_baseline_mean = np.nanmean(all_err, axis=0)
per_node_baseline_prc50 = np.nanpercentile(all_err, 50, axis=0)
per_node_baseline_prc75 = np.nanpercentile(all_err, 75, axis=0)
per_node_baseline_prc90 = np.nanpercentile(all_err, 90, axis=0)
per_node_baseline_prc95 = np.nanpercentile(all_err, 95, axis=0)
per_node_baseline_prc99 = np.nanpercentile(all_err, 99, axis=0)

per_node_baseline_mean, per_node_baseline_prc90


#Calibrated distance metrics
model_err = dist_metrics["dist.dists"]
model_per_node_mean = np.nanmean(model_err, axis=0)
model_per_node_prc50 = np.nanpercentile(model_err, 50, axis=0)
model_per_node_prc75 = np.nanpercentile(model_err, 75, axis=0)
model_per_node_prc90 = np.nanpercentile(model_err, 90, axis=0)
model_per_node_prc95 = np.nanpercentile(model_err, 95, axis=0)
model_per_node_prc99 = np.nanpercentile(model_err, 99, axis=0)

model_per_node_mean, model_per_node_prc90



cons_frame_pairs = []

# Create a dictionary for quick lookup of predicted frames
# The key should be a tuple of (video_filename, frame_idx)
pr_frames_map = {}
for lf_pr in labels_consistency_pr:
    pr_frames_map[(lf_pr.video.filename, lf_pr.frame_idx)] = lf_pr

# Iterate through ground truth labels and find corresponding predicted labels
for lf_gt in labels_consistency_gt:
    key = (lf_gt.video.filename, lf_gt.frame_idx)
    if key in pr_frames_map:
        lf_pr = pr_frames_map[key]
        cons_frame_pairs.append((lf_gt, lf_pr))
    else:
        # Optionally, handle cases where a ground truth frame has no corresponding prediction
        print(f"Warning: Ground truth frame {lf_gt.video.filename}, frame {lf_gt.frame_idx} has no matching prediction.")


#Metrics
print("Frame pairs:", len(cons_frame_pairs))

cons_positive_pairs, cons_false_negatives = match_frame_pairs(cons_frame_pairs, stddev=sigmas)

print("Positive pairs:", len(cons_positive_pairs))
print("False negatives:", len(cons_false_negatives))

#Calibrated OKS
cons_pair_oks = np.array([oks for _, _, oks, _ in cons_positive_pairs])
cons_mOKS = cons_pair_oks.mean()

print(f"mOKS: {cons_mOKS}") # Changed from mOKS to cons_mOKS

cons_voc_metrics = compute_generalized_voc_metrics(cons_positive_pairs, cons_false_negatives, match_scores=cons_pair_oks, name="oks_voc")

for k in ["oks_voc.mAP", "oks_voc.mAR"]:
    print(f"{k}: {cons_voc_metrics[k]}")


#Calibrated distances
cons_dist_metrics = compute_dist_metrics(compute_dists(cons_positive_pairs))

for k in ["dist.p50", "dist.p75", "dist.p90", "dist.p95", "dist.p99"]:
    print(f"{k}: {cons_dist_metrics[k]}")


#Plot calibrated OKS probabilities
sns.set_context("talk")

plt.figure(figsize=(6, 4), dpi=120)
sns.histplot(cons_pair_oks, kde=True, stat="probability", bins=np.linspace(0, 1.0, 20), element="step")
plt.xlabel("Accuracy (Object Keypoint Similarity)")
plt.title("Calibrated model accuracy (per detection)")
sns.despine()


#Calibrated localization error distribution as kernel density estimate plots
node_names = labels_gt.skeleton.node_names
df = pd.DataFrame(cons_dist_metrics["dist.dists"], columns=node_names)
df = df.melt(var_name="Keypoint", value_name="Distance")

sns.set_context("talk")

# pal = sns.cubehelix_palette(3, rot=-.25, light=.7)
pal = sns.color_palette("Set2", n_colors=len(node_names))

g = sns.FacetGrid(df, row="Keypoint", hue="Keypoint",
                  aspect=4, height=1.5, palette=pal,
                  subplot_kws={"facecolor": (0, 0, 0, 0)})

# Draw the densities in a few steps
g.map(sns.kdeplot, "Distance",
      bw_adjust=0.3, clip_on=False,
      fill=True, alpha=1, linewidth=1.5,
      clip=(0, 100))
g.map(sns.kdeplot, "Distance", clip_on=False, color="w", lw=2.5, bw_adjust=0.3, clip=(0, 100))

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
g.set(xlabel="Error distance (pixels)")
g.figure.suptitle("Localization error", ha="center", x=0.6);


#Visualize calibrated error distance percentile radii overlaid onto representative image
percentiles = [50, 75, 90]
prcs = np.nanpercentile(cons_dist_metrics["dist.dists"], percentiles, axis=0)


# lf = labels_gt[8]
lf = labels_consistency_gt[5]
inst = lf[0]
ref_pts = inst.numpy()
img = lf.image
print(img.shape)

plt.figure(figsize=(6, 4), dpi=200)
plt.imshow(img, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())


pal = sns.color_palette("Set2", n_colors=len(ref_pts))
for xy, col in zip(ref_pts, pal):
    plt.plot(xy[0], xy[1], "x", ms=12, mew=2, lw=1, c=col)


cmap = sns.color_palette("viridis", n_colors=len(prcs))
for percentile, prc, col in zip(percentiles, prcs, cmap):
    first_pass = True
    for p, xy in zip(prc, ref_pts):
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

plt.legend(loc="lower left", title="Error");


#Calibrated Precision-Recall Curve
plt.figure(figsize=(6, 4), dpi=120, facecolor="w")
for precision, thresh in zip(cons_voc_metrics["oks_voc.precisions"][::2], cons_voc_metrics["oks_voc.match_score_thresholds"][::2]):
    plt.plot(cons_voc_metrics["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left");
plt.title("Calibrated model accuracy (overall)")
sns.despine()


#Save data for reporting
assert len(cons_positive_pairs) == len(cons_dist_metrics["dist.dists"])

cons_pred_df = []
for pp_ind, (instance_gt, instance_pr, oks, lf_ind) in enumerate(cons_positive_pairs):
    lf = labels_consistency_gt[lf_ind]
    video_filename = lf.video.filename
    frame_idx = lf.frame_idx
    cons_pred_df.append({
        "video_filename": video_filename,
        "frame_idx": frame_idx,
        "oks": oks,
        "pp_ind": pp_ind,
        "pts_gt": instance_gt.numpy(),
        "pts_pr": instance_pr.numpy(),
        "dist": cons_dist_metrics["dist.dists"][pp_ind],
    })
cons_pred_df = pd.DataFrame(cons_pred_df)
cons_pred_df


#Coerce data for human-model comparisons
all_obs = []

for ind, (pts, consensus) in consistency_df[["pts", "consensus"]].iterrows():
    for node_name, pt_gt, pt_pr in zip(node_names, consensus, pts):
        if np.isnan(pt_gt).any() or np.isnan(pt_pr).any():
            continue

        all_obs.append({
            "ind": ind,
            "Source": "Human",
            "Node": node_name,
            "x_gt": pt_gt[0], "y_gt": pt_gt[1],
            "x_pr": pt_pr[0], "y_pr": pt_pr[1],
            "Error": np.linalg.norm(pt_gt - pt_pr),
        })


for ind, (pts_gt, pts_pr) in cons_pred_df[["pts_gt", "pts_pr"]].iterrows():
    for node_name, pt_gt, pt_pr in zip(node_names, pts_gt, pts_pr):
        if np.isnan(pt_gt).any() or np.isnan(pt_pr).any():
            continue

        all_obs.append({
            "ind": ind,
            "Source": "Model",
            "Node": node_name,
            "x_gt": pt_gt[0], "y_gt": pt_gt[1],
            "x_pr": pt_pr[0], "y_pr": pt_pr[1],
            "Error": np.linalg.norm(pt_gt - pt_pr),
        })


all_obs = pd.DataFrame(all_obs)
all_obs


#Human-model error percentiles
paired_summary = all_obs.groupby(["Node", "Source"])["Error"].agg([
    "median", "mean", "std", "count",
    ("prc75", lambda x: np.percentile(x, 75)),
    ("prc90", lambda x: np.percentile(x, 90)),
    ("prc95", lambda x: np.percentile(x, 95)),
    ("prc99", lambda x: np.percentile(x, 99)),
])
paired_summary


#Visualize human-model error distances as bar graph
sns.set_context("talk")

# pal = sns.color_palette("Set2", n_colors=2)
pal = sns.color_palette("tab10", n_colors=2)

plt.figure(figsize=(3, 6), dpi=120)
sns.barplot(data=all_obs, x="Node", y="Error", hue="Source", palette=pal, linewidth=1, edgecolor=".5", estimator="median", errorbar="se")
# sns.swarmplot(data=all_obs, y="Node", x="Error", hue="Source", dodge=True, size=3, palette="dark:.25", alpha=0.5)
plt.ylim([0, 50])
plt.xlabel("Keypoint")
plt.ylabel("Error distance (pixels)");


#Visualize human-model error distances as error radii at the 90th percentile
paired_summary = all_obs.groupby(["Node", "Source"])["Error"].agg([
    "median", "mean", "std", "count",
    ("prc75", lambda x: np.percentile(x, 75)),
    ("prc90", lambda x: np.percentile(x, 90)),
    ("prc95", lambda x: np.percentile(x, 95)),
    ("prc99", lambda x: np.percentile(x, 99)),
])


lf = labels_gt[1]
inst = lf[0]
ref_pts = inst.numpy()
img = lf.image
print(img.shape)

plt.figure(figsize=(6, 4), dpi=200)
plt.imshow(img, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())


# pal = sns.color_palette("Set2", n_colors=len(ref_pts))
# for xy, col in zip(ref_pts, pal):
#     plt.plot(xy[0], xy[1], "x", ms=12, mew=2, lw=1, c=col)


cmap = sns.color_palette("tab10", n_colors=2)
for source, col in zip(["Human", "Model"], cmap):
    first_pass = True
    for node_name, xy in zip(node_names, ref_pts):
        radius = paired_summary.loc[(node_name, source)]["prc90"]

        plt.gca().add_patch(plt.Circle(
            xy,
            radius,
            fill=False,
            ec=col,
            lw=2,
            alpha=0.8,
            label=source if first_pass else None,
            zorder=999,
        ))
        first_pass = False

plt.legend(loc="lower right", title="Error (90%)");

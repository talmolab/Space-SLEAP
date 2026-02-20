#Pipeline for standardizing RR-1 videos to a common resolution of 640 x 480 pixels from a common centerpoint
!pip install sleap-io

import numpy as np
import pandas as pd
import sleap_io as sio
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copyfile

raw_videos_folder = "Path/to/raw/videos"
target_videos_folder = "Path/to/desired/output"
overwrite = False

#Find all feeder MP4s; trimmers sometimes named files filter, Filter, feeder or Feeder
mp4s = list(raw_videos_folder.glob("**/*eeder*.mp4"))
mp4s += list(raw_videos_folder.glob("**/*ilter*.mp4"))
print(f"Found {len(mp4s)} videos.")

#Defines logic for aspect ratio, if original video is larger in either axis, downsampling occurs; if larger padding occurs
def compute_center_crop(height, width, target_aspect_ratio):
    """Returns the cropping coordinates to get the video as close as possible to the target aspect ratio."""
    aspect_ratio = width / height

    if aspect_ratio > target_aspect_ratio:
        # Video is wider than target
        # aspect_ratio = width / height
        # (w0 - dx) / h0 = target_aspect_ratio
        # w0 - dx = target_aspect_ratio * h0
        # - dx = target_aspect_ratio * h0 - w0
        # dx = w0 - target_aspect_ratio * h0
        dx = width - (target_aspect_ratio * height)
        new_width = width - dx
        x_offset = dx // 2

        new_height = height
        y_offset = 0
    elif aspect_ratio < target_aspect_ratio:
        # Video is taller than target
        # width / (height - dy) = target_aspect_ratio
        # width = target_aspect_ratio * (height - dy)
        # width = target_aspect_ratio * height - target_aspect_ratio * dy
        # width - target_aspect_ratio * height = - target_aspect_ratio * dy
        # - width + target_aspect_ratio * height = target_aspect_ratio * dy
        # target_aspect_ratio * dy = target_aspect_ratio * height - width
        # dy = (target_aspect_ratio * height - width) / target_aspect_ratio
        # dy = height - (width / target_aspect_ratio)
        dy = height - (width / target_aspect_ratio)
        new_height = height - dy
        y_offset = dy // 2

        new_width = width
        x_offset = 0

    else:
        # Video is already in the right size.
        new_height, new_width, x_offset, y_offset = height, width, 0, 0

    new_height = int(new_height)
    new_width = int(new_width)
    x_offset = int(x_offset)
    y_offset = int(y_offset)

    return new_height, new_width, x_offset, y_offset


#Target resolution
target_resolution = (480, 640)
target_height, target_width = target_resolution
target_aspect_ratio = target_width / target_height

metadata = []

for i in range(len(mp4s)):
    mp4 = mp4s[i]
    target_mp4 = target_videos_folder / mp4.relative_to(raw_videos_folder)
    target_mp4.parent.mkdir(parents=True, exist_ok=True)

    md = {
        "source_video": mp4,
        "target_video": target_mp4,
    }

    # Load source video.
    video = sio.load_video(mp4)
    height, width = video.shape[1:3]

    md["frames"] = video.shape[0]
    md["source_height"] = video.shape[1]
    md["source_width"] = video.shape[2]

    # Compute center cropping coordinates.
    new_height, new_width, x_offset, y_offset = compute_center_crop(height, width, target_aspect_ratio)
    md["x_offset"] = x_offset
    md["y_offset"] = y_offset
    md["crop_height"] = new_height
    md["crop_width"] = new_width

    if target_mp4.exists():
        can_load = False
        is_scaled = False
        try:
            target_video = sio.load_video(target_mp4)
            is_scaled = target_video.shape[1] == target_height and target_video.shape[2] == target_width
            can_load = True
        except:
            pass

        if can_load and is_scaled and not overwrite:
            md["target_height"] = target_video.shape[1]
            md["target_width"] = target_video.shape[2]
            metadata.append(md)
            print(f"[{i + 1}/{len(mp4s)}] Skipping {mp4} because it already exists.")
            continue

    # Just copy if we're already at the target.
    if height == target_height and width == target_width:
        # Copy and continue.
        copyfile(mp4, target_mp4)

        target_video = sio.load_video(target_mp4)
        md["target_height"] = target_video.shape[1]
        md["target_width"] = target_video.shape[2]
        metadata.append(md)
        print(f"[{i + 1}/{len(mp4s)}] Copied {mp4} because it already has the correct size.")
        continue

    # Create writer for new video.
    with sio.VideoWriter(target_mp4) as writer:
        for frame in tqdm(video):
            # Apply the center crop.
            frame = frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width]

            # Rescale.
            if new_height != target_height or new_width != target_width:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # Save.
            writer(frame)

    target_video = sio.load_video(target_mp4)
    md["target_height"] = target_video.shape[1]
    md["target_width"] = target_video.shape[2]
    metadata.append(md)
    print(f"[{i + 1}/{len(mp4s)}] Saved: {target_mp4}")


metadata = pd.DataFrame(metadata)
metadata.to_csv(target_videos_folder / "metadata.csv", index=False)

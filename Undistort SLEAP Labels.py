#Pipeline for translating SLEAP labels based first on video standardization https://github.com/talmolab/Space-SLEAP/blob/main/Standardize%20Archived%20Videos.py
#Then on undistortion parameters via the video undistortion pipeline https://github.com/talmolab/spacecage-undistort
!pip install sleap-io

!git clone https://github.com/talmolab/spacecage-undistort
%cd spacecage-undistort
!git switch -c master origin/master
!pip install .
%cd ..

import numpy as np
import pandas as pd
import sleap_io as sio
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copyfile
import spacecage_undistort
import spacecage_undistort.coordinates
from spacecage_undistort import transform_slp_coordinates

#Video processing in this project follows: Raw Videos -> Standardized Videos -> Undistorted Videos
#Annotations were performed on raw videos, and also need to be translated

#First we translate annotations from raw video coordinates to Standardized video coordinates
#Generate metadata by measuring translations from raw videos to Standardized videos
raw_videos_folder = "Path/to/Raw/Videos"
new_videos_folder = "Path/to/Standardized/Videos"

videos_md = pd.read_csv(new_videos_folder / "metadata.csv")
videos_md

raw_labels_path = "/Path/to/Raw_Labels.slp"
new_labels_path = "/Path/to/Standardized_Labels.slp"

labels = sio.load_file(raw_labels_path)

# Check that all raw videos were found and have corresponding standardized one.
data_check = []
for video in labels.videos:
    data_check.append({
        "video": video.filename,
        "exists": video.exists(),
        "has_rescaled": videos_md.source_video.str.contains(video.filename).any(),
    })

data_check = pd.DataFrame(data_check)
data_check


#Translate labels
for raw_video in labels.videos:

    #Find frames that correspond to this video
    lfs = [lf for lf in labels if lf.video == raw_video]

    #Get rescaling metadata
    md = videos_md.loc[videos_md.source_video == raw_video.filename]
    if len(md) == 0:
        print(f"No metadata found for {raw_video.filename}")
        continue
    md = md.iloc[0]
    x_offset, y_offset = float(md["x_offset"]), float(md["y_offset"])
    x_scale = md["target_width"] / md["crop_width"]
    y_scale = md["target_height"] / md["crop_height"]

    #Update the points
    for lf in lfs:
        insts = []
        for inst in lf:
            pts = inst.numpy()
            pts[:, 0] -= x_offset
            pts[:, 1] -= y_offset
            pts[:, 0] *= x_scale
            pts[:, 1] *= y_scale

            inst_adjusted = inst.from_numpy(pts, skeleton=inst.skeleton, track=inst.track, tracking_score=inst.tracking_score)
            insts.append(inst_adjusted)
        lf.instances = insts
      
#Updated associated videos
labels.replace_filenames(filename_map={src: dst for _, (src, dst) in videos_md[["source_video", "target_video"]].iterrows()})
labels.videos

#Save the translated labels
Path(new_labels_path).parent.mkdir(parents=True, exist_ok=True)
labels.save(new_labels_path)


#Now we transform the Standardized labels to the Undistorted video coordinates
transform_slp_coordinates(new_labels_path,
                          "Path/to/Undistorted_Labels.slp",
                          "Path/to/Undistort_Protocol_calibration.yml",
                          )

#View videos in undistorted label files
undistorted_vids_slp = "Path/to/Undistorted_Labels.slp"

if 'undistorted_vids_slp' in locals():
    labels = sio.load_file(undistorted_vids_slp)
    print("Video filenames in the SLEAP project:")
    for video in labels.videos:
        print(video.filename)
else:
    print("Please define the 'old_labels_path' variable with the path to your SLEAP project file.")

#Update undistorted label files to include undistorted videos
labels = sio.load_file(undistorted_vids_slp, open_videos=False)

# Fix paths using prefix replacement, include just path prefixes here
labels.replace_filenames(prefix_map={
    new_videos_folder:
    "/Path/to/Undistorted/Videos",
})

# Save labels with updated paths
labels.save(undistorted_vids_slp)

#Double check that slp file has updated video files
if 'undistorted_vids_slp' in locals():
    labels = sio.load_file(undistorted_vids_slp)
    print("Video filenames in the SLEAP project:")
    for video in labels.videos:
        print(video.filename)
else:
    print("Please define the 'old_labels_path' variable with the path to your SLEAP project file.")

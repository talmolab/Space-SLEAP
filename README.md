# Space-SLEAP
A repo for scripts associated with the Space SLEAP project, which points SLEAP to videos of freely behaving mice aboard the International Space Station and performs kinematic analyses.

![Alt text](https://github.com/talmolab/Space-SLEAP/blob/main/Images/xz_gforceExample.gif)

### Here we include:
- Standardize Archived Videos.py for standardizing video resolution by a common centerpoint
- Undistort SLEAP Labels.py for translating SLEAP labels across video correction phases
- interannotator_labelling_consistency.py for assessing labelling consistency among human annotators
- Space SLEAP Model Evaluation.py for assessing model performance relative to human error
- Circling Analysis, a pipeline to estimate kinematics-based centrifugal acceleration
  
🐍 Python ≥ 3.10

To run scripts associated with this project, one must request the video dataset and .slp files from NASA's Open Science Data Repository (OSDR), project [OSD-952](https://osdr.nasa.gov/bio/repo/data/studies/OSD-952), doi.org/10.26030/6m3b-gd71




This work was supported by NASA grant 80NSSC24K0346

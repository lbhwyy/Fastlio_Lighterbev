# FastLIO LighterBEV Workspace

ROS1 catkin workspace that combines FAST-LIO odometry with a `fast_lio_sam_sc_qn` loop-closure stack built around LighterBEV place recognition and Nano-GICP refinement.

## What is in this repository

- `src/fast_lio_sam_sc_qn`: main integration package in this workspace.
- `src/third_party/FAST_LIO`: vendored FAST-LIO package used as the LiDAR-inertial odometry frontend.
- `src/third_party/LighterBEV`: vendored LighterBEV inference code and shipped model checkpoint.
- `src/third_party/nano_gicp`: vendored Nano-GICP package.
- `src/third_party/livox_ros_driver`: vendored Livox ROS driver for Livox-based FAST-LIO launch files.
- `src/third_party/fastlio_config_launch`: dataset-specific launch and config files used by this workspace.
- `scripts/record_nclt_run.sh`: helper script for replaying a rosbag and recording the RViz session.

`vendored` means a third-party dependency is copied into this repository instead of being fetched from a package manager or git submodule. That makes the workspace self-contained, but it also means the repository maintainer must preserve upstream licenses and document where the code came from.

## Requirements

- Ubuntu with ROS Noetic
- Catkin build tools
- PCL
- Eigen3
- OpenCV
- GTSAM
- libtorch (PyTorch C++ distribution)

For Livox sensors, source a workspace that provides `livox_ros_driver` before launching the Livox FAST-LIO pipelines.

## Configure libtorch

This repository no longer hardcodes a machine-local libtorch path. Point CMake to your local libtorch install by setting `Torch_DIR` to the directory that contains `TorchConfig.cmake`.

Example:

```bash
export Torch_DIR=/path/to/libtorch/share/cmake/Torch
```

If you prefer, you can also pass it directly during configuration through catkin/CMake.

## Build

```bash
source /opt/ros/noetic/setup.bash
cd /path/to/Fastlio_lighterbev_ws
catkin_make -DTorch_DIR="${Torch_DIR}"
source devel/setup.bash
```

## Run

Launch the integrated node plus the FAST-LIO frontend:

```bash
roslaunch fast_lio_sam_sc_qn run.launch lidar:=nclt
```

Common launch arguments:

- `lidar`: selects which FAST-LIO launch file is included. Examples in this workspace include `nclt`, `ouster`, `livox`, `kitti`, and `mulran`.
- `rviz`: whether to launch RViz.
- `sam_downsample`: whether to downsample point clouds before the LighterBEV preprocessing stage.
- `downsample_voxel_size`: voxel size used when `sam_downsample:=true`.

## Runtime outputs

- BEV PNG export is controlled by `LighterBEV.save_png` in `src/fast_lio_sam_sc_qn/config/config.yaml`.
- `LighterBEV.save_dir` can override the default BEV image directory. If left empty, it defaults to `src/fast_lio_sam_sc_qn/<seq_name>/bev_imgs`.
- Bag, PCD, and KITTI-format export settings live under the `result` section of the same config.

## NCLT recording helper

`scripts/record_nclt_run.sh` no longer assumes a bag file exists on a specific local disk. Set `BAG_PATH` explicitly before running:

```bash
export BAG_PATH=/path/to/2012-01-15.bag
./scripts/record_nclt_run.sh
```

## License and third-party code

- Original integration code in this repository is currently distributed under the repository-level notice in `LICENSE`.
- Vendored dependencies keep their own upstream licenses and attribution requirements.
- See `THIRD_PARTY_NOTICES.md` before publishing or redistributing the repository.

If you intend to make the repository publicly open source, review `LICENSE` and `THIRD_PARTY_NOTICES.md` first and replace the repository-level license notice with the license you actually want to grant.

# Third-Party Notices

This repository vendors several upstream components under `src/third_party/`. They are not relicensed by the repository-level `LICENSE` file.

## Included components

### FAST_LIO

- Path: `src/third_party/FAST_LIO`
- Purpose: LiDAR-inertial odometry frontend and sensor-specific launch/config files.
- Local license evidence: `src/third_party/FAST_LIO/LICENSE`
- Local package metadata: `src/third_party/FAST_LIO/package.xml` declares `BSD`

### livox_ros_driver

- Path: `src/third_party/livox_ros_driver`
- Purpose: Livox ROS driver required by Livox FAST-LIO launch files.
- Local package metadata: `src/third_party/livox_ros_driver/livox_ros_driver/package.xml` declares `MIT`

### nano_gicp

- Path: `src/third_party/nano_gicp`
- Purpose: point cloud registration backend used for loop-closure refinement.
- Local package metadata: `src/third_party/nano_gicp/package.xml` declares `MIT`
- Additional local file headers mention BSD-3-Clause for some bundled sources and headers.

### LighterBEV

- Path: `src/third_party/LighterBEV`
- Purpose: BEV descriptor extraction and matching logic, including the shipped `pca_kitti_best.pt` checkpoint.
- Local license evidence: no standalone license file was found in the current working tree.
- Action required: verify the provenance and redistribution terms for this directory and its model checkpoint before publishing the repository as an open-source project.

### fastlio_config_launch

- Path: `src/third_party/fastlio_config_launch`
- Purpose: dataset-specific launch and YAML files used by this workspace.
- Local license evidence: no standalone license file was found in the current working tree.
- Action required: document the origin of these launch/config files if they were copied from another repository or publication artifact.

## Maintainer checklist before public release

- Confirm that each vendored directory has a known upstream source.
- Preserve upstream LICENSE files when available.
- Preserve file-level copyright and license headers.
- Verify that bundled model weights can be redistributed.
- Replace the repository-level `LICENSE` if you intend to grant a public open-source license for your original integration code.

## PRISM (Projection + Interpolation + Sensor-color Mapping)

High-performance ROS2 pipeline for LiDAR-to-multi-camera projection, color extraction, and fused colored point cloud publishing.

### Packages and nodes
- `prism_fusion_node` (main):
  - Subscribes: LiDAR (`/ouster/points`) and two cameras (`/usb_cam_1/image_raw`, `/usb_cam_2/image_raw`)
  - Optional LiDAR interpolation (FILC-style grid) to increase vertical resolution
  - Projects points to cameras, extracts per-camera colors, fuses to one RGB point cloud
  - Publishes: colored cloud on `/ouster/points/colored`
- `projection_debug_node` (debug visualization):
  - Subscribes: LiDAR (raw or interpolated) and cameras
  - Publishes: overlay images under `/prism/projection_debug/<camera_id>` and stats `/prism/projection_debug/statistics`

### Build & run
```bash
# In your ROS2 workspace
colcon build
source install/setup.bash

# Launch PRISM fusion + debug nodes (parameters come from prism/config/prism_params.yaml)
ros2 launch prism prism.launch.py
```

### Topics
- Input
  - `/ouster/points` (sensor_msgs/PointCloud2)
  - `/usb_cam_1/image_raw`, `/usb_cam_2/image_raw` (sensor_msgs/Image)
- Output
  - `/ouster/points/colored` (sensor_msgs/PointCloud2, RGB)
  - `/prism/debug/interpolated` (optional, interpolated LiDAR cloud)
  - `/prism/projection_debug/camera_1`, `/prism/projection_debug/camera_2` (overlay images)
  - `/prism/projection_debug/statistics` (std_msgs/String)

### Calibration
- Multi-camera calibration files (CALICO-style) are loaded from:
  - `config/multi_camera_intrinsic_calibration.yaml`
  - `config/multi_camera_extrinsic_calibration.yaml`
- Camera IDs must match YAML keys (e.g., `camera_1`, `camera_2`).

### Key parameters (see `config/prism_params.yaml`)
- `topics.*`
  - `lidar_input`, `camera_1_input`, `camera_2_input`, `colored_output`, `debug_interpolated`
- `synchronization.*`
  - `queue_size` (default 1)
  - `min_interval_ms` (default 0; set >0 to throttle)
  - `use_latest_image` (true): LiDAR drives pipeline; latest camera images cached
  - `image_freshness_ms` (150): heuristic freshness guard
- `interpolation.*`
  - `enabled` (true/false)
  - `scale_factor` (e.g., 2.0)
  - `input_channels` (e.g., 32)
  - `grid_mode` (true): enables FILC-style 1024×N grid interpolation
  - `discontinuity_threshold`
  - `output_topic`
- `projection.*`
  - `enable_distortion_correction`, `enable_frustum_culling`
  - `min_depth`, `max_depth`
  - `parallel_cameras` (true)
  - `sample_stride` (point subsampling; 2 = every 2nd point)
  - `max_points` (cap per frame; 0 disables)
- `extraction.*`, `fusion.*` (color extraction and fusion strategy/config)

### FILC-style grid interpolation (grid_mode: true)
- Organizes the 3D cloud onto a fixed `1024 × input_channels` grid using azimuth-based nearest-neighbor per column.
- Vertically blends adjacent rings with discontinuity-aware interpolation.
- Upscales rows by `scale_factor` (e.g., `32 × 2 = 64`).
- Improves projection density and color coverage with consistent angular sampling.

### Performance tips
- Theoretical max publish rate ≈ `min(LiDAR_rate, 1000 / avg_processing_ms)`.
  - Example: `avg_processing_time ≈ 42 ms` → ~23.8 Hz upper-bound; if LiDAR is ~19 Hz, output caps near 19 Hz.
- If output rate is lower than expected:
  - Reduce per-frame work: increase `projection.sample_stride` (e.g., 3 or 4) and/or lower `projection.max_points` (e.g., 80k–100k)
  - Keep `synchronization.min_interval_ms = 0` to avoid artificial throttling
  - Ensure QoS for colored cloud is sensor-style (keep_last(1), best_effort) to avoid backpressure
  - Disable excessive logging; use DEBUG for per-frame logs

### QoS & reliability
- Colored/interpolated outputs use `SensorDataQoS` (keep_last(1), best_effort) by default to minimize latency and queueing.
- If your downstream consumers require reliability, consider republishing with a reliable QoS bridge at the cost of latency.

### Debugging
- Runtime stats:
```bash
ros2 topic echo /prism/projection_debug/statistics --once --full-length --truncate-length 0
```
- Publish rate:
```bash
ros2 topic hz /ouster/points/colored
```
- Visual overlays per camera are available on `/prism/projection_debug/<camera_id>`.

### Launch parameters override
- Override parameters at launch via a custom YAML or by passing `params_file`:
```bash
ros2 launch prism prism.launch.py params_file:=/path/to/your_params.yaml
```

### License
- See package manifest and repository license files.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRISM (Point-cloud & RGB Integrated Sensing Module) is a high-performance ROS2 package for LiDAR-camera fusion. It performs real-time point cloud interpolation (32→96 channels) and multi-camera color mapping for enhanced 3D perception.

## Build Commands

```bash
# Build the package
cd ~/ROS2_Workspace/ros2_ws  # Or use $ROS2_WS environment variable
colcon build --packages-select prism

# Build with optimizations enabled
colcon build --packages-select prism --cmake-args -DPRISM_ENABLE_NATIVE_OPT=ON

# Clean build
rm -rf build/prism install/prism
colcon build --packages-select prism
```

## Test Commands

```bash
# Run all tests
colcon test --packages-select prism
colcon test-result --verbose

# Run specific unit test
cd ~/ROS2_Workspace/ros2_ws  # Or use $ROS2_WS environment variable
./build/prism/test_interpolation_engine
./build/prism/test_projection_engine
./build/prism/test_calibration_manager
./build/prism/test_memory_pool
./build/prism/test_execution_mode
./build/prism/test_simple_projection

# Run tests with output
colcon test --packages-select prism --event-handlers console_direct+
```

## Launch Commands

```bash
# Source ROS2 environment first
source /opt/ros/humble/setup.bash
source ~/ROS2_Workspace/ros2_ws/install/setup.bash  # Or $ROS2_WS/install/setup.bash

# Main PRISM launch
ros2 launch prism prism.launch.py

# Debug projection visualization
ros2 launch prism projection_debug.launch.py
```

## Architecture Overview

### Processing Pipeline
1. **Interpolation Engine** (`src/interpolation/`): Upsamples LiDAR from 32 to 96 channels using Catmull-Rom splines with SIMD optimization
2. **Projection Engine** (`src/projection/`): Projects 3D points to dual camera image planes using calibrated intrinsic/extrinsic parameters
3. **Color Extraction** (`src/projection/color_extractor.cpp`): Samples RGB values from multiple cameras with bilinear interpolation
4. **Multi-Camera Fusion** (`src/projection/multi_camera_fusion.cpp`): Combines colors from multiple cameras based on visibility and confidence

### Key Components
- **CalibrationManager** (`src/core/calibration_manager.cpp`): Loads and manages camera intrinsics/extrinsics from YAML configs
- **MemoryPool** (`src/core/memory_pool.cpp`): Pre-allocated memory management for zero-copy operations
- **PointCloudSOA** (`src/core/point_cloud_soa.cpp`): Structure-of-Arrays layout for SIMD-friendly point cloud processing
- **ExecutionMode** (`src/execution_mode.cpp`): Runtime mode selection (CPU/GPU/Hybrid)

### Node Structure
- `prism_interpolation_node`: Standalone interpolation processing
- `prism_projection_debug_node`: Visualization and debugging
- `prism_color_node`: Full color mapping pipeline
- `prism_fusion_node`: Complete LiDAR-camera fusion

### Configuration Files
- `config/prism_params.yaml`: Runtime parameters
- `config/multi_camera_intrinsic_calibration.yaml`: Camera matrix, distortion coefficients
- `config/multi_camera_extrinsic_calibration.yaml`: LiDAR-to-camera transformations

### Performance Optimizations
- SIMD kernels for vectorized operations (`src/interpolation/simd_kernels.cpp`)
- OpenMP parallelization for multi-core processing
- SOA memory layout for cache efficiency
- Pre-allocated memory pools to avoid runtime allocation

## Development Workflow

1. For interpolation changes: Modify `interpolation/` components, test with `test_interpolation_engine`
2. For projection changes: Update `projection/` modules, verify with `test_projection_engine` and `projection_debug_node`
3. For calibration: Edit YAML configs in `config/`, test with `test_calibration_manager`
4. For performance: Use `benchmark/` tests, enable native optimizations with cmake flag

## Key Dependencies
- ROS2 Humble
- PCL 1.10+ (point cloud processing)
- OpenCV 4+ (image operations)
- Eigen3 (linear algebra)
- yaml-cpp (configuration loading)
- TBB (parallel algorithms)
# PRISM Package Structure Guide

## Overview

PRISM (Point-cloud and RGB Integrated Sensing Module) is a ROS2 package designed for advanced LiDAR-camera fusion, combining high-resolution point cloud interpolation with precision color mapping. This document provides detailed documentation of the package structure, dependencies, and architectural patterns.

## Directory Organization

```
prism/
├── CMakeLists.txt              # Build configuration
├── package.xml                 # ROS2 package manifest
├── readme.md                   # Project overview
├── include/                    # Public header files
│   └── prism/                  # Package namespace headers
│       ├── core/               # Core functionality headers
│       ├── interpolation/      # Interpolation algorithm headers
│       ├── projection/         # Camera projection headers
│       ├── fusion/             # LiDAR-camera fusion headers
│       └── utils/              # Utility function headers
├── src/                        # Implementation source files
│   ├── core/                   # Core module implementations
│   ├── interpolation/          # Interpolation algorithms
│   ├── projection/             # Projection utilities
│   ├── fusion/                 # Sensor fusion logic
│   ├── nodes/                  # ROS2 node implementations
│   └── utils/                  # Utility implementations
├── docs/                       # Documentation
│   ├── package_structure.md    # This document
│   ├── interpolation_integration.md
│   ├── projection_color_mapping.md
│   └── performance_optimization.md
├── config/                     # Configuration files
│   ├── prism_params.yaml                       # Node parameters
│   ├── multi_camera_intrinsic_calibration.yaml # Camera intrinsics
│   └── multi_camera_extrinsic_calibration.yaml # LiDAR-Camera extrinsics
├── test/                       # Unit and integration tests
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── benchmark/              # Performance benchmarks
└── launch/                     # ROS2 launch files
    ├── prism_node.launch.py
    ├── multi_camera.launch.py
    └── benchmark.launch.py
```

## Module Responsibilities

### Core Module (`src/core/`, `include/prism/core/`)
The core module provides fundamental data structures and base classes:

**Key Components:**
- `PointCloudTypes.hpp` - Custom point cloud type definitions with RGB and intensity
- `CameraModel.hpp` - Camera intrinsic/extrinsic parameter management
- `SensorFusion.hpp` - Base sensor fusion interface
- `DataSynchronizer.hpp` - Time-based data synchronization utilities

**Example Header Structure:**
```cpp
// include/prism/core/PointCloudTypes.hpp
#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace prism {
namespace core {

struct EIGEN_ALIGN16 PointXYZIRGB {
    PCL_ADD_POINT4D
    float intensity;
    union {
        struct {
            uint8_t b;
            uint8_t g;
            uint8_t r;
            uint8_t a;
        };
        uint32_t rgba;
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using PointCloudXYZIRGB = pcl::PointCloud<PointXYZIRGB>;

} // namespace core
} // namespace prism

POINT_CLOUD_REGISTER_POINT_STRUCT(
    prism::core::PointXYZIRGB,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint32_t, rgba, rgba)
)
```

### Interpolation Module (`src/interpolation/`, `include/prism/interpolation/`)
Advanced point cloud interpolation algorithms based on FILC improvements:

**Key Components:**
- `CubicInterpolator.hpp` - Catmull-Rom (cardinal Hermite spline) cubic interpolation
- `BeamAltitudeManager.hpp` - OS1-32 beam angle management
- `AdaptiveInterpolation.hpp` - Discontinuity-aware interpolation
- `InterpolationOptimizer.hpp` - SIMD-optimized algorithms

**Core Algorithm Pattern:**
```cpp
// include/prism/interpolation/CubicInterpolator.hpp
namespace prism {
namespace interpolation {

class CubicInterpolator {
public:
    struct InterpolationResult {
        std::vector<pcl::PointXYZI> points;
        std::vector<float> confidence_scores;
        std::chrono::microseconds processing_time;
    };
    
    InterpolationResult interpolate(
        const pcl::PointCloud<pcl::PointXYZI>& input,
        float scale_factor,
        const std::vector<float>& beam_altitudes);
        
private:
    float catmullRomSpline(float p0, float p1, float p2, float p3, float t);  // Cardinal Hermite with tension=0.5
    bool detectDiscontinuity(const pcl::PointXYZI& p1, const pcl::PointXYZI& p2);
};

} // namespace interpolation
} // namespace prism
```

### Projection Module (`src/projection/`, `include/prism/projection/`)
Camera-LiDAR coordinate transformation and projection utilities:

**Key Components:**
- `ProjectionManager.hpp` - Multi-camera projection coordinator
- `CalibrationLoader.hpp` - YAML calibration data parser
- `DistortionCorrection.hpp` - Lens distortion correction algorithms
- `DepthBuffer.hpp` - Z-buffering for occlusion handling (v1.1+). v1.0은 근사적 오클루전 처리.

**Integration with CALICO:**
```cpp
// include/prism/projection/ProjectionManager.hpp
namespace prism {
namespace projection {

class ProjectionManager {
    struct CameraConfig {
        cv::Mat camera_matrix;
        cv::Mat distortion_coeffs;
        Eigen::Matrix4d extrinsic_matrix;
        cv::Size image_size;
    };
    
    std::vector<cv::Point2f> projectToCamera(
        const std::vector<pcl::PointXYZI>& lidar_points,
        size_t camera_index);
    
    cv::Vec3b extractColor(
        const cv::Mat& image,
        const cv::Point2f& pixel_coord,
        InterpolationMethod method = InterpolationMethod::BILINEAR);
};

} // namespace projection
} // namespace prism
```

### Fusion Module (`src/fusion/`, `include/prism/fusion/`)
High-level sensor fusion orchestration:

**Key Components:**
- `PRISMFusionNode.hpp` - Main ROS2 node implementation
- `MultiCameraFusion.hpp` - Multi-camera color fusion strategies
- `TemporalAlignment.hpp` - Time-based sensor synchronization
- `QualityAssessment.hpp` - Fusion result quality metrics

## Header File Architecture

### Dependency Hierarchy
```
prism/core/            (Level 0 - Foundation)
    ↓
prism/utils/           (Level 1 - Utilities)
    ↓
prism/interpolation/   (Level 2 - Algorithms)
prism/projection/      
    ↓
prism/fusion/          (Level 3 - Integration)
```

### Include Patterns
```cpp
// Standard pattern for PRISM headers
#pragma once

// System includes
#include <vector>
#include <memory>
#include <chrono>

// Third-party includes
#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// PRISM includes (hierarchical order)
#include "prism/core/PointCloudTypes.hpp"
#include "prism/utils/CommonTypes.hpp"

namespace prism {
namespace module_name {

// Implementation

} // namespace module_name
} // namespace prism
```

## Source File Implementation Patterns

### Node Implementation Structure
```cpp
// src/nodes/prism_fusion_node.cpp
#include "prism/fusion/PRISMFusionNode.hpp"

namespace prism {
namespace fusion {

PRISMFusionNode::PRISMFusionNode(const rclcpp::NodeOptions& options)
    : Node("prism_fusion_node", options) {
    
    // 1. Parameter declaration and retrieval
    declareParameters();
    loadParameters();
    
    // 2. Component initialization
    initializeInterpolator();
    initializeProjection();
    
    // 3. ROS2 interface setup
    setupSubscribers();
    setupPublishers();
    setupTimers();
    
    // 4. Performance monitoring
    setupMetrics();
}

} // namespace fusion
} // namespace prism
```

### Algorithm Implementation Pattern
```cpp
// src/interpolation/cubic_interpolator.cpp
namespace prism {
namespace interpolation {

CubicInterpolator::InterpolationResult 
CubicInterpolator::interpolate(const pcl::PointCloud<pcl::PointXYZI>& input,
                              float scale_factor,
                              const std::vector<float>& beam_altitudes) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Pre-processing
    validateInput(input, scale_factor);
    prepareOutputCloud(input.size() * scale_factor);
    
    // Core algorithm (with TBB parallel processing)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, input.width),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t col = range.begin(); col != range.end(); ++col) {
                processColumn(input, col, scale_factor, beam_altitudes);
            }
        });
    
    // Post-processing and result packaging
    auto end_time = std::chrono::high_resolution_clock::now();
    return {std::move(result_cloud_), confidence_scores_, 
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)};
}

} // namespace interpolation
} // namespace prism
```

## Build System Configuration

### CMakeLists.txt Structure
```cmake
cmake_minimum_required(VERSION 3.16)
project(prism)

# Compiler configuration
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
  # Optional native optimization
  option(PRISM_ENABLE_NATIVE_OPT "Enable native CPU optimizations" OFF)
  if(PRISM_ENABLE_NATIVE_OPT)
    add_compile_options(-march=native)
  endif()
endif()

# C++17 requirement for advanced features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Third-party dependencies
find_package(PCL 1.12 REQUIRED COMPONENTS common io kdtree search)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc calib3d)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(TBB REQUIRED)

# Performance libraries (TBB is required, no OpenMP)
find_package(TBB REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Core library
add_library(${PROJECT_NAME}_core
  src/core/PointCloudTypes.cpp
  src/core/CameraModel.cpp
  src/interpolation/CubicInterpolator.cpp
  src/projection/ProjectionManager.cpp
  src/fusion/PRISMFusion.cpp
)

target_link_libraries(${PROJECT_NAME}_core
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
  TBB::tbb
)

# Node executables
add_executable(prism_fusion_node
  src/nodes/prism_fusion_node.cpp
)

target_link_libraries(prism_fusion_node
  ${PROJECT_NAME}_core
)

# Dependencies for ROS2 components
ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
)

# Installation
install(TARGETS
  ${PROJECT_NAME}_core
  prism_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include/
)

install(DIRECTORY
  config/
  launch/
  DESTINATION share/${PROJECT_NAME}/
)

# Testing configuration
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  
  ament_lint_auto_find_test_dependencies()
  
  # Unit tests
  ament_add_gtest(test_cubic_interpolation
    test/unit/test_cubic_interpolation.cpp
  )
  target_link_libraries(test_cubic_interpolation
    ${PROJECT_NAME}_core
  )
  
  # Benchmark tests
  add_executable(benchmark_interpolation
    test/benchmark/benchmark_interpolation.cpp
  )
  target_link_libraries(benchmark_interpolation
    ${PROJECT_NAME}_core
  )
endif()

ament_package()
```

## Dependencies and Third-Party Library Integration

### Required Dependencies

**ROS2 Core:**
- `rclcpp` - ROS2 C++ client library
- `sensor_msgs` - Point cloud and image message types
- `geometry_msgs` - Transformation message types
- `tf2_ros` - Coordinate transformation utilities

**Computer Vision and Mathematics:**
- `PCL 1.12+` - Point Cloud Library for 3D processing
- `OpenCV 4.5+` - Computer vision and image processing
- `Eigen3 3.3+` - Linear algebra and matrix operations
- `TBB` - Intel Threading Building Blocks for all parallelization (unified concurrency framework)

**Optional Performance Libraries:**
- `Intel IPP` - Integrated Performance Primitives for SIMD
- `CUDA` - GPU acceleration (if available)
- `Benchmarking tools` - Google Benchmark for performance measurement

### Package.xml Configuration
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>prism</name>
  <version>1.0.0</version>
  <description>Point-cloud and RGB Integrated Sensing Module</description>
  
  <maintainer email="dev@prism-project.org">PRISM Development Team</maintainer>
  <license>Apache-2.0</license>
  
  <buildtool_depend>ament_cmake</buildtool_depend>
  
  <!-- ROS2 dependencies -->
  <depend>rclcpp</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_geometry_msgs</depend>
  
  <!-- Computer vision -->
  <depend>pcl_ros</depend>
  <depend>pcl_conversions</depend>
  <depend>cv_bridge</depend>
  <depend>image_transport</depend>
  
  <!-- System dependencies -->
  <exec_depend>libpcl-dev</exec_depend>
  <exec_depend>libopencv-dev</exec_depend>
  <exec_depend>libeigen3-dev</exec_depend>
  <exec_depend>libtbb-dev</exec_depend>
  
  <!-- Testing -->
  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>
  <test_depend>ament_cmake_gtest</test_depend>
  
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Testing Framework Setup

### Unit Testing Structure
```cpp
// test/unit/test_cubic_interpolation.cpp
#include <gtest/gtest.h>
#include "prism/interpolation/CubicInterpolator.hpp"

namespace prism {
namespace test {

class CubicInterpolationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data
        input_cloud_ = generateTestPointCloud();
        interpolator_ = std::make_unique<interpolation::CubicInterpolator>();
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_;
    std::unique_ptr<interpolation::CubicInterpolator> interpolator_;
};

TEST_F(CubicInterpolationTest, BasicInterpolation) {
    auto result = interpolator_->interpolate(*input_cloud_, 2.0f, beam_altitudes_);
    
    EXPECT_GT(result.points.size(), input_cloud_->size());
    EXPECT_LT(result.processing_time.count(), 10000); // 10ms max
}

TEST_F(CubicInterpolationTest, DiscontinuityHandling) {
    auto discontinuous_cloud = generateDiscontinuousCloud();
    auto result = interpolator_->interpolate(*discontinuous_cloud, 3.0f, beam_altitudes_);
    
    // Verify no artifacts at discontinuities
    validateDiscontinuityHandling(result.points);
}

} // namespace test
} // namespace prism
```

### Integration Testing
```cpp
// test/integration/test_fusion_pipeline.cpp
#include <gtest/gtest.h>
#include "prism/fusion/PRISMFusionNode.hpp"

namespace prism {
namespace test {

class FusionIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        rclcpp::init(0, nullptr);
        fusion_node_ = std::make_shared<fusion::PRISMFusionNode>();
    }
    
    void TearDown() override {
        rclcpp::shutdown();
    }
    
    std::shared_ptr<fusion::PRISMFusionNode> fusion_node_;
};

TEST_F(FusionIntegrationTest, EndToEndProcessing) {
    // Simulate sensor inputs
    auto lidar_msg = generateTestPointCloud();
    auto camera_msg = generateTestImage();
    
    // Process through pipeline
    fusion_node_->processLidarData(lidar_msg);
    fusion_node_->processCameraData(camera_msg);
    
    // Verify output
    auto result = fusion_node_->getLatestFusedCloud();
    ASSERT_TRUE(result);
    
    validateColoredPointCloud(*result);
}

} // namespace test
} // namespace prism
```

### Benchmark Testing
```cpp
// test/benchmark/benchmark_interpolation.cpp
#include <benchmark/benchmark.h>
#include "prism/interpolation/CubicInterpolator.hpp"

static void BM_CubicInterpolation_OS132_2x(benchmark::State& state) {
    auto interpolator = prism::interpolation::CubicInterpolator();
    auto input_cloud = generateOS132Cloud(); // 32x1024 points
    
    for (auto _ : state) {
        auto result = interpolator.interpolate(*input_cloud, 2.0f, os132_altitudes);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * input_cloud->size());
    state.SetBytesProcessed(state.iterations() * input_cloud->size() * sizeof(pcl::PointXYZI));
}
BENCHMARK(BM_CubicInterpolation_OS132_2x);

static void BM_ProjectionToCamera(benchmark::State& state) {
    auto projection_mgr = prism::projection::ProjectionManager();
    auto lidar_points = generateTestPoints(state.range(0));
    
    for (auto _ : state) {
        auto projected = projection_mgr.projectToCamera(lidar_points, 0);
        benchmark::DoNotOptimize(projected);
    }
    
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_ProjectionToCamera)->Range(1000, 100000)->Complexity();

BENCHMARK_MAIN();
```

## Configuration Management

### Parameter Files Structure
```yaml
# config/prism_params.yaml
prism_fusion_node:
  ros__parameters:
    # Interpolation settings
    interpolation:
      scale_factor: 3.0
      method: "cubic"  # "linear", "cubic", "adaptive"
      discontinuity_threshold: 0.5
      
    # Projection settings  
    projection:
      camera_count: 2
      intrinsic_file: "config/multi_camera_intrinsic_calibration.yaml"
      extrinsic_file: "config/multi_camera_extrinsic_calibration.yaml"
      projection_threads: 4
      
    # Fusion settings
    fusion:
      color_fusion_method: "weighted_average"  # "nearest", "weighted_average", "confidence_based"
      temporal_window: 0.1  # seconds
      max_projection_distance: 100.0  # meters
      
    # Performance settings
    performance:
      enable_simd: true
      thread_count: -1  # -1 for auto-detect
      memory_pool_size: 1048576  # 1MB
      
    # Quality settings
    quality:
      min_confidence: 0.7
      outlier_removal: true
      smoothing_enabled: false
```

This package structure provides a robust foundation for the PRISM system, with clear separation of concerns, comprehensive testing, and performance optimization opportunities throughout the codebase.
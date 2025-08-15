# PRISM (Point-cloud & RGB Integrated Sensing Module) Architecture Design

## System Architecture Overview

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    PRISM Node                               │
├─────────────────────────────────────────────────────────────┤
│  ROS2 Interface Layer                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  LiDAR Sub  │  │  Cam1 Sub   │  │  Cam2 Sub   │          │
│  │ /ouster/pts │  │ /usb_cam_1  │  │ /usb_cam_2  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                            │                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Temporal Synchronizer                      │    │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────────────────┐   │    │
│  │   │ 20Hz    │ │  30Hz   │ │     Message         │   │    │
│  │   │ Buffer  │ │ Buffer  │ │     Filters         │   │    │
│  │   └─────────┘ └─────────┘ └─────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Processing Pipeline                       │    │
│  │                                                     │    │
│  │  ┌─────────────────┐    ┌─────────────────────┐     │    │
│  │  │ Interpolation   │    │   Calibration       │     │    │
│  │  │    Engine       │    │    Manager          │     │    │
│  │  │  32 → 96        │    │                     │     │    │
│  │  │  Channels       │    │  ┌─────────────┐    │     │    │
│  │  └─────────────────┘    │  │ Intrinsic   │    │     │    │
│  │           │             │  │ Extrinsic   │    │     │    │
│  │           v             │  │ Parameters  │    │     │    │
│  │  ┌─────────────────┐    │  └─────────────┘    │     │    │
│  │  │   Projection    │◄───┘                     │     │    │
│  │  │    Engine       │                          │     │    │
│  │  │ 3D → 2D (Dual)  │                          │     │    │
│  │  └─────────────────┘                          │     │    │
│  │           │                                   │     │    │
│  │           v                                   │     │    │
│  │  ┌─────────────────┐                          │     │    │
│  │  │ Color Extract.  │                          │     │    │
│  │  │    Engine       │                          │     │    │
│  │  │ RGB Sampling    │                          │     │    │
│  │  │ Multi-Cam Fuse  │                          │     │    │
│  │  └─────────────────┘                          │     │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Output Publisher                           │    │
│  │    /ouster/points/colored (PointCloud2+RGB)         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Design
```
Input Stage:          Processing Pipeline:              Output Stage:
                                                    
┌─────────────┐       ┌─────────────────────┐       ┌─────────────────┐
│ LiDAR 20Hz  │──────►│  Interpolation      │       │                 │
│ 32 channels │       │  (32 → 96 ch)       │       │                 │
└─────────────┘       │                     │       │                 │
                      │         │           │       │  Colored        │
┌─────────────┐       │         v           │       │  Point Cloud    │
│ Camera 1    │──────►│  ┌─────────────┐    │──────►│                 │
│ 30Hz        │       │  │ Projection  │    │       │  20Hz Output    │
└─────────────┘       │  │ 3D → 2D     │    │       │                 │
                      │  └─────────────┘    │       │                 │
┌─────────────┐       │         │           │       │                 │
│ Camera 2    │──────►│         v           │       │                 │
│ 30Hz        │       │  ┌─────────────┐    │       └─────────────────┘
└─────────────┘       │  │ Color       │    │       
                      │  │ Extract &   │    │       
                      │  │ Fusion      │    │       
                      │  └─────────────┘    │       
                      └─────────────────────┘       
```

## Detailed Module Design

### 1. Interpolation Engine
**Purpose**: Increase LiDAR point density from 32 to 96 channels using cubic interpolation

**Core Algorithm**:
- Hermite spline interpolation for smooth channel transitions
- Vectorized computation using Eigen arrays
- Integration with existing improved_interpolation_node.cpp logic

**Interface**:
```cpp
class InterpolationEngine {
public:
    struct Config {
        int interpolation_factor = 3;  // 32 → 96
        bool use_simd = true;
    };
    
    void interpolate(const PointCloudSoA& input, 
                    PointCloudSoA& output);
private:
    void hermiteSplineInterpolation(const Eigen::VectorXf& rings);
};
```

### 2. Calibration Manager
**Purpose**: Load and provide calibration data from CALICO format files

**Functionality**:
- Parse YAML calibration files (intrinsic/extrinsic)
- Hot-reload capability for runtime updates
- Thread-safe access to calibration parameters
- Transformation chain management

**Interface**:
```cpp
class CalibrationManager {
public:
    struct CameraCalib {
        Eigen::Matrix3f K;           // Intrinsic matrix
        Eigen::VectorXf dist_coeffs; // Distortion coefficients
        Eigen::Matrix4f T_cam_lidar; // Extrinsic transform
    };
    
    bool loadCalibration(const std::string& intrinsic_path,
                        const std::string& extrinsic_path);
    const CameraCalib& getCameraCalibration(int camera_id) const;
};
```

### 3. Projection Engine
**Purpose**: Project 3D points to 2D camera coordinates with distortion correction

**Key Features**:
- Parallel processing for dual cameras
- Frustum culling optimization
- Sub-pixel precision projection
- Efficient batch processing

**Interface**:
```cpp
class ProjectionEngine {
public:
    struct ProjectionResult {
        std::vector<Eigen::Vector2f> pixel_coords;
        std::vector<bool> valid_projections;
        std::vector<float> depths;
    };
    
    void projectPoints(const PointCloudSoA& points,
                      const CameraCalib& calib,
                      ProjectionResult& result);
};
```

### 4. Color Extraction Engine
**Purpose**: Extract RGB values and handle multi-camera fusion

**Key Features**:
- Bilinear interpolation for sub-pixel sampling
- Distance-weighted multi-camera blending
- Occlusion handling with depth buffering
- Cache-optimized image access patterns

**Interface**:
```cpp
class ColorExtractionEngine {
public:
    struct MultiCameraInput {
        cv::Mat image1, image2;
        ProjectionResult proj1, proj2;
    };
    
    void extractColors(const MultiCameraInput& input,
                      PointCloudSoA& colored_points);
};
```

### 5. Temporal Synchronizer
**Purpose**: Align multi-rate sensor data (20Hz LiDAR, 30Hz cameras)

**Key Features**:
- Adaptive buffering for different frequencies
- Timestamp-based message alignment
- Graceful handling of missing messages
- Configurable synchronization tolerance

## Package Structure

```
src/prism/
├── CMakeLists.txt
├── package.xml
├── include/prism/
│   ├── prism_node.hpp
│   ├── interpolation_engine.hpp
│   ├── calibration_manager.hpp
│   ├── projection_engine.hpp
│   ├── color_extraction_engine.hpp
│   ├── temporal_synchronizer.hpp
│   └── utils/
│       ├── memory_pool.hpp
│       ├── threading_utils.hpp
│       └── point_cloud_utils.hpp
├── src/
│   ├── prism_node.cpp
│   ├── interpolation_engine.cpp
│   ├── calibration_manager.cpp
│   ├── projection_engine.cpp
│   ├── color_extraction_engine.cpp
│   └── temporal_synchronizer.cpp
├── launch/
│   └── prism.launch.py
├── config/
│   ├── prism_params.yaml
│   ├── multi_camera_intrinsic_calibration.yaml
│   └── multi_camera_extrinsic_calibration.yaml
├── test/
│   ├── unit_tests/
│   ├── integration_tests/
│   └── benchmark_tests/
└── docs/
    ├── architecture.md
    ├── package_structure.md
    ├── interpolation_integration.md
    ├── projection_color_mapping.md
    ├── performance_optimization.md
    └── PRISM_PRD.md
```

## Implementation Strategy

### Key Data Structures

**1. Structure-of-Arrays Point Cloud Layout**:
```cpp
struct PointCloudSoA {
    std::vector<float> x, y, z;         // Positions
    std::vector<uint8_t> r, g, b;       // Colors  
    std::vector<float> intensity;       // LiDAR intensity
    std::vector<uint16_t> ring;         // Channel/ring info
    size_t size() const { return x.size(); }
};
```

**2. Memory Pool Management**:
```cpp
class MemoryPool {
    std::vector<PointCloudSoA> pool;
    std::queue<size_t> available_indices;
    std::mutex pool_mutex;
public:
    std::unique_ptr<PointCloudSoA> acquire();
    void release(std::unique_ptr<PointCloudSoA> ptr);
};
```

### Threading Architecture
```
Main Thread:           Worker Threads:
┌─────────────┐       ┌─────────────────┐
│ ROS2        │       │ Interpolation   │
│ Callbacks   │──────►│ Thread Pool     │
│             │       └─────────────────┘
│             │       ┌─────────────────┐
│             │──────►│ Projection      │
│             │       │ Thread Pool     │
│             │       └─────────────────┘
│             │       ┌─────────────────┐
│             │──────►│ Color Extract   │
│             │       │ Thread Pool     │
└─────────────┘       └─────────────────┘
```

## Interface Specifications

### ROS2 Topics
**Inputs**:
- `/ouster/points` (sensor_msgs::PointCloud2) - 32-channel LiDAR at 20Hz
- `/usb_cam_1/image_raw/uncompressed` (sensor_msgs::Image) - Camera 1 at 30Hz
- `/usb_cam_2/image_raw/uncompressed` (sensor_msgs::Image) - Camera 2 at 30Hz

**Outputs**:
- `/ouster/points/colored` (sensor_msgs::PointCloud2 with RGB fields) - Colored point cloud at 20Hz

### ROS2 Parameters
```yaml
prism:
  interpolation_factor: 3                    # 32 → 96 channels
  max_processing_latency_ms: 50             # Target latency
  camera_1_frame_id: "usb_cam_1"           # Camera 1 frame
  camera_2_frame_id: "usb_cam_2"           # Camera 2 frame  
  lidar_frame_id: "os_sensor"              # LiDAR frame
  calibration_intrinsic_file: "path/to/intrinsic.yaml"
  calibration_extrinsic_file: "path/to/extrinsic.yaml"
  threading:
    num_interpolation_threads: 4
    num_projection_threads: 2
    memory_pool_size: 10
```

## Performance Optimization Plan

### 1. Parallelization Strategy
- **TBB parallel_for** for interpolation loops
- **Dual-camera projection** in parallel threads  
- **SIMD vectorization** with Eigen for geometric operations
- **Lock-free ring buffers** for inter-thread communication

### 2. Memory Management
- **Pre-allocated memory pools** for point clouds
- **Stack allocation** for small temporary objects
- **Copy-free message passing** between modules
- **Structure-of-Arrays layout** for vectorization

### 3. Cache Optimization
- **Spatial locality** in point cloud processing
- **Tiled image access patterns** for color extraction
- **Data prefetching** for predictable access patterns
- **Memory alignment** for SIMD operations

### 4. Algorithm Optimizations
- **Frustum culling** to eliminate out-of-view points early
- **Hierarchical processing** for large point clouds  
- **Adaptive interpolation** based on point density
- **Early termination** for occluded points

## Development Roadmap

### Phase 1: Infrastructure Foundation
**Milestones**:
1. Package structure setup with CMakeLists.txt configuration
2. Threading framework implementation using TBB
3. Memory pool and RAII wrapper development
4. ROS2 node skeleton with message handling

**Dependencies**: ROS2 Humble, Eigen, TBB, OpenCV, PCL

### Phase 2: Core Module Development
**Milestones**:
1. Integration and optimization of interpolation engine
2. CALICO calibration file parser implementation
3. Projection engine with dual-camera support
4. Color extraction engine with multi-camera fusion

**Dependencies**: Phase 1 completion, access to calibration files

### Phase 3: Integration and Optimization
**Milestones**:
1. Temporal synchronization implementation
2. End-to-end pipeline integration and testing
3. SIMD optimization and performance tuning
4. Memory usage optimization and leak detection

**Dependencies**: Phase 2 completion, real sensor data access

### Phase 4: Validation and Documentation
**Milestones**:
1. Comprehensive unit and integration testing
2. Performance validation against latency targets
3. Technical documentation completion
4. Code review and final optimizations

**Dependencies**: Phase 3 completion, test environment setup

### Critical Path Analysis
```
Interpolation Engine → Projection Engine → Color Extraction → Full Integration
        ↓                     ↓                   ↓               ↓
   (Week 3-4)           (Week 4-5)         (Week 5)        (Week 6-7)
```

## Risk Mitigation Strategies

### Technical Risks
1. **Latency Target Miss**: Implement progressive optimization with fallback algorithms
2. **Memory Constraints**: Use adaptive memory pools with configurable limits
3. **Threading Complexity**: Extensive unit testing with thread sanitizers
4. **Calibration Integration**: Flexible parser supporting multiple CALICO formats

### Performance Risks
1. **CPU Usage Spike**: Implement adaptive thread count based on system load
2. **Memory Fragmentation**: Use fixed-size memory pools with defragmentation
3. **Cache Misses**: Profile-guided optimization with performance counters

This comprehensive architecture provides a robust foundation for real-time multi-modal sensor fusion while maintaining the flexibility for future enhancements and optimizations. The modular design enables independent development and testing of components while ensuring optimal performance through careful threading and memory management strategies.